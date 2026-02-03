import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from dataload.init import getdata
import utils.utils as utils
from utils.utils import gpu_manage, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
import numpy as np
# from .losses import CharbonnierLoss, EdgeLoss
# from model.mn import MPRNet

"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""
from pdb import set_trace as stx

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d (
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer (nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super (CALayer, self).__init__ ( )
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d (1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential (
            nn.Conv2d (channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU (inplace=True),
            nn.Conv2d (channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid ( )
        )

    def forward(self, x):
        y = self.avg_pool (x)
        y = self.conv_du (y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB (nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super (CAB, self).__init__ ( )
        modules_body = []
        modules_body.append (conv (n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append (act)
        modules_body.append (conv (n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer (n_feat, reduction, bias=bias)
        self.body = nn.Sequential (*modules_body)

    def forward(self, x):
        res = self.body (x)
        res = self.CA (res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM (nn.Module):
    def __init__(self, n_feat, kernel_size, bias, channel):
        super (SAM, self).__init__ ( )
        self.conv1 = conv (n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv (n_feat, channel, kernel_size, bias=bias)
        self.conv3 = conv (channel, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1 (x)
        img = self.conv2 (x) + x_img
        x2 = torch.sigmoid (self.conv3 (img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder (nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super (Encoder, self).__init__ ( )

        self.encoder_level1 = [CAB (n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range (2)]
        self.encoder_level2 = [CAB (n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range (2)]
        self.encoder_level3 = [CAB (n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range (2)]

        self.encoder_level1 = nn.Sequential (*self.encoder_level1)
        self.encoder_level2 = nn.Sequential (*self.encoder_level2)
        self.encoder_level3 = nn.Sequential (*self.encoder_level3)

        self.down12 = DownSample (n_feat, scale_unetfeats)
        self.down23 = DownSample (n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d (n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d (n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d (n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                        bias=bias)

            self.csff_dec1 = nn.Conv2d (n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d (n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d (n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                        bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1 (x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1 (encoder_outs[0]) + self.csff_dec1 (decoder_outs[0])

        x = self.down12 (enc1)

        enc2 = self.encoder_level2 (x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2 (encoder_outs[1]) + self.csff_dec2 (decoder_outs[1])

        x = self.down23 (enc2)

        enc3 = self.encoder_level3 (x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3 (encoder_outs[2]) + self.csff_dec3 (decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder (nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super (Decoder, self).__init__ ( )

        self.decoder_level1 = [CAB (n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range (2)]
        self.decoder_level2 = [CAB (n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range (2)]
        self.decoder_level3 = [CAB (n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range (2)]

        self.decoder_level1 = nn.Sequential (*self.decoder_level1)
        self.decoder_level2 = nn.Sequential (*self.decoder_level2)
        self.decoder_level3 = nn.Sequential (*self.decoder_level3)

        self.skip_attn1 = CAB (n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB (n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample (n_feat, scale_unetfeats)
        self.up32 = SkipUpSample (n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3 (enc3)

        x = self.up32 (dec3, self.skip_attn2 (enc2))
        dec2 = self.decoder_level2 (x)

        x = self.up21 (dec2, self.skip_attn1 (enc1))
        dec1 = self.decoder_level1 (x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample (nn.Module):
    def __init__(self, in_channels, s_factor):
        super (DownSample, self).__init__ ( )
        self.down = nn.Sequential (nn.Upsample (scale_factor=0.5, mode='bilinear', align_corners=False),
                                   nn.Conv2d (in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down (x)
        return x


class UpSample (nn.Module):
    def __init__(self, in_channels, s_factor):
        super (UpSample, self).__init__ ( )
        self.up = nn.Sequential (nn.Upsample (scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d (in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up (x)
        return x


class SkipUpSample (nn.Module):
    def __init__(self, in_channels, s_factor):
        super (SkipUpSample, self).__init__ ( )
        self.up = nn.Sequential (nn.Upsample (scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d (in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up (x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB (nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super (ORB, self).__init__ ( )
        modules_body = []
        modules_body = [CAB (n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range (num_cab)]
        modules_body.append (conv (n_feat, n_feat, kernel_size))
        self.body = nn.Sequential (*modules_body)

    def forward(self, x):
        res = self.body (x)
        res += x
        return res


##########################################################################
class ORSNet (nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super (ORSNet, self).__init__ ( )

        self.orb1 = ORB (n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB (n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB (n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample (n_feat, scale_unetfeats)
        self.up_dec1 = UpSample (n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential (UpSample (n_feat + scale_unetfeats, scale_unetfeats),
                                      UpSample (n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential (UpSample (n_feat + scale_unetfeats, scale_unetfeats),
                                      UpSample (n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d (n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1 (x)
        x = x + self.conv_enc1 (encoder_outs[0]) + self.conv_dec1 (decoder_outs[0])

        x = self.orb2 (x)
        x = x + self.conv_enc2 (self.up_enc1 (encoder_outs[1])) + self.conv_dec2 (self.up_dec1 (decoder_outs[1]))

        x = self.orb3 (x)
        x = x + self.conv_enc3 (self.up_enc2 (encoder_outs[2])) + self.conv_dec3 (self.up_dec2 (decoder_outs[2]))

        return x


##########################################################################
class MPRNet (nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super (MPRNet, self).__init__ ( )

        act = nn.PReLU ( )
        self.shallow_feat1 = nn.Sequential (conv (in_c, n_feat, kernel_size, bias=bias),
                                            CAB (n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential (conv (in_c, n_feat, kernel_size, bias=bias),
                                            CAB (n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential (conv (in_c, n_feat, kernel_size, bias=bias),
                                            CAB (n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder (n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder (n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder (n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder (n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet (n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                     num_cab)

        self.sam12 = SAM (n_feat, kernel_size=1, bias=bias, channel=in_c)
        self.sam23 = SAM (n_feat, kernel_size=1, bias=bias, channel=in_c)

        self.concat12 = conv (n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv (n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv (n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img, isTrain=False):
        # Original-resolution Image for Stage 3
        H = x3_img.size (2)
        W = x3_img.size (3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int (H / 2), :]
        x2bot_img = x3_img[:, :, int (H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int (W / 2)]
        x1rtop_img = x2top_img[:, :, :, int (W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int (W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int (W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1 (x1ltop_img)
        x1rtop = self.shallow_feat1 (x1rtop_img)
        x1lbot = self.shallow_feat1 (x1lbot_img)
        x1rbot = self.shallow_feat1 (x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder (x1ltop)
        feat1_rtop = self.stage1_encoder (x1rtop)
        feat1_lbot = self.stage1_encoder (x1lbot)
        feat1_rbot = self.stage1_encoder (x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat ((k, v), 3) for k, v in zip (feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat ((k, v), 3) for k, v in zip (feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder (feat1_top)
        res1_bot = self.stage1_decoder (feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12 (res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12 (res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat ([stage1_img_top, stage1_img_bot], 2)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2 (x2top_img)
        x2bot = self.shallow_feat2 (x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12 (torch.cat ([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12 (torch.cat ([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder (x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder (x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat ((k, v), 2) for k, v in zip (feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder (feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23 (res2[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3 (x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23 (torch.cat ([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet (x3_cat, feat2, res2)

        stage3_img = self.tail (x3_cat)
        n = [stage3_img + x3_img, stage2_img, stage1_img]
        if (isTrain):
            return n
        else:
            return 1, torch.clamp (n[0], 0, 1)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x[:,:3,:,:]), self.laplacian_kernel(y[:,:3,:,:]))
        return loss

def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    train_dataset, validation_dataset = getdata(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = MPRNet(in_c=config.in_ch, out_c=config.in_ch)
    print('Total params: %.2fM' % (sum(p.numel() for p in gen.parameters())/1000000.0))


    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))



    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    

    gen = gen.cuda()
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterionMSE = nn.MSELoss()

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a = Variable(real_a_cpu).cuda()
            real_b = Variable(real_b_cpu).cuda()
            M = Variable(M_cpu).cuda()

            opt_gen.zero_grad()

            restored = gen(real_a,isTrain=True)
    
            # Compute loss at each stage
            loss_char = sum(criterion_char(restored[j],real_b) for j in range(len(restored)))
            loss_edge = sum([criterion_edge(restored[j],real_b) for j in range(len(restored))])
            loss = (loss_char) + (0.05*loss_edge)
            
            loss.backward()
            opt_gen.step()



        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0 and epoch > 50:
            checkpoint(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)
