import os
import cv2
import random
import numpy as np
import torch
from torch.backends import cudnn
import torch.nn.functional as F
from PIL import Image
from math import exp
from skimage.metrics import structural_similarity as ssim_skimage
from torchvision import transforms

def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))

    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_image(out_dir, x, num, epoch, filename=None):
    if isinstance (x, tuple):
     x = x[0]
    img = (x * 255).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous().cpu()[0]
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename+'.tif')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.tif'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img)).save(test_path)


def save_image_2(out_dir, x, num, epoch, filename=None):
    # 如果 x 是元组，只取第一个元素
    if isinstance (x, tuple):
        x = x[0]

    # 检查 x 是否有 4 个维度，如果只有 3 个维度（例如 [C, H, W]），添加一个 batch 维度
    if len (x.shape) == 3:
        x = x.unsqueeze (0)  # 添加一个维度，变成 [1, C, H, W]

    img = (x * 255).clamp (0, 255).to (torch.uint8)

    # permute 用于调整维度顺序：从 [batch_size, channels, height, width] -> [batch_size, height, width, channels]
    img = img.permute (0, 2, 3, 1)  # 确保维度是 [batch_size, height, width, channels]
    img = img.contiguous ( ).cpu ( )[0]  # 取第一个图像（假设 batch_size = 1）

    test_dir = os.path.join (out_dir, 'epoch_{0:04d}'.format (epoch))
    if filename is not None:
        test_path = os.path.join (test_dir, filename + '.tif')
    else:
        test_path = os.path.join (test_dir, 'test_{0:04d}.tif'.format (num))

    if not os.path.exists (test_dir):
        os.makedirs (test_dir)

    # 保存图像
    Image.fromarray (np.uint8 (img)).save (test_path)


def save_image_1(out_dir, x, num, epoch, filename=None):
    img = (x * 255).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)[0]
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename + '.png')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img)).save(test_path)

def save_imagenir(out_dir, x, num, epoch, filename=None):
    img = (x*255).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)[0]
    w,h,c =img.shape
    nir = img[:,:,3]
    nir = nir.expand(3, w,h)
    nir = nir.permute( 1, 2, 0)
    img = img.contiguous().cpu()
    nir = nir.contiguous().cpu()
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename+'.png')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))
    if filename is not None:
        test_path_nir= os.path.join(test_dir, filename+'nir.png')
    else:
        test_path_nir = os.path.join(test_dir, 'test_{0:04d}nir.png'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img[:,:,0:3])).save(test_path)
    Image.fromarray(np.uint8(nir)).save(test_path_nir)


def save_13tif(out_dir, x, num, epoch, filename=None):
    img = (x*255).clamp(0, 255).to(torch.uint8)
    img = img[0]
    c,w,h =img.shape


    # img = img.contiguous().cpu()

    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path1 = os.path.join(test_dir, filename+'.tif')
        test_path4 = os.path.join(test_dir, filename+'_band4.tif')
        test_path5 = os.path.join(test_dir, filename + '_band5.tif')
        test_path6 = os.path.join(test_dir, filename + '_band6.tif')
        test_path7 = os.path.join(test_dir, filename + '_band7.tif')
        test_path8 = os.path.join(test_dir, filename + '_band8.tif')
        test_path8A = os.path.join(test_dir, filename + '_band8A.tif')
        test_path9 = os.path.join(test_dir, filename + '_band9.tif')
        test_path10 = os.path.join(test_dir, filename + '_band10.tif')
        test_path11 = os.path.join(test_dir, filename + '_band11.tif')
        test_path12 = os.path.join(test_dir, filename + '_band12.tif')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.tif'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img[[2,1,0],:,:].permute(2,1,0).cpu())).save(test_path1)
    image4 = img[:3, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image5 = img[4, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image6 = img[5, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image7 = img[6, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image8 = img[7, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image8A = img[8, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image9 = img[9, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image10 = img[10, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image11 = img[11, :, :].expand(3,w,h).permute(2,1,0).cpu()
    image12 = img[12, :, :].expand(3,w,h).permute(2,1,0).cpu()

    Image.fromarray(np.uint8(image4)).save(test_path4)
    Image.fromarray(np.uint8(image5)).save(test_path5)
    Image.fromarray(np.uint8(image6)).save(test_path6)
    Image.fromarray(np.uint8(image7)).save(test_path7)
    Image.fromarray(np.uint8(image8)).save(test_path8)
    Image.fromarray(np.uint8(image8A)).save(test_path8A)
    Image.fromarray(np.uint8(image9)).save(test_path9)
    Image.fromarray(np.uint8(image10)).save(test_path10)
    Image.fromarray(np.uint8(image11)).save(test_path11)
    Image.fromarray(np.uint8(image12)).save(test_path12)
def save_4tif(out_dir, x, num, epoch, filename=None):
    img = (x * 255).clamp(0, 255).to(torch.uint8)
    img = img[0]
    c, h, w = img.shape

    # img = img.contiguous().cpu()

    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path1 = os.path.join(test_dir, filename + '.tif')
        test_path4 = os.path.join(test_dir, filename + '_band4.tif')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.tif'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img[[2, 1, 0], :, :].permute(2, 1, 0).cpu())).save(test_path1)
    image4 = img[:3, :, :].expand(3, h, w).permute(2, 1, 0).cpu()
    Image.fromarray(np.uint8(image4)).save(test_path4)



def save_10tif(out_dir, x, num, epoch, filename=None):
    img = (x*255).clamp(0, 255).to(torch.uint8)
    img = img[0]
    c,h,w =img.shape


    # img = img.contiguous().cpu()

    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path1 = os.path.join(test_dir, filename+'.tif')
        test_path4 = os.path.join(test_dir, filename+'_band4.tif')
        test_path5 = os.path.join(test_dir, filename + '_band5.tif')
        test_path6 = os.path.join(test_dir, filename + '_band6.tif')
        test_path7 = os.path.join(test_dir, filename + '_band7.tif')
        test_path8 = os.path.join(test_dir, filename + '_band8.tif')
        test_path8A = os.path.join(test_dir, filename + '_band8A.tif')
        test_path9 = os.path.join(test_dir, filename + '_band9.tif')
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.tif'.format(num))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    Image.fromarray(np.uint8(img[[2,1,0],:,:].permute(2,1,0).cpu())).save(test_path1)
    image4 = img[:3, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image5 = img[4, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image6 = img[5, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image7 = img[6, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image8 = img[7, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image8A = img[8, :, :].expand(3,h,w).permute(2,1,0).cpu()
    image9 = img[9, :, :].expand(3,h,w).permute(2,1,0).cpu()

    Image.fromarray(np.uint8(image4)).save(test_path4)
    Image.fromarray(np.uint8(image5)).save(test_path5)
    Image.fromarray(np.uint8(image6)).save(test_path6)
    Image.fromarray(np.uint8(image7)).save(test_path7)
    Image.fromarray(np.uint8(image8)).save(test_path8)
    Image.fromarray(np.uint8(image8A)).save(test_path8A)
    Image.fromarray(np.uint8(image9)).save(test_path9)





'''def checkpoint(config, epoch, gen, dis=None):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    net_gen_model_out_path = os.path.join(model_dir, 'gen_model_epoch_{}.pth'.format(epoch))
    torch.save(gen.state_dict(), net_gen_model_out_path)
    if(dis):
         net_dis_model_out_path = os.path.join(model_dir, 'dis_model_epoch_{}.pth'.format(epoch))
         torch.save(dis.state_dict(), net_dis_model_out_path)
    print("Checkpoint saved to {}".format(model_dir))'''


def checkpoint(config, epoch, gen, dis=None):
    model_dir = os.path.join(config.out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # 保存生成器
    gen_path = os.path.join(model_dir, f'gen_{epoch}.pth')
    torch.save(gen.state_dict(), gen_path)
    print(f'Saved generator checkpoint: {gen_path}')

    # 保存判别器（如果存在）
    if dis is not None:
        dis_path = os.path.join(model_dir, f'dis_{epoch}.pth')
        torch.save(dis.state_dict(), dis_path)
        print(f'Saved discriminator checkpoint: {dis_path}')

    # 保存最新模型
    latest_path = os.path.join(model_dir, 'latest.pth')
    torch.save({
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict() if dis else None,
        'epoch': epoch
    }, latest_path)
    print(f'Saved latest checkpoint: {latest_path}')


def load_checkpoint(config, gen, dis=None):
    model_dir = os.path.join(config.out_dir, 'models')
    latest_path = os.path.join(model_dir, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])

        if dis and checkpoint['dis_state_dict']:
            dis.load_state_dict(checkpoint['dis_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        print(f'Loaded checkpoint from epoch {epoch}')
        return epoch
    else:
        print('No checkpoint found. Starting from scratch.')
        return 0

#以上新增


def make_manager():
    if not os.path.exists('.job'):
        os.makedirs('.job')
        with open('.job/job.txt', 'w', encoding='UTF-8') as f:
            f.write('0')


def job_increment():
    with open('.job/job.txt', 'r', encoding='UTF-8') as f:
        n_job = f.read()
        n_job = int(n_job)
    with open('.job/job.txt', 'w', encoding='UTF-8') as f:
        f.write(str(n_job + 1))
    
    return n_job

def heatmap(img):
    if len(img.shape) == 3:
        b,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        b,c,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
    return heat

def save_attention_as_heatmap(filename, att):
    att_heat = heatmap(att)
    cv2.imwrite(filename, att_heat)
    print(filename, 'saved')

#新增
def calculate_ssim(img1, img2, data_range=1.0):
    """
    使用PyTorch计算SSIM (GPU加速)
    参数:
        img1 (torch.Tensor): 第一个图像张量, shape [N, C, H, W]
        img2 (torch.Tensor): 第二个图像张量, shape [N, C, H, W]
        data_range (float): 图像数据的范围
    返回:
        float: 平均 SSIM 值
    """
    # 确保输入是4维张量
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # 获取图像尺寸
    _, _, h, w = img1.size()

    # 根据图像尺寸动态确定窗口大小
    win_size = min(11, min(h, w))
    if win_size % 2 == 0:  # 确保是奇数
        win_size = max(7, win_size - 1)
    if win_size < 7:
        win_size = 7  # 最小窗口大小

    # 确定值范围
    L = data_range

    # 获取图像通道数
    channel = img1.size(1)

    # 创建高斯窗口
    window = create_window(win_size, channel).to(img1.device)

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=win_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=win_size // 2, groups=channel)

    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=win_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=win_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2, groups=channel) - mu1_mu2

    # SSIM常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # SSIM公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    return ssim_map.mean().item()


def gaussian(window_size, sigma):
    """生成高斯分布权重"""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    """创建2D高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 其他工具函数...
def save_images(real_a, real_b, fake_b, save_dir, index):
    """保存真实和生成的图像"""
    os.makedirs(save_dir, exist_ok=True)

    # 转换为PIL图像并保存
    to_pil = transforms.ToPILImage()

    # 保存输入图像
    input_img = to_pil(real_a[0].cpu())
    input_img.save(os.path.join(save_dir, f"{index}_input.png"))

    # 保存真实目标图像
    target_img = to_pil(real_b[0].cpu())
    target_img.save(os.path.join(save_dir, f"{index}_target.png"))

    # 保存生成图像
    gen_img = to_pil(fake_b[0].cpu())
    gen_img.save(os.path.join(save_dir, f"{index}_generated.png"))