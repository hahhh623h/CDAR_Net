# train_CDAR_Net.py
'''
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

# 从新模型文件中导入
from model.msda.CDAR_Net import CDAR_Net_Generator, CDAR_Loss, compute_gradient_penalty

from dataload.init import getdata
from log_report import LogReport, TestReport
# from .perceptual import LossNetwork # 如果使用感知损失，请确保此路径正确
from torchvision.models import vgg16


# 判别器定义保持不变
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


def train(config):
    # GPU管理
    device = torch.device("cuda" if config.gpu_ids != -1 else "cpu")
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)
        cudnn.benchmark = True

    # 数据集加载
    print('===> Loading datasets')
    train_dataset, validation_dataset = getdata(config)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)

    # 模型加载
    print('===> Loading models')
    # 使用新的生成器
    gen = CDAR_Net_Generator(height=config.height, width=config.width, num_dense_layer=config.num_dense_layer,
                             growth_rate=config.growth_rate, in_channels=config.in_ch)
    dis = Discriminator(input_nc=config.in_ch)

    if config.gen_init:
        gen.load_state_dict(torch.load(config.gen_init))
        print(f'Loaded generator from {config.gen_init}')
    if config.dis_init:
        dis.load_state_dict(torch.load(config.dis_init))
        print(f'Loaded discriminator from {config.dis_init}')

    # 损失函数
    # vgg_model = vgg16(pretrained=True).features[:16].to(device).eval()
    # for param in vgg_model.parameters(): param.requires_grad = False
    # loss_network = LossNetwork(vgg_model)
    cdar_loss = CDAR_Loss(task_weight=1.0, domain_weight=config.beta, mcam_mask_weight=config.gamma,
                          spectral_weight=config.delta, consistency_weight=config.delta)

    # 优化器
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    # 设备转移
    gen.to(device)
    dis.to(device)
    cdar_loss.to(device)

    # 日志报告
    logreport = LogReport(log_dir=config.out_dir)

    print('===> Training start')
    for epoch in range(1, config.epoch + 1):
        for iteration, batch in enumerate(training_data_loader, 1):
            # M_cpu是真实的云掩膜
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a = real_a_cpu.to(device)  # 有云图像
            real_b = real_b_cpu.to(device)  # 无云真值
            M = M_cpu.to(device).unsqueeze(1)  # 云掩膜真值

            # ===== 1. 训练判别器 =====
            opt_dis.zero_grad()

            # 真实样本损失
            d_out_real = dis(real_b)
            loss_d_real = -torch.mean(d_out_real)

            # 生成样本损失
            with torch.no_grad():
                # **核心改动**: gen不再需要M作为输入
                fake_b, _, _ = gen(real_a)
            fake_b = fake_b.detach()
            d_out_fake = dis(fake_b)
            loss_d_fake = torch.mean(d_out_fake)

            # 梯度惩罚
            gp = compute_gradient_penalty(dis, real_b, fake_b)

            # 总判别损失
            total_dis_loss = loss_d_real + loss_d_fake + config.gp_weight * gp
            total_dis_loss.backward()
            opt_dis.step()

            # ===== 2. 训练生成器 =====
            if iteration % 5 == 0:  # 判别器每训练5次，生成器训练1次
                opt_gen.zero_grad()

                # **核心改动**: gen的forward返回多个值
                # 注意：这里为了计算consistency loss，需要获取CAFF模块前后的特征
                # 这需要对CDAR_Net_Generator做微小修改，使其能返回这些中间特征
                # 为简化，我们暂时忽略consistency_loss，或假设其为0
                fake_b, pred_mask, domain_logits = gen(real_a)

                # 计算生成器对抗损失
                d_out_gen = dis(fake_b)
                loss_g_adv = -torch.mean(d_out_gen)

                # 计算重建损失和掩膜损失
                # 为简化，暂时不传入caff特征
                loss_task = cdar_loss.task_loss_l1(fake_b, real_b)
                loss_mcam = cdar_loss.cloud_mask_loss(pred_mask, M)
                loss_spec = cdar_loss.spectral_loss(fake_b * (1 - M), real_a * (1 - M))

                # 计算总生成器损失
                total_gen_loss = (cdar_loss.task_weight * loss_task +
                                  cdar_loss.domain_weight * loss_g_adv +
                                  cdar_loss.mcam_mask_weight * loss_mcam +
                                  cdar_loss.spectral_weight * loss_spec)

                total_gen_loss.backward()
                opt_gen.step()

                # ===== 日志记录 =====
                if iteration % config.print_every == 0:
                    print(f"Epoch[{epoch}]({iteration}/{len(training_data_loader)}): "
                          f"G_Loss: {total_gen_loss.item():.4f} | D_Loss: {total_dis_loss.item():.4f} | "
                          f"Task: {loss_task.item():.4f} | Adv: {loss_g_adv.item():.4f} | "
                          f"Mask: {loss_mcam.item():.4f} | Spectral: {loss_spec.item():.4f}")
                    log = {
                        'epoch': epoch, 'iteration': iteration,
                        'loss/gen_total': total_gen_loss.item(), 'loss/dis_total': total_dis_loss.item(),
                        'loss/task': loss_task.item(), 'loss/adv': loss_g_adv.item(),
                        'loss/mcam_mask': loss_mcam.item(), 'loss/spectral': loss_spec.item()
                    }
                    logreport(log)

        # 保存模型等...
        if epoch % config.snapshot_interval == 0:
            gen_path = os.path.join(config.out_dir, f'gen_epoch_{epoch}.pth')
            dis_path = os.path.join(config.out_dir, f'dis_epoch_{epoch}.pth')
            torch.save(gen.state_dict(), gen_path)
            torch.save(dis.state_dict(), dis_path)
            print(f'Checkpoint saved to {gen_path} and {dis_path}')

        logreport.save_lossgraph()'''

# train_CDAR_Net.py (Ablation Study Version)

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

#from model.msda.CDAR_Net import Generator_MCAM_Only, Generator_MCAM_CAFF, CDAR_Net_Generator, compute_gradient_penalty
#from dataload.init import getdata
#from log_report import LogReport

'''
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


def train(config):
    device = torch.device("cuda" if config.gpu_ids != -1 else "cpu")
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)
        cudnn.benchmark = True

    train_dataset, _ = getdata(config)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)

    print(f"===> Loading Ablation Mode: {config.ablation_mode}")
    gen_args = {'height': config.height, 'width': config.width, 'num_dense_layer': config.num_dense_layer,
                'growth_rate': config.growth_rate, 'in_channels': config.in_ch}

    if config.ablation_mode == 'mcam_only':
        gen = Generator_MCAM_Only(**gen_args)
    elif config.ablation_mode == 'mcam_caff':
        gen = Generator_MCAM_CAFF(**gen_args)
    elif config.ablation_mode == 'full':
        gen = CDAR_Net_Generator(**gen_args)
    else:
        raise ValueError(f"Unknown ablation_mode: {config.ablation_mode}")

    dis = Discriminator(input_nc=config.in_ch)

    if config.gen_init: gen.load_state_dict(torch.load(config.gen_init))
    if config.dis_init: dis.load_state_dict(torch.load(config.dis_init))

    # 根据消融模式确定是否需要对抗损失
    use_gan_loss = (config.ablation_mode == 'full')

    task_loss_l1 = nn.L1Loss().to(device)
    cloud_mask_loss = nn.BCELoss().to(device)
    spectral_loss = nn.L1Loss().to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    if use_gan_loss:
        opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    gen.to(device)
    dis.to(device)
    logreport = LogReport(log_dir=config.out_dir)

    print('===> Training start')
    for epoch in range(1, config.epoch + 1):
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b, M_gt = batch[0].to(device), batch[1].to(device), batch[2].to(device).unsqueeze(1)

            # --- 判别器训练 (仅在完整模式下) ---
            if use_gan_loss:
                opt_dis.zero_grad()
                d_out_real = dis(real_b)
                loss_d_real = -torch.mean(d_out_real)
                with torch.no_grad():
                    fake_b, _, _ = gen(real_a)
                d_out_fake = dis(fake_b.detach())
                loss_d_fake = torch.mean(d_out_fake)
                gp = compute_gradient_penalty(dis, real_b, fake_b.detach())
                total_dis_loss = loss_d_real + loss_d_fake + config.gp_weight * gp
                total_dis_loss.backward()
                opt_dis.step()

            # --- 生成器训练 ---
            if iteration % (5 if use_gan_loss else 1) == 0:
                opt_gen.zero_grad()
                fake_b, pred_mask, domain_logits = gen(real_a)

                loss_task = config.alpha * task_loss_l1(fake_b, real_b)
                loss_mcam = config.gamma * cloud_mask_loss(pred_mask, M_gt)
                non_cloud_mask = 1 - M_gt
                loss_spec = config.delta * spectral_loss(fake_b * non_cloud_mask, real_a * non_cloud_mask)

                total_gen_loss = loss_task + loss_mcam + loss_spec

                loss_g_adv = torch.tensor(0.0)
                if use_gan_loss:
                    d_out_gen = dis(fake_b)
                    loss_g_adv = config.beta * -torch.mean(d_out_gen)
                    total_gen_loss += loss_g_adv

                total_gen_loss.backward()
                opt_gen.step()

                if iteration % config.print_every == 0:
                    log_msg = f"E[{epoch}]({iteration}): G_Loss:{total_gen_loss.item():.4f} [Task:{loss_task.item():.4f}, Mask:{loss_mcam.item():.4f}, Spec:{loss_spec.item():.4f}]"
                    if use_gan_loss: log_msg += f" [Adv:{loss_g_adv.item():.4f}, D_Loss:{total_dis_loss.item():.4f}]"
                    print(log_msg)
                    log = {'epoch': epoch, 'iteration': iteration, 'loss/gen_total': total_gen_loss.item(),
                           'loss/task': loss_task.item(), 'loss/mcam_mask': loss_mcam.item(),
                           'loss/spectral': loss_spec.item()}
                    if use_gan_loss: log.update(
                        {'loss/adv': loss_g_adv.item(), 'loss/dis_total': total_dis_loss.item()})
                    logreport(log)

        if epoch % config.snapshot_interval == 0:
            torch.save(gen.state_dict(), os.path.join(config.out_dir, f'gen_epoch_{epoch}.pth'))
            if use_gan_loss: torch.save(dis.state_dict(), os.path.join(config.out_dir, f'dis_epoch_{epoch}.pth'))
        logreport.save_lossgraph()'''

# train_CDAR_Net.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
from model.CDAR_Net.CDAR_Net import (
    Generator_Baseline, Generator_MCAM, Generator_CAFF, Generator_CCAR,
    Generator_MCAM_CAFF, Generator_MCAM_CCAR, CDAR_Net_Generator, Generator_CAFF_CCAR,
    compute_gradient_penalty,
    CloudAwareFusion
)
from dataload.init import getdata
from log_report import LogReport
from torchvision.models import vgg16, VGG16_Weights

# 判别器定义 (保持不变)
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


def train(config):
    # --- 1. 环境和设备设置 ---
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.manualSeed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.manualSeed)
        cudnn.benchmark = True

    # --- 2. 数据加载 ---
    print('===> Loading datasets')
    train_dataset, _ = getdata(config)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)
    print(f"Training dataset size: {len(train_dataset)}")

    # --- 3. 模型选择和加载 ---
    ablation_models = {
        "baseline": Generator_Baseline,
        "baseline_mcam": Generator_MCAM,
        "baseline_caff": Generator_CAFF,
        "baseline_ccar": Generator_CCAR,
        "baseline_mcam_caff": Generator_MCAM_CAFF,
        "baseline_mcam_ccar": Generator_MCAM_CCAR,
        "baseline_caff_ccar": Generator_CAFF_CCAR,
        "full": CDAR_Net_Generator,
    }

    print(f"===> Loading Ablation Mode: {config.ablation_mode}")
    gen_class = ablation_models.get(config.ablation_mode)
    if not gen_class:
        raise ValueError(f"Unknown ablation_mode: {config.ablation_mode}")

    gen_args = {'height': config.height, 'width': config.width, 'num_dense_layer': config.num_dense_layer,
                'growth_rate': config.growth_rate, 'in_channels': config.in_ch}
    gen = gen_class(**gen_args)
    # 动态设置 CAFF 模块的阈值
    if hasattr(gen, 'caff') and isinstance(gen.caff, CloudAwareFusion):
        gen.caff.threshold = config.caff_threshold
        print(f"Set CAFF threshold to: {config.caff_threshold}")

    # 根据模式决定是否需要判别器
    #use_gan = 'ccar' in config.ablation_mode or 'full' in config.ablation_mode
    loss_mode = getattr(config, 'loss_mode', 'task_adv_cons')
    use_gan = ('adv' in loss_mode)
    dis = Discriminator(input_nc=config.in_ch) if use_gan else None

    if config.gen_init and os.path.exists(config.gen_init):
        gen.load_state_dict(torch.load(config.gen_init, map_location=device))
        print(f"Loaded generator from {config.gen_init}")
    if use_gan and config.dis_init and os.path.exists(config.dis_init):
        dis.load_state_dict(torch.load(config.dis_init, map_location=device))
        print(f"Loaded discriminator from {config.dis_init}")

    # --- 4. 损失函数和优化器 ---
    task_loss_l1 = nn.L1Loss().to(device)
    cloud_mask_loss = nn.BCELoss().to(device)
    spectral_loss = nn.L1Loss().to(device)
    consistency_loss = nn.MSELoss().to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999)) if use_gan else None


    gen.to(device)
    if use_gan: dis.to(device)

    vgg = None
    if hasattr(config, 'use_perceptual') and config.use_perceptual:
        weights = VGG16_Weights.DEFAULT
        vgg_full = vgg16(weights=weights).features.to(device).eval()
        vgg_layers = getattr(config, 'vgg_layers', 16)
        vgg = torch.nn.Sequential(*list(vgg_full.children())[:vgg_layers]).to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

    def perceptual_loss_vgg(x, y):
        """
        x,y: (B,3,H,W), 建议范围[-1,1]或[0,1]均可，但两者一致即可
        用 L1 对齐特征
        """
        if vgg is None:
            return torch.tensor(0.0, device=device)
        fx = vgg(x)
        fy = vgg(y)
        return torch.mean(torch.abs(fx - fy))


    # --- 5. 日志和训练 ---
    logreport = LogReport(log_dir=config.out_dir)
    print('===> Training start')

    for epoch in range(1, config.epoch + 1):
        gen.train()
        if use_gan: dis.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            # 假设数据集返回 (有云图, 无云真值, 云掩膜真值)
            real_a, real_b, M_gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # 确保 M_gt 有通道维度
            if M_gt.dim() == 3: M_gt = M_gt.unsqueeze(1)

            # --- 判别器训练 (仅在需要时) ---
            if use_gan:
                opt_dis.zero_grad()

                # 使用 no_grad 生成 fake_b 以避免梯度流入生成器
                with torch.no_grad():
                    fake_b, _, _, _, _ = gen(real_a, M_gt)

                # 真实样本
                d_out_real = dis(real_b)
                loss_d_real = -torch.mean(d_out_real)

                # 生成样本
                d_out_fake = dis(fake_b.detach())
                loss_d_fake = torch.mean(d_out_fake)

                # 梯度惩罚
                gp = compute_gradient_penalty(dis, real_b, fake_b.detach())
                total_dis_loss = loss_d_real + loss_d_fake + config.gp_weight * gp

                total_dis_loss.backward()
                opt_dis.step()

            # --- 生成器训练 ---
            # 遵循WGAN-GP的建议，判别器训练n次，生成器训练1次
            train_gen_now = (iteration % 5 == 0) if use_gan else True
            if train_gen_now:
                opt_gen.zero_grad()

                fake_b, pred_mask, domain_logits, caff_in, caff_out = gen(real_a, M_gt)

                # ========== 1) Task loss: L1 +/or VGG ==========
                # 开关：是否使用 L1、是否使用 VGG，由 loss_mode 决定
                use_l1 = True
                use_vgg = (hasattr(config, 'use_perceptual') and config.use_perceptual)

                if loss_mode == 'task_adv_L1only':
                    use_vgg = False
                if loss_mode == 'task_adv_VGGonly':
                    use_l1 = False
                    use_vgg = True

                loss_l1 = torch.tensor(0.0, device=device)
                if use_l1:
                    loss_l1 = task_loss_l1(fake_b, real_b)

                loss_vgg = torch.tensor(0.0, device=device)
                if use_vgg:
                    loss_vgg = perceptual_loss_vgg(fake_b, real_b) * getattr(config, 'perceptual_weight', 0.04)

                # 任务损失总和
                loss_task_total = loss_l1 + loss_vgg
                total_gen_loss = getattr(config, 'alpha', 1.0) * loss_task_total

                # ========== 2) MCAM mask loss（如果你希望一直训练 mask，可保留；不在你三项损失里也可以不写入总损失） ==========
                loss_mcam = torch.tensor(0.0, device=device)
                if pred_mask is not None:
                    if pred_mask.shape[2:] != M_gt.shape[2:]:
                        M_gt_resized = F.interpolate(M_gt, size=pred_mask.shape[2:], mode='nearest')
                    else:
                        M_gt_resized = M_gt
                    loss_mcam = getattr(config, 'gamma', 0.5) * cloud_mask_loss(pred_mask, M_gt_resized)
                    total_gen_loss = total_gen_loss + loss_mcam

                # ========== 3) Consistency loss: spectral + feature ==========
                use_consistency = ('cons' in loss_mode)  # task_cons / task_adv_cons 才启用

                loss_spec = torch.tensor(0.0, device=device)
                loss_cons = torch.tensor(0.0, device=device)
                if use_consistency and (caff_in is not None) and (caff_out is not None):
                    # non-cloud mask 对齐到图像尺寸
                    non_cloud_mask_img = 1 - F.interpolate(M_gt, size=fake_b.shape[2:], mode='nearest')
                    # 光谱/像素保持：无云区 fake_b 应尽量保持与输入 real_a 一致（你原代码就是这个定义）
                    loss_spec = spectral_loss(fake_b * non_cloud_mask_img, real_a * non_cloud_mask_img)

                    # 特征一致性：无云区 CAFF 前后特征不应改变
                    non_cloud_mask_feat = 1 - F.interpolate(M_gt, size=caff_in.shape[2:], mode='nearest')
                    loss_cons = consistency_loss(caff_out * non_cloud_mask_feat, caff_in * non_cloud_mask_feat)

                    # 用 delta 作为一致性总权重（也可拆成两个权重）
                    total_gen_loss = total_gen_loss + getattr(config, 'delta', 0.2) * (loss_spec + loss_cons)

                # ========== 4) Adversarial loss (WGAN) ==========
                loss_g_adv = torch.tensor(0.0, device=device)
                if use_gan:
                    d_out_gen = dis(fake_b)
                    loss_g_adv = -torch.mean(d_out_gen)  # WGAN generator loss
                    total_gen_loss = total_gen_loss + getattr(config, 'beta', 0.1) * loss_g_adv

                total_gen_loss.backward()
                opt_gen.step()

                # --- logging ---
                if iteration % config.print_every == 0:
                    print(
                        f"E[{epoch}]({iteration}/{len(training_data_loader)}): "
                        f"G:{total_gen_loss.item():.4f} | "
                        f"Task(L1:{loss_l1.item():.4f}, VGG:{loss_vgg.item():.4f}) | "
                        f"Cons(Spec:{loss_spec.item():.4f}, Feat:{loss_cons.item():.4f}) | "
                        f"Adv:{loss_g_adv.item():.4f} | "
                        f"Mask:{loss_mcam.item():.4f} | mode={loss_mode}"
                    )

            # --- 日志记录 ---
            '''if iteration % config.print_every == 0:
                log_msg = f"E[{epoch}]({iteration}/{len(training_data_loader)}): G_Loss:{total_gen_loss.item():.4f} [Task:{loss_task.item():.4f}, Mask:{loss_mcam.item():.4f}]"
                if use_gan: log_msg += f" [Adv:{loss_g_adv.item():.4f}, D_Loss:{total_dis_loss.item():.4f}]"
                if caff_in is not None: log_msg += f" [Spec:{loss_spec.item():.4f}, Cons:{loss_cons.item():.4f}]"
                print(log_msg)

                log = {'epoch': epoch, 'iteration': iteration, 'loss/gen_total': total_gen_loss.item(),
                       'loss/task': loss_task.item(), 'loss/mcam_mask': loss_mcam.item(),
                       'loss/spectral': loss_spec.item(), 'loss/consistency': loss_cons.item()}
                if use_gan: log.update({'loss/adv': loss_g_adv.item(), 'loss/dis_total': total_dis_loss.item()})
                logreport(log)'''

        # --- Epoch结束，保存模型 ---
        if epoch % config.snapshot_interval == 0:
            gen_path = os.path.join(config.out_dir, f'gen_epoch_{epoch}.pth')
            torch.save(gen.state_dict(), gen_path)
            if use_gan:
                dis_path = os.path.join(config.out_dir, f'dis_epoch_{epoch}.pth')
                torch.save(dis.state_dict(), dis_path)
                print(f'Checkpoints saved to {gen_path} and {dis_path}')
            else:
                print(f'Checkpoint saved to {gen_path}')

        logreport.save_lossgraph()