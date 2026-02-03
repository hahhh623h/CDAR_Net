# evaluation.py

import os
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision.utils import save_image
import numpy as np
from model.CDAR_Net.CDAR_Net import (
    CDAR_Net_Generator,
    CloudAwareFusion
)

def to_01(x: torch.Tensor) -> torch.Tensor:
    """
    把张量统一转到[0,1]，兼容[-1,1]或[0,1]
    x: (B,C,H,W)
    """
    if x.min().item() < 0:      # 认为是[-1,1]
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)

def to_rgb_preview(x: torch.Tensor, mode="s2_13_to_rgb") -> torch.Tensor:
    """
    把任意通道输入转成可保存的RGB预览 (B,3,H,W)
    - 若 C==3：直接返回
    - 若 C>=13：默认按 Sentinel-2 常见顺序 [B1..B12,B8A]，取 B4,B3,B2 => idx 3,2,1 (0-based)
      你如果 13 通道顺序不同，只要改这里的索引即可。
    """
    b, c, h, w = x.shape
    if c == 3:
        return x
    if c >= 13 and mode == "s2_13_to_rgb":
        rgb = x[:, [3, 2, 1], :, :]  # B4,B3,B2（假设 band 顺序是 B1,B2,B3,B4,...）
        return rgb
    # 兜底：取前3个通道
    return x[:, :3, :, :]

def test_cdar_net(config, data_loader, gen, device):
    """
    专门为 CDAR_Net 设计的测试函数，计算 PSNR, SSIM, LPIPS。
    """
    # 1. 设置模型为评估模式
    gen.eval()

    # 2. 初始化 LPIPS 模型
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # 3. 创建用于保存结果的目录和列表
    output_dir = os.path.join(config.out_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print("===> Starting testing...")
    with torch.no_grad():
        # 使用 tqdm 创建进度条
        for i, batch in enumerate(tqdm(data_loader, desc="Testing")):
            # 加载数据
            real_a, real_b, M_gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if M_gt.dim() == 3:
                M_gt = M_gt.unsqueeze(1)

            # 模型推理: 传入 M_gt 并接收所有输出
            fake_b, pred_mask, domain_logits, caff_in, caff_out = gen(real_a, M_gt)

            fake_b_01 = to_01(fake_b)
            real_b_01 = to_01(real_b)

            # 输入可能是13通道：先转RGB预览再保存
            real_a_rgb = to_rgb_preview(to_01(real_a))  # 先归一化到[0,1]再选RGB

            save_image(fake_b_01, os.path.join(output_dir, f'output_{i + 1}.png'))
            save_image(real_a_rgb, os.path.join(output_dir, f'input_{i + 1}.png'))
            save_image(real_b_01, os.path.join(output_dir, f'gt_{i + 1}.png'))

            # 将图像保存到文件夹
            #save_image(fake_b, os.path.join(output_dir, f'output_{i + 1}.png'))
            #save_image(real_a, os.path.join(output_dir, f'input_{i + 1}.png'))
            #save_image(real_b, os.path.join(output_dir, f'gt_{i + 1}.png'))

            # --- 指标计算 ---
            # 将 PyTorch 张量转换为 NumPy 数组以计算 PSNR 和 SSIM
            # 转换过程: GPU -> CPU -> NumPy -> [0, 255]范围 -> 维度转置 (C,H,W) -> (H,W,C)
            # 假设输入图像范围是 [-1, 1]，我们先将其转换到 [0, 1]
            #fake_b_np = (fake_b.squeeze(0).cpu().numpy() + 1) / 2.0
            #real_b_np = (real_b.squeeze(0).cpu().numpy() + 1) / 2.0
            fake_b_np = fake_b_01.squeeze(0).detach().cpu().numpy()
            real_b_np = real_b_01.squeeze(0).detach().cpu().numpy()
            # 转置维度
            fake_b_np = np.transpose(fake_b_np, (1, 2, 0))
            real_b_np = np.transpose(real_b_np, (1, 2, 0))

            # 计算 PSNR
            # data_range 是图像像素值的范围，对于[0,1]的浮点数，data_range=1
            current_psnr = psnr(real_b_np, fake_b_np, data_range=1.0)
            psnr_scores.append(current_psnr)

            # 计算 SSIM
            # channel_axis=-1 表示最后一个维度是通道
            current_ssim = ssim(real_b_np, fake_b_np, data_range=1.0, channel_axis=-1, win_size=7)
            ssim_scores.append(current_ssim)

            # 计算 LPIPS
            # LPIPS库直接使用PyTorch张量，范围应为[-1, 1]
            current_lpips = lpips_fn(real_b, fake_b).item()
            lpips_scores.append(current_lpips)

    # 计算并打印平均值
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)

    print("\n" + "=" * 20)
    print("      Test Results      ")
    print("=" * 20)
    print(f"Average PSNR:  {avg_psnr:.4f}")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("=" * 20)
    print(f"Results and images saved in: {output_dir}")

    # 将结果保存到文件
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Test Results Summary\n")
        f.write("=" * 20 + "\n")
        f.write(f"Average PSNR:  {avg_psnr:.4f}\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

    # 4. 恢复模型到训练模式
    gen.train()

    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_lpips': avg_lpips
    }