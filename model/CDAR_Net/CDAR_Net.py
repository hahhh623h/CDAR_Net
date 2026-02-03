# CDAR_Net.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.msda.rdb3 import RDB  # 假设此导入路径正确
from torch.autograd import Variable
import torch.autograd as autograd


# ==============================================================================
# 1. 多尺度上下文聚合模块 (MCAM) - 根据论文重构
# ==============================================================================
class MCAM(nn.Module):
    """
    多尺度上下文聚合模块 (MCAM)，根据论文描述重构。
    通过双分支结构，并行提取多尺度特征并预测云掩膜。
    """

    def __init__(self, in_channels=3, base_channels=64):
        super(MCAM, self).__init__()

        # 初始特征提取层 (共享)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 分支1: 多尺度特征提取 (空洞卷积金字塔)
        self.dilation1 = nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1)
        self.dilation2 = nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2)
        self.dilation3 = nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4)  # 论文中为r=1,2,4,8
        self.dilation4 = nn.Conv2d(base_channels, base_channels, 3, padding=8, dilation=8)  # 这里与论文保持一致

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 分支2: 云掩膜预测
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1, padding=0),
            nn.Sigmoid()  # 输出为0-1之间的概率掩膜
        )

    def forward(self, x):
        # 共享初始特征
        initial_feat = self.initial_conv(x)

        # 分支1: 提取多尺度特征
        d1 = self.dilation1(initial_feat)
        d2 = self.dilation2(initial_feat)
        d3 = self.dilation3(initial_feat)
        d4 = self.dilation4(initial_feat)

        # 拼接并融合
        concat_feat = torch.cat([d1, d2, d3, d4], dim=1)
        fused_feat = self.feature_fusion(concat_feat)

        # 分支2: 预测云掩膜
        pred_cloud_mask = self.mask_predictor(initial_feat)

        return fused_feat, pred_cloud_mask


# ==============================================================================
# 2. 云感知特征融合 (CAFF) - 假设结构不变，仅作展示
# ==============================================================================
class CloudAwareFusion(nn.Module):
    def __init__(self, channels):
        super(CloudAwareFusion, self).__init__()
        # 轻量增强特征提取模块 (EFEM) - 示例
        self.efem = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, feature_map, cloud_mask):
        # 对云掩膜进行插值以匹配特征图尺寸
        if feature_map.shape[2:] != cloud_mask.shape[2:]:
            mask = F.interpolate(cloud_mask, size=feature_map.shape[2:], mode='bilinear', align_corners=False)
        else:
            mask = cloud_mask

        # 仅对云区特征进行增强
        enhanced_features = self.efem(feature_map)

        # 根据掩膜融合：无云区保留原特征，有云区使用增强特征
        # M * F_enhanced + (1-M) * F_original
        fused_features = enhanced_features * mask + feature_map * (1 - mask)
        return fused_features


# ==============================================================================
# 3. 云感知域对齐 (CADA) / 对抗模块 - 假设结构不变，仅作展示
# ==============================================================================
class CloudAwareDomainAlign(nn.Module):
    def __init__(self, channels):
        super(CloudAwareDomainAlign, self).__init__()
        # 这里是简化的域对齐/对抗逻辑，输出logits
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // 2, 1, 1)
        )

    def forward(self, feature_map, cloud_mask):
        # 在此版本中，我们仅用它来生成用于对抗损失的 logits
        # 论文中的CCAR更为复杂，包含梯度反转和空间加权损失，
        # 此处为简化实现，使其与原训练代码兼容
        domain_logits = self.domain_classifier(feature_map)
        return feature_map, domain_logits.view(feature_map.size(0), -1)


# ==============================================================================
# 4. 主生成器网络 (CDAR_Net_Generator) - 整合了MCAM
# ==============================================================================
# 辅助模块，如UpSample等，从原代码中假设存在且正确
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


class CDAR_Net_Generator(nn.Module):
    def __init__(self, height=3, width=6, num_dense_layer=4, growth_rate=16, in_channels=3):
        super(CDAR_Net_Generator, self).__init__()
        self.height = height
        self.width = width
        self.stride = 2
        depth_rate = growth_rate

        self.mcam = MCAM(in_channels=in_channels, base_channels=depth_rate * 2)
        self.conv_in = nn.Conv2d(depth_rate * 2, depth_rate, kernel_size=3, padding=1)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)

        self.rdb_module = nn.ModuleDict()
        self.down_module = nn.ModuleDict()

        channel_num = depth_rate
        for i in range(self.height):
            for j in range(self.width // 2):
                if j > 0:
                    self.rdb_module['{}_{}'.format(i, j - 1)] = RDB(channel_num, num_dense_layer, growth_rate)
            if i < self.height - 1:
                self.down_module['{}_{}'.format(i, i + 1)] = nn.Conv2d(channel_num, channel_num * self.stride, 4, 2, 1)
                channel_num *= self.stride

        self.caff = CloudAwareFusion(channels=channel_num)
        self.cada = CloudAwareDomainAlign(channels=channel_num)
        self.rdb_out = RDB(channel_num, num_dense_layer, growth_rate)

        # ========================== 新增部分开始 ==========================
        # 新增一个模块字典，用于存放调整通道数的1x1卷积层
        self.channel_adjust_module = nn.ModuleDict()
        # 从最深层向上回溯，为每一级上采样路径创建一个通道调整层
        # channel_num 此时是最大通道数
        _current_channels = channel_num
        for i in range(self.height - 2, -1, -1):
            self.channel_adjust_module[str(i)] = nn.Conv2d(
                in_channels=_current_channels,
                out_channels=_current_channels // self.stride,
                kernel_size=1, stride=1, padding=0
            )
            _current_channels //= self.stride
        # ========================== 新增部分结束 ==========================

        self.up_module = nn.ModuleDict()
        self.rdb_up_module = nn.ModuleDict()
        # channel_num 在循环外已经最大化, 此处重置为解码器输入时的通道数
        channel_num_decoder = channel_num
        for i in range(self.height - 1, 0, -1):
            self.up_module['{}_{}'.format(i, i - 1)] = UpSample(channel_num_decoder, channel_num_decoder // self.stride)
            channel_num_decoder //= self.stride
            self.rdb_up_module['{}_{}'.format(i, i - 1)] = RDB(channel_num_decoder, num_dense_layer, growth_rate)

        self.conv_out = nn.Conv2d(channel_num_decoder, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        mcam_feat, pred_cloud_mask = self.mcam(x)
        inp = self.conv_in(mcam_feat)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        x_index[0][0] = self.rdb_in(inp)

        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j - 1)](x_index[0][j - 1])

        for i in range(1, self.height):
            x_index[i][0] = self.down_module['{}_{}'.format(i - 1, i)](x_index[i - 1][0])
            for j in range(1, self.width // 2):
                x_index[i][j] = self.rdb_module['{}_{}'.format(i, j - 1)](x_index[i][j - 1])

        # ========================== 修正部分开始 ==========================
        x_index[self.height - 1][self.width // 2] = x_index[self.height - 1][self.width // 2 - 1]
        for j in range(self.width // 2 + 1, self.width):
            x_index[self.height - 1][j] = x_index[self.height - 1][j - 1]

        # 向上回传特征时，同时进行空间上采样和通道调整
        for i in range(self.height - 2, -1, -1):
            # 获取当前层的目标空间尺寸
            target_size = x_index[i][self.width // 2 - 1].shape[2:]

            # 1. 对来自深层的特征图进行空间上采样
            upsampled_feat_spatial = F.interpolate(x_index[i + 1][self.width // 2], size=target_size, mode='bilinear',
                                                   align_corners=False)
            # 2. 使用1x1卷积调整通道数
            adjusted_feat = self.channel_adjust_module[str(i)](upsampled_feat_spatial)

            # 现在空间和通道维度都匹配了，可以安全相加
            x_index[i][self.width // 2] = x_index[i][self.width // 2 - 1] + adjusted_feat

            for j in range(self.width // 2 + 1, self.width):
                inner_target_size = x_index[i][j - 1].shape[2:]
                inner_upsampled_spatial = F.interpolate(x_index[i + 1][j], size=inner_target_size, mode='bilinear',
                                                        align_corners=False)
                # 使用同一个通道调整层
                inner_adjusted_feat = self.channel_adjust_module[str(i)](inner_upsampled_spatial)
                x_index[i][j] = x_index[i][j - 1] + inner_adjusted_feat
        # ========================== 修正部分结束 ==========================

        last_feature = x_index[-1][-1]

        caff_feature = self.caff(last_feature, pred_cloud_mask)
        aligned_feat, domain_logits = self.cada(caff_feature, pred_cloud_mask)
        out = self.rdb_out(aligned_feat)

        for i in range(self.height - 1, 0, -1):
            out = self.up_module['{}_{}'.format(i, i - 1)](out, x_index[i - 1][-1].shape[2:])
            out = self.rdb_up_module['{}_{}'.format(i, i - 1)](out)

        out = self.conv_out(out)

        return out, pred_cloud_mask, domain_logits


# ==============================================================================
# 5. 联合损失函数 (CDAR_Loss) - 根据论文重构
# ==============================================================================
class Lap(nn.Module):
    def __init__(self):
        super(Lap, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        return self.conv(x)


class CDAR_Loss(nn.Module):
    def __init__(self, task_weight=1.0, domain_weight=0.1, mcam_mask_weight=0.5,
                 spectral_weight=0.2, consistency_weight=0.2):
        super().__init__()
        self.task_weight = task_weight
        self.domain_weight = domain_weight
        self.mcam_mask_weight = mcam_mask_weight
        self.spectral_weight = spectral_weight
        self.consistency_weight = consistency_weight

        # 任务损失 (像素级+感知) - 简化为L1，感知损失在训练循环中单独计算
        self.task_loss_l1 = nn.L1Loss()

        # 对抗/域损失
        self.domain_loss = nn.BCEWithLogitsLoss()

        # MCAM模块的云掩膜预测损失
        self.cloud_mask_loss = nn.BCELoss()

        # 云感知物理约束损失 (根据论文)
        self.spectral_loss = nn.L1Loss()  # 光谱保真
        self.consistency_loss = nn.MSELoss()  # 特征一致性

    def forward(self, pred_img, target_img, pred_mask, gt_mask,
                domain_logits, is_real, original_cloudy_img, caff_in_feat, caff_out_feat):

        # 确保尺寸一致
        if pred_mask.size() != gt_mask.size():
            gt_mask_resized = F.interpolate(gt_mask, size=pred_mask.shape[2:], mode='nearest')
        else:
            gt_mask_resized = gt_mask

        # 1. L_task: 主任务损失 (重建损失)
        loss_task = self.task_loss_l1(pred_img, target_img)

        # 2. L_adv: 对抗损失 (生成器)
        target = torch.ones_like(domain_logits) if is_real else torch.zeros_like(domain_logits)
        loss_domain = self.domain_loss(domain_logits, target)

        # 3. L_mcam: MCAM的云掩膜监督损失
        loss_mcam = self.cloud_mask_loss(pred_mask, gt_mask_resized)

        # 4. L_spectral: 光谱保真损失 (在无云区)
        # gt_mask=0的区域是无云区
        non_cloud_mask = 1 - gt_mask_resized
        loss_spec = self.spectral_loss(pred_img * non_cloud_mask, original_cloudy_img * non_cloud_mask)

        # 5. L_consistency: 特征一致性损失 (在无云区)
        loss_cons = self.consistency_loss(caff_out_feat * non_cloud_mask, caff_in_feat * non_cloud_mask)

        # --- 计算总损失 ---
        total_loss = (self.task_weight * loss_task +
                      self.domain_weight * loss_domain +
                      self.mcam_mask_weight * loss_mcam +
                      self.spectral_weight * loss_spec +
                      self.consistency_weight * loss_cons)

        # 返回总损失和各分量，用于日志记录
        loss_components = {
            'total': total_loss,
            'task': loss_task,
            'domain': loss_domain,
            'mcam_mask': loss_mcam,
            'spectral': loss_spec,
            'consistency': loss_cons
        }

        return total_loss, loss_components


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(real_samples.device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty'''

# CDAR_Net.py (Ablation Study Version)
# CDAR_Net.py (Final Corrected Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

try:
    from model.msda.rdb3 import RDB
except ImportError:
    print("Warning: Could not import RDB. Please check the path.")
    RDB = nn.Identity


# ==============================================================================
# 1. 核心模块 (MCAM, CAFF, CCAR) - OK
# ==============================================================================
class MCAM(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(MCAM, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 多尺度特征分支（示意）
        self.dilation1 = nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1)
        self.dilation2 = nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2)
        self.dilation4 = nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4)
        self.dilation8 = nn.Conv2d(base_channels, base_channels, 3, padding=8, dilation=8)
        self.fuse_conv = nn.Conv2d(base_channels * 4, base_channels, kernel_size=1)

        # ========= 云概率图分支：关键在这里 =========
        # 1) 3x3 conv
        self.mask_feat_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        # 2) Sigmoid 前最后一层 conv（Grad-CAM 推荐 hook 这层）
        self.mask_pre_sigmoid_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        # 3) Sigmoid
        self.mask_sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat0 = self.initial_conv(x)

        # 多尺度上下文特征
        f1 = F.relu(self.dilation1(feat0), inplace=True)
        f2 = F.relu(self.dilation2(feat0), inplace=True)
        f4 = F.relu(self.dilation4(feat0), inplace=True)
        f8 = F.relu(self.dilation8(feat0), inplace=True)
        feat = torch.cat([f1, f2, f4, f8], dim=1)
        feat = F.relu(self.fuse_conv(feat), inplace=True)

        # 云概率图
        m_feat = F.relu(self.mask_feat_conv(feat0), inplace=True)
        m_logit = self.mask_pre_sigmoid_conv(m_feat)   # <-- hook 的目标层输出（logit）
        m_prob = self.mask_sigmoid(m_logit)            # [B,1,H,W] in [0,1]

        return feat, m_prob



class CloudAwareFusion(nn.Module):
    def __init__(self, channels):
        super(CloudAwareFusion, self).__init__()
        self.efem = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, feature_map, cloud_mask):
        if feature_map.shape[2:] != cloud_mask.shape[2:]:
            mask = F.interpolate(cloud_mask, size=feature_map.shape[2:], mode='bilinear', align_corners=False)
        else:
            mask = cloud_mask
        enhanced_features = self.efem(feature_map)
        fused_features = enhanced_features * mask + feature_map * (1 - mask)
        return fused_features, feature_map, fused_features

'''class CloudAwareFusion(nn.Module):
    # 修改 __init__ 方法，增加 threshold 参数
    def __init__(self, channels, threshold=None):  # 添加 threshold 参数
        super(CloudAwareFusion, self).__init__()
        self.threshold = threshold  # 保存阈值
        # 轻量增强特征提取模块 (EFEM) - 示例
        self.efem = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    # 修改 forward 方法
    def forward(self, feature_map, cloud_mask):
        # 对云掩膜进行插值以匹配特征图尺寸
        if feature_map.shape[2:] != cloud_mask.shape[2:]:
            mask = F.interpolate(cloud_mask, size=feature_map.shape[2:], mode='bilinear', align_corners=False)
        else:
            mask = cloud_mask

        # --- 消融实验的关键修改：使用 self.threshold 进行二值化 ---
        # 使用配置中的阈值进行二值化
        current_threshold = self.threshold if self.threshold is not None else 0.5
        binary_mask = (mask > self.threshold).float()

        # 分解为云区特征和无云区特征
        cloud_region_feat = feature_map * binary_mask
        non_cloud_region_feat = feature_map * (1 - binary_mask)

        # 对云区特征进行增强
        enhanced_cloud_feat = self.efem(cloud_region_feat)

        # 融合：增强的云区特征 + 原始无云区特征
        fused_features = enhanced_cloud_feat + non_cloud_region_feat

        # 为后续损失计算返回CAFF输入和输出特征（用于一致性损失）
        return fused_features, feature_map, fused_features  # 返回三个值：fused, original, enhanced'''

class CloudAwareDomainAlign(nn.Module):
    def __init__(self, channels):
        super(CloudAwareDomainAlign, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels // 2, out_channels=1, kernel_size=1)
        )

    def forward(self, feature_map):
        domain_logits = self.domain_classifier(feature_map)
        return domain_logits.view(feature_map.size(0), -1)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


# ==============================================================================
# 2. 基础生成器网络 (Baseline) - 再次修正解码器通道逻辑
# ==============================================================================
class BaseGenerator(nn.Module):
    '''def __init__(self, height, width, num_dense_layer, growth_rate, in_channels):
        super(BaseGenerator, self).__init__()
        stride = 2
        depth_rate = growth_rate

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=depth_rate, kernel_size=3, stride=1, padding=1)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)

        # --- 编码器 ---
        self.down_module = nn.ModuleDict()
        self.rdb_down_module = nn.ModuleDict()
        encoder_channels = [depth_rate]
        current_channels = depth_rate
        for i in range(height):
            self.rdb_down_module[f'rdb_{i}'] = RDB(current_channels, num_dense_layer, growth_rate)
            if i < height - 1:
                self.down_module[f'down_{i}'] = nn.Conv2d(in_channels=current_channels,
                                                          out_channels=current_channels * stride, kernel_size=4,
                                                          stride=stride, padding=1)
                current_channels *= stride
                encoder_channels.append(current_channels)

        self.rdb_middle = RDB(current_channels, num_dense_layer, growth_rate)

        # --- 解码器 (彻底修正的逻辑) ---
        self.up_module = nn.ModuleDict()
        self.rdb_up_module = nn.ModuleDict()

        decoder_current_channels = current_channels
        for i in range(height - 1, 0, -1):
            # i = 2, 1 (for height=3)
            # upsample 输入是 decoder_current_channels, 输出是 decoder_current_channels / 2
            upsample_out_channels = decoder_current_channels // stride
            self.up_module[f'up_{i - 1}'] = UpSample(in_channels=decoder_current_channels,
                                                     out_channels=upsample_out_channels)

            # skip connection 的通道数来自 encoder_channels 列表
            # 当 i=2, skip_index=1, skip_channels = encoder_channels[1] = 32
            # 当 i=1, skip_index=0, skip_channels = encoder_channels[0] = 16
            skip_index = i - 1
            skip_channels = encoder_channels[skip_index]

            # RDB的输入是 upsample_out 和 skip_connection 的拼接
            rdb_input_channels = upsample_out_channels + skip_channels
            self.rdb_up_module[f'rdb_{i}'] = RDB(rdb_input_channels, num_dense_layer, growth_rate)

            # RDB的输出通道数等于输入通道数, 这将作为下一次 upsample 的输入
            decoder_current_channels = rdb_input_channels

        self.conv_out = nn.Conv2d(in_channels=decoder_current_channels, out_channels=in_channels, kernel_size=3,
                                  stride=1, padding=1)'''

    def __init__(self, height, width, num_dense_layer, growth_rate, in_channels):
        super(BaseGenerator, self).__init__()
        stride = 2
        depth_rate = growth_rate

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=depth_rate, kernel_size=3, stride=1, padding=1)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)

        # --- 编码器 ---
        self.down_module = nn.ModuleDict()
        self.rdb_down_module = nn.ModuleDict()
        encoder_channels = []  # Stores channels before downsampling at each level
        current_enc_channels = depth_rate  # Start with depth_rate

        for i in range(height):
            encoder_channels.append(current_enc_channels)  # Save channels for skip connection
            self.rdb_down_module[f'rdb_{i}'] = RDB(current_enc_channels, num_dense_layer, growth_rate)
            if i < height - 1:
                print(f"[debug] down_{i}: current_enc_channels={current_enc_channels}")
                self.down_module[f'down_{i}'] = nn.Conv2d(in_channels=current_enc_channels,
                                                          out_channels=current_enc_channels * stride, kernel_size=4,
                                                          stride=stride, padding=1)
                current_enc_channels *= stride  # Update channels for next level

        # `current_enc_channels` now holds the bottleneck channels
        bottleneck_channels = current_enc_channels
        self.rdb_middle = RDB(bottleneck_channels, num_dense_layer, growth_rate)

        # --- 解码器 ---
        self.up_module = nn.ModuleDict()
        self.rdb_up_module = nn.ModuleDict()

        # The initial `dec_feat` entering the decoder loop in CDAR_Net_Generator.forward
        # is `self.backbone.rdb_middle(caff_feat)` (bottleneck_channels)
        # concatenated with `skip_connections[height - 1]` (bottleneck_channels).
        # So the input to `rdb_up_module[f'rdb_{height-1}']` is `bottleneck_channels * 2`.
        self.rdb_up_module[f'rdb_{height - 1}'] = RDB(bottleneck_channels * 2, num_dense_layer, growth_rate)

        # `decoder_current_channels` will track the channel count after each RDB in the decoder.
        # Its initial value is the output of `rdb_up_module[f'rdb_{height-1}']`, which is `bottleneck_channels * 2`.
        decoder_current_channels = bottleneck_channels * 2

        # Loop for remaining decoder stages (from height-2 down to 0)
        # These correspond to `up_module[f'up_{i}']` and `rdb_up_module[f'rdb_{i}']` in CDAR_Net_Generator.forward
        for i in range(height - 2, -1, -1):  # e.g., for height=3, i will be 1, then 0
            # `up_module[f'up_{i}']` takes `decoder_current_channels` as input.
            # Its output should match the corresponding skip connection `encoder_channels[i]`.
            upsample_output_channels = encoder_channels[i]
            self.up_module[f'up_{i}'] = UpSample(in_channels=decoder_current_channels,
                                                 out_channels=upsample_output_channels)

            # The RDB `rdb_up_module[f'rdb_{i}']` takes `(upsample_output_channels + encoder_channels[i])` as input.
            rdb_input_channels = upsample_output_channels + encoder_channels[i]
            self.rdb_up_module[f'rdb_{i}'] = RDB(rdb_input_channels, num_dense_layer, growth_rate)

            # Update `decoder_current_channels` to be the output of this RDB,
            # which serves as the input to the next `up_module`.
            decoder_current_channels = rdb_input_channels

            # --- Output Layer ---
        # The final `conv_out` takes the channel count from the last `decoder_current_channels`
        self.conv_out = nn.Conv2d(in_channels=decoder_current_channels, out_channels=in_channels, kernel_size=3,
                                  stride=1, padding=1)
    def forward(self, x):
        x = self.conv_in(x)
        x = self.rdb_in(x)

        skip_connections = []
        for i in range(len(self.down_module) + 1):
            x = self.rdb_down_module[f'rdb_{i}'](x)
            skip_connections.append(x)
            if f'down_{i}' in self.down_module:
                x = self.down_module[f'down_{i}'](x)

        x = self.rdb_middle(x)

        for i in range(len(self.up_module), 0, -1):
            skip_feat = skip_connections[i - 1]
            x = self.up_module[f'up_{i - 1}'](x, target_size=skip_feat.shape[2:])
            x = torch.cat([x, skip_feat], dim=1)
            x = self.rdb_up_module[f'rdb_{i}'](x)

        out = self.conv_out(x)
        return out


# ==============================================================================
# 3. 消融模型定义 (其余部分保持不变, 仅为完整性提供)
# ==============================================================================

# ① 基线
class Generator_Baseline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = BaseGenerator(**kwargs)

    def forward(self, x, gt_mask=None):
        img = self.backbone(x)
        return img, None, None, None, None


# ② 基线 + MCAM
class Generator_MCAM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mcam = MCAM(in_channels=kwargs['in_channels'], base_channels=kwargs['growth_rate'] * 2)
        self.adapter = nn.Conv2d(in_channels=kwargs['growth_rate'] * 2, out_channels=kwargs['in_channels'],
                                 kernel_size=1)
        self.backbone = BaseGenerator(**kwargs)

    def forward(self, x, gt_mask=None):
        feat, pred_mask = self.mcam(x)
        adapted_feat = self.adapter(feat)
        img = self.backbone(adapted_feat)
        return img, pred_mask, None, None, None


# ③ 基线 + CAFF
class Generator_CAFF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = BaseGenerator(**kwargs)
        channel_num = kwargs['growth_rate'] * (2 ** (kwargs['height'] - 1))
        self.caff = CloudAwareFusion(channels=channel_num)

    def forward(self, x, gt_mask=None):
        skip_connections = []
        x = self.backbone.conv_in(x)
        x = self.backbone.rdb_in(x)
        for i in range(len(self.backbone.down_module) + 1):
            x = self.backbone.rdb_down_module[f'rdb_{i}'](x)
            skip_connections.append(x)
            if f'down_{i}' in self.backbone.down_module:
                x = self.backbone.down_module[f'down_{i}'](x)

        x, caff_in, caff_out = self.caff(x, gt_mask)

        x = self.backbone.rdb_middle(x)
        for i in range(len(self.backbone.up_module), 0, -1):
            skip_feat = skip_connections[i - 1]
            x = self.backbone.up_module[f'up_{i - 1}'](x, target_size=skip_feat.shape[2:])
            x = torch.cat([x, skip_feat], dim=1)
            x = self.backbone.rdb_up_module[f'rdb_{i}'](x)

        img = self.backbone.conv_out(x)
        return img, None, None, caff_in, caff_out


# ④ 基线 + CCAR
class Generator_CCAR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = BaseGenerator(**kwargs)
        channel_num = kwargs['growth_rate'] * (2 ** (kwargs['height'] - 1))
        self.ccar = CloudAwareDomainAlign(channels=channel_num)

    def forward(self, x, gt_mask=None):
        skip_connections = []
        x = self.backbone.conv_in(x)
        x = self.backbone.rdb_in(x)
        for i in range(len(self.backbone.down_module) + 1):
            x = self.backbone.rdb_down_module[f'rdb_{i}'](x)
            skip_connections.append(x)
            if f'down_{i}' in self.backbone.down_module:
                x = self.backbone.down_module[f'down_{i}'](x)

        domain_logits = self.ccar(x)

        x = self.backbone.rdb_middle(x)
        for i in range(len(self.backbone.up_module), 0, -1):
            skip_feat = skip_connections[i - 1]
            x = self.backbone.up_module[f'up_{i - 1}'](x, target_size=skip_feat.shape[2:])
            x = torch.cat([x, skip_feat], dim=1)
            x = self.backbone.rdb_up_module[f'rdb_{i}'](x)

        img = self.backbone.conv_out(x)
        return img, None, domain_logits, None, None


# ⑤ 基线 + MCAM + CAFF
class Generator_MCAM_CAFF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mcam = MCAM(in_channels=kwargs['in_channels'], base_channels=kwargs['growth_rate'] * 2)
        self.adapter = nn.Conv2d(in_channels=kwargs['growth_rate'] * 2, out_channels=kwargs['in_channels'],
                                 kernel_size=1)
        self.caff_backbone = Generator_CAFF(**kwargs)

    def forward(self, x, gt_mask=None):
        feat, pred_mask = self.mcam(x)
        adapted_feat = self.adapter(feat)
        img, _, _, caff_in, caff_out = self.caff_backbone(adapted_feat, pred_mask)
        return img, pred_mask, None, caff_in, caff_out


# ⑥ 基线 + MCAM + CCAR
class Generator_MCAM_CCAR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mcam = MCAM(in_channels=kwargs['in_channels'], base_channels=kwargs['growth_rate'] * 2)
        self.adapter = nn.Conv2d(in_channels=kwargs['growth_rate'] * 2, out_channels=kwargs['in_channels'],
                                 kernel_size=1)
        self.ccar_backbone = Generator_CCAR(**kwargs)

    def forward(self, x, gt_mask=None):
        feat, pred_mask = self.mcam(x)
        adapted_feat = self.adapter(feat)
        img, _, domain_logits, _, _ = self.ccar_backbone(adapted_feat)
        return img, pred_mask, domain_logits, None, None


# ⑦ 基线 + CAFF + CCAR
class Generator_CAFF_CCAR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = BaseGenerator(**kwargs)
        channel_num = kwargs['growth_rate'] * (2 ** (kwargs['height'] - 1))
        self.caff = CloudAwareFusion(channels=channel_num)
        self.ccar = CloudAwareDomainAlign(channels=channel_num)

    def forward(self, x, gt_mask=None):
        skip_connections = []
        x = self.backbone.conv_in(x)
        x = self.backbone.rdb_in(x)
        for i in range(len(self.backbone.down_module) + 1):
            x = self.backbone.rdb_down_module[f'rdb_{i}'](x)
            skip_connections.append(x)
            if f'down_{i}' in self.backbone.down_module:
                x = self.backbone.down_module[f'down_{i}'](x)

        caff_feat, caff_in, caff_out = self.caff(x, gt_mask)
        domain_logits = self.ccar(caff_feat)

        dec_feat = self.backbone.rdb_middle(caff_feat)
        for i in range(len(self.backbone.up_module), 0, -1):
            skip_feat = skip_connections[i - 1]
            dec_feat = self.backbone.up_module[f'up_{i - 1}'](dec_feat, target_size=skip_feat.shape[2:])
            dec_feat = torch.cat([dec_feat, skip_feat], dim=1)
            dec_feat = self.backbone.rdb_up_module[f'rdb_{i}'](dec_feat)

        img = self.backbone.conv_out(dec_feat)
        return img, None, domain_logits, caff_in, caff_out


# ⑧ 完整模型 (CDAR_Net)
class CDAR_Net_Generator(nn.Module):
    '''def __init__(self, **kwargs):
        super().__init__()
        self.mcam = MCAM(in_channels=kwargs['in_channels'], base_channels=kwargs['growth_rate'] * 2)
        self.adapter = nn.Conv2d(in_channels=kwargs['growth_rate'] * 2, out_channels=kwargs['in_channels'],
                                 kernel_size=1)
        self.backbone = BaseGenerator(**kwargs)
        channel_num = kwargs['growth_rate'] * (2 ** (kwargs['height'] - 1))
        self.caff = CloudAwareFusion(channels=channel_num)
        #self.caff = CloudAwareFusion(channels=channel_num, threshold=0.5)
        self.ccar = CloudAwareDomainAlign(channels=channel_num)

    def forward(self, x, gt_mask=None):

        mcam_feat, pred_cloud_mask = self.mcam(x)
        inp = self.conv_in(mcam_feat)
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        x_index[0][0] = self.rdb_in(inp)

        feat, pred_mask = self.mcam(x)
        adapted_feat = self.adapter(feat)
        skip_connections = []
        enc_feat = self.backbone.conv_in(adapted_feat)
        enc_feat = self.backbone.rdb_in(enc_feat)
        for i in range(len(self.backbone.down_module) + 1):
            enc_feat = self.backbone.rdb_down_module[f'rdb_{i}'](enc_feat)
            skip_connections.append(enc_feat)
            if f'down_{i}' in self.backbone.down_module:
                enc_feat = self.backbone.down_module[f'down_{i}'](enc_feat)

        caff_feat, caff_in, caff_out = self.caff(enc_feat, pred_mask)
        domain_logits = self.ccar(caff_feat)

        dec_feat = self.backbone.rdb_middle(caff_feat)
        for i in range(len(self.backbone.up_module), 0, -1):
            skip_feat = skip_connections[i - 1]
            dec_feat = self.backbone.up_module[f'up_{i - 1}'](dec_feat, target_size=skip_feat.shape[2:])
            dec_feat = torch.cat([dec_feat, skip_feat], dim=1)
            dec_feat = self.backbone.rdb_up_module[f'rdb_{i}'](dec_feat)

        img = self.backbone.conv_out(dec_feat)
        return img, pred_mask, domain_logits, caff_in, caff_out'''

    def __init__(self, **kwargs):
        super().__init__()
        self.height = kwargs['height']  # 添加 height 和 width 属性
        self.width = kwargs['width']

        self.mcam = MCAM(in_channels=kwargs['in_channels'], base_channels=kwargs['growth_rate'] * 2)
        self.adapter = nn.Conv2d(in_channels=kwargs['growth_rate'] * 2, out_channels=kwargs['in_channels'],
                                 kernel_size=1)
        self.backbone = BaseGenerator(**kwargs)

        channel_num = kwargs['growth_rate'] * (2 ** (kwargs['height'] - 1))
        self.caff = CloudAwareFusion(channels=channel_num)
        #self.caff = CloudAwareFusion(channels=channel_num, threshold=0.5)
        self.ccar = CloudAwareDomainAlign(channels=channel_num)

    # 保持 gt_mask=None，因为该模块内部不直接使用它来修改 MCAM/CAFF/CCAR 的行为
    def forward(self, x, gt_mask=None):

        # 1. MCAM 模块
        mcam_feat, pred_cloud_mask = self.mcam(x)
        # 2. 特征适配 (如果 adapter 在这里使用，通常是在 backbone.conv_in 之前)
        adapted_feat = self.adapter(mcam_feat)  # 使用 mcam_feat，而不是重复调用 mcam(x)
        # 3. Backbone (Encoder-Decoder) 路径
        skip_connections = []
        enc_feat = self.backbone.conv_in(adapted_feat)  # 假设 backbone 有 conv_in
        enc_feat = self.backbone.rdb_in(enc_feat)  # 假设 backbone 有 rdb_in

        for i in range(self.height):  # 遍历编码器层
            # 存储 RDB 后的特征作为跳跃连接
            enc_feat = self.backbone.rdb_down_module[f'rdb_{i}'](enc_feat)
            skip_connections.append(enc_feat)
            if i < self.height - 1:
                enc_feat = self.backbone.down_module[f'down_{i}'](enc_feat)
        # 4. CAFF 模块
        caff_feat, caff_in, caff_out = self.caff(enc_feat, pred_cloud_mask)
        # 5. CCAR 模块
        # CCAR 作用于 CAFF 融合后的特征
        domain_logits = self.ccar(caff_feat)
        # 6. Decoder 路径
        dec_feat = self.backbone.rdb_middle(caff_feat)  # 假设 backbone 有 rdb_middle
        dec_feat = torch.cat([dec_feat, skip_connections[self.height - 1]], dim=1)  # (bottleneck_channels * 2)
        dec_feat = self.backbone.rdb_up_module[f'rdb_{self.height - 1}'](dec_feat)
        for i in range(self.height - 2, -1, -1):
            current_skip_feat = skip_connections[i]

            # Up-sample `dec_feat` from previous stage
            dec_feat = self.backbone.up_module[f'up_{i}'](dec_feat, target_size=current_skip_feat.shape[2:])

            # Concatenate with corresponding skip connection
            dec_feat = torch.cat([dec_feat, current_skip_feat], dim=1)

            # Apply RDB
            dec_feat = self.backbone.rdb_up_module[f'rdb_{i}'](dec_feat)
        # 7. 最终输出图像
        img = self.backbone.conv_out(dec_feat)  # 假设 backbone 有 conv_out
        # 返回所有需要的输出
        return img, pred_cloud_mask, domain_logits, caff_in, caff_out

# WGAN-GP 工具函数
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(real_samples.device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
