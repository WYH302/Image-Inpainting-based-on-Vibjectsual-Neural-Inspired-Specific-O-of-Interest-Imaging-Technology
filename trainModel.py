import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
from torch import amp

# 额外工具
from skimage.metrics import structural_similarity as ssim_np
import time
import warnings

import torch
import os


# -------------------- 超参数设置 --------------------
batch_size = 20
epochs = 100
lr = 0.0002
step_size = 90

# 固定损失权重
L1_WEIGHT = 1.0
FFT_WEIGHT = 0.05
MSE_WEIGHT = 0.1
VGG_WEIGHT = 0.01  # 感知损失权重

# 文件路径配置（请按需要修改）
train_folder = "autodl-tmp/three_categories/three_categories/elephant/train/images"
target_folder = "autodl-tmp/three_categories/three_categories/elephant/train/target"
validation_folder = "autodl-tmp/three_categories/three_categories/elephant/val/images"
validation_target_folder = "autodl-tmp/three_categories/three_categories/elephant/val/target"


# -------------------- 数据集 (已修改：支持 patch 随机裁剪/中心裁剪) --------------------
class CircleDataset(Dataset):
    """
    自定义数据集：支持将整张 LR/HR 图像裁剪成 patches 作为训练样本。
    - 如果指定 patch_size 与 scale_factor，则使用 patch 策略（训练时随机裁剪，验证时中心裁剪）。
    - 若图像尺寸小于所需 patch，会进行插值放大以保证尺寸足够。
    """

    def __init__(self, train_folder, target_folder,
                 input_size=(128, 128), output_size=(512, 512),
                 train_transform=None, target_transform=None,
                 patch_size=None, scale_factor=4, is_train=True):
        """
        patch_size: 输入 LR patch 的边长（像素），例如 128。若为 None 则不做 patch（保留原行为，对输入/目标进行传入的 transform）。
        scale_factor: HR 相对于 LR 的放大倍数（例如 4）
        is_train: 是否训练集（训练使用随机裁剪，验证/测试使用中心裁剪）
        """
        seed = random.randint(1, 99)
        self.train_folder = train_folder
        self.target_folder = target_folder
        self.input_size = input_size
        self.output_size = output_size

        # 如果启用 patch 模式，则内部 transform 只做 ToTensor()（裁剪已处理尺寸）
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train

        if patch_size is not None:
            # 内部使用的 transform：仅转为 tensor，不再 Resize（避免和 patch 冲突）
            self.train_transform = transforms.ToTensor()
            self.target_transform = transforms.ToTensor()
        else:
            # 保持外部传入的 transform（兼容你原来的用法）
            self.train_transform = train_transform or transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()
            ])
            self.target_transform = target_transform or transforms.Compose([
                transforms.Resize(output_size),
                transforms.ToTensor()
            ])

        self.train_images = sorted([os.path.join(train_folder, img) for img in os.listdir(train_folder)])
        self.target_images = sorted([os.path.join(target_folder, img) for img in os.listdir(target_folder)])
        random.seed(seed)
        random.shuffle(self.train_images)
        random.seed(seed)
        random.shuffle(self.target_images)

        # 基本检查：确保数量匹配
        if len(self.train_images) != len(self.target_images):
            # 仍允许不同数量，但会按最小长度裁剪；这里仅打印警告
            print(f"警告：LR 图像数 ({len(self.train_images)}) 与 HR 图像数 ({len(self.target_images)}) 不一致。将按较小数量匹配。")
            minlen = min(len(self.train_images), len(self.target_images))
            self.train_images = self.train_images[:minlen]
            self.target_images = self.target_images[:minlen]

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        # 读取整张图像（可能是 LR 和 HR 对应的大小）
        lr_img = Image.open(self.train_images[index]).convert('RGB')
        hr_img = Image.open(self.target_images[index]).convert('RGB')

        if self.patch_size is None:
            # 原先行为：直接使用传入的 transform（可能包含 Resize）
            lr = self.train_transform(lr_img)
            hr = self.target_transform(hr_img)
            return lr, hr

        # ---------- patch 模式逻辑 ----------
        p = int(self.patch_size)
        s = int(self.scale_factor)
        p_hr = p * s  # 目标 HR patch 大小

        # 确保 LR 图像至少有 p x p，大于则随机/中心裁剪；否则先放大 LR 到所需大小，并按比例放大 HR
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size

        # 若 LR 尺寸不足，放大 LR 至至少 p，并将 HR 以相同比例放大（尽量保持对应关系）
        if lr_w < p or lr_h < p:
            new_lr_w = max(lr_w, p)
            new_lr_h = max(lr_h, p)
            lr_img = lr_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            # 将 HR 放大到 LR * scale（保持整数）
            hr_img = hr_img.resize((new_lr_w * s, new_lr_h * s), Image.BICUBIC)
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size

        # 若 HR 不足 p_hr，尝试按 lr 的尺寸乘以 scale 来调整 HR
        if hr_w < p_hr or hr_h < p_hr:
            # 若 HR 明显小于 lr*scale，则通过调整 HR 尺寸到 lr*s 保持配对
            expected_hr_w = lr_w * s
            expected_hr_h = lr_h * s
            if expected_hr_w >= p_hr and expected_hr_h >= p_hr:
                hr_img = hr_img.resize((expected_hr_w, expected_hr_h), Image.BICUBIC)
                hr_w, hr_h = hr_img.size
            else:
                # 兜底：直接放大 HR 至 p_hr
                hr_img = hr_img.resize((max(hr_w, p_hr), max(hr_h, p_hr)), Image.BICUBIC)
                hr_w, hr_h = hr_img.size

        # 计算 LR 的裁剪左上坐标（训练随机裁剪，验证中心裁剪）
        if self.is_train:
            if lr_w - p > 0:
                x = random.randint(0, lr_w - p)
            else:
                x = 0
            if lr_h - p > 0:
                y = random.randint(0, lr_h - p)
            else:
                y = 0
        else:
            x = max(0, (lr_w - p) // 2)
            y = max(0, (lr_h - p) // 2)

        # 裁剪 LR patch
        lr_patch = lr_img.crop((x, y, x + p, y + p))

        # HR patch 的左上角应与 LR 的对应（乘以 scale）
        hr_x = x * s
        hr_y = y * s

        # 如果 HR 图像尺寸与 LR*scale 对齐，则直接裁剪；否则尝试按相应尺度裁剪（上面已尽力调整）
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + p_hr, hr_y + p_hr))

        # 最后应用 ToTensor()（并保证值在 [0,1]）
        lr_tensor = self.train_transform(lr_patch)
        hr_tensor = self.target_transform(hr_patch)

        return lr_tensor, hr_tensor


# -------------------- 损失函数 (保留原 FFT 实现) --------------------
def create_highpass_mask(size, radius=10):
    """
    size: (H, W)
    radius: 中心低频半径（像素）
    返回 float mask，shape (H, W) ，中心为0（低频去除），其余为1
    """
    H, W = size
    cy, cx = H // 2, W // 2
    y = np.arange(0, H)[:, None]
    x = np.arange(0, W)[None, :]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask = (dist > radius).astype(np.float32)
    return torch.from_numpy(mask)


def high_frequency_loss(pred, target, radius=10):
    device = pred.device
    # 将图像在频域表示后做 fftshift（将零频移到中心）
    # pred/target 应为 real tensor (batch, C, H, W) ，对每通道独立做 fft2（复数）
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm='ortho'), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm='ortho'), dim=(-2, -1))

    mask = create_highpass_mask(pred.shape[-2:], radius=radius).to(device)  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    diff = (pred_fft - target_fft) * mask  # 复数
    mag_diff = torch.abs(diff)
    # mag_diff shape: (B, C, H, W) -> mean
    return torch.mean(mag_diff)


# -------------------- 注意力机制 --------------------
class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 修复：确保 hidden >= 1
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """卷积块注意力模块 (通道注意力 + 空间注意力)"""

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# -------------------- 核心网络模块 (增强版本，沿用你原设计并略微修正小问题) --------------------
class StripConv(nn.Module):
    """增强版条带卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=31):
        super().__init__()
        # 使用深度可分离卷积进一步轻量化
        self.h_conv = nn.Conv2d(in_channels, in_channels, (1, kernel_size),
                                padding=(0, kernel_size // 2), groups=in_channels)
        self.v_conv = nn.Conv2d(in_channels, in_channels, (kernel_size, 1),
                                padding=(kernel_size // 2, 0), groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        out = self.pw_conv(h_feat + v_feat)
        return self.activation(out)


class PartialLargeKernel(nn.Module):
    """增强版部分大核卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=31, ratio=0.25):
        super().__init__()
        self.ratio = ratio
        self.part_channels = int(in_channels * ratio) if int(in_channels * ratio) > 0 else 1

        # 使用深度可分离卷积
        self.h_conv = nn.Conv2d(self.part_channels, self.part_channels,
                                (1, kernel_size), padding=(0, kernel_size // 2),
                                groups=self.part_channels)
        self.v_conv = nn.Conv2d(self.part_channels, self.part_channels,
                                (kernel_size, 1), padding=(kernel_size // 2, 0),
                                groups=self.part_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        # 通道重排序增强信息交互
        x_shuffled = self.channel_shuffle(x)

        # 仅对部分通道应用大核卷积
        part_x = x_shuffled[:, :self.part_channels]
        h_feat = self.h_conv(part_x)
        v_feat = self.v_conv(part_x)
        large_k_feat = h_feat + v_feat

        # 合并特征
        merged = torch.cat([large_k_feat, x_shuffled[:, self.part_channels:]], dim=1)
        out = self.pw_conv(merged)
        return self.activation(out)

    def channel_shuffle(self, x, groups=4):
        # 更通用的 channel shuffle：自动调整 groups
        batch, channels, height, width = x.size()
        if channels % groups != 0:
            # 如果不能整除，退回到 groups=1（即不 shuffle）或寻找最大整除因子
            for g in range(groups, 0, -1):
                if channels % g == 0:
                    groups = g
                    break
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x


class AdaptiveFusionGate(nn.Module):
    """增强版自适应融合门"""

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 8), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 8), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, strip_feat, largek_feat):
        combined = strip_feat + largek_feat
        attn = self.attention(combined)
        return strip_feat * attn + largek_feat * (1 - attn)


class DFMB(nn.Module):
    """增强版双路径特征调制模块"""

    def __init__(self, channels, kernel_size=31, ratio=0.25):
        super().__init__()
        self.strip_path = StripConv(channels, channels, kernel_size)
        self.largek_path = PartialLargeKernel(channels, channels, kernel_size, ratio)
        self.fusion_gate = AdaptiveFusionGate(channels)

        # 添加注意力机制
        self.cbam = CBAM(channels, reduction=16)

        # 将残差缩放参数化为可学习变量
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        strip_feat = self.strip_path(x)
        largek_feat = self.largek_path(x)
        fused = self.fusion_gate(strip_feat, largek_feat)

        # 应用注意力机制
        fused = self.cbam(fused)

        alpha = torch.sigmoid(self.alpha)
        return fused * alpha + x


# -------------------- 多尺度 EdgeNet（修改点） --------------------
class EnhancedEdgeNet(nn.Module):
    """
    增强版边缘提取网络（多尺度输出）
    输出：list of tensors, length = num_scales（每个尺度的 edge 特征，与 encoder 每层对应）
    设计：先做若干基础 conv 提取，再使用一组具有不同 dilation 的 convs 生成多个尺度特征
    """

    def __init__(self, in_channels=3, out_channels=64, num_scales=8):
        super().__init__()
        self.num_scales = num_scales
        # 基础特征提取
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 构建多尺度 conv 列表（使用不同 dilation 以获得不同感受野）
        # dilation sequence: 1,1,2,2,3,3,4,4 ... 截断到 num_scales
        dilations = []
        d = 1
        while len(dilations) < num_scales:
            dilations.append(d)
            if len(dilations) < num_scales:
                dilations.append(d)
            d += 1
        dilations = dilations[:num_scales]

        self.scale_convs = nn.ModuleList()
        for dil in dilations:
            # padding = dilation to keep spatial size
            self.scale_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=dil, dilation=dil),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        # 基础特征
        identity = self.res_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        base = self.bn3(self.conv3(out))
        base = base + identity
        base = self.relu(base)

        # 生成多尺度特征（所有尺度保持与输入相同的空间分辨率）
        feats = []
        for conv in self.scale_convs:
            feats.append(conv(base))
        # feats: list of length num_scales, each (B, out_channels, H, W)
        return feats


# -------------------- 其余模块（保持） --------------------
class EnhancedEIFBlock(nn.Module):
    """增强版边缘信息融合块"""

    def __init__(self, in_channels_img, in_channels_edge, out_channels):
        super().__init__()

        # 使用深度可分离卷积减少参数
        self.conv_img = nn.Sequential(
            nn.Conv2d(in_channels_img, in_channels_img, 3, padding=1, groups=in_channels_img),
            nn.Conv2d(in_channels_img, out_channels, 1)
        )
        self.conv_edge = nn.Conv2d(in_channels_edge, out_channels, 1)

        # 增强版权重生成器
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, max(8, out_channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, out_channels // 4), 2, 1),
            nn.Softmax(dim=1)
        )

        # 添加注意力机制
        self.cbam = CBAM(out_channels, reduction=16)

    def forward(self, feat_img, feat_edge):
        proj_img = self.conv_img(feat_img)
        proj_edge = self.conv_edge(feat_edge)

        feat_cat = torch.cat([proj_img, proj_edge], dim=1)
        weights = self.weight_generator(feat_cat)
        w_img, w_edge = weights.chunk(2, dim=1)

        fused_feat = proj_img * w_img + proj_edge * w_edge

        # 应用注意力机制
        fused_feat = self.cbam(fused_feat)

        return fused_feat


class ProgressiveEdgeFusion(nn.Module):
    """保留原实现（但我们将在新的网络中按层逐一融合，不再依赖这个类）"""

    def __init__(self, num_blocks, img_channels=128, edge_channels=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.fusion_indices = [num_blocks // 4, num_blocks // 2, 3 * num_blocks // 4]
        self.eif_blocks = nn.ModuleList()
        for idx in self.fusion_indices:
            self.eif_blocks.append(EnhancedEIFBlock(img_channels, edge_channels, img_channels))

    def forward(self, deep_feat, edge_feat, block_idx):
        if block_idx in self.fusion_indices:
            fusion_idx = self.fusion_indices.index(block_idx)
            return self.eif_blocks[fusion_idx](deep_feat, edge_feat)
        return deep_feat


# -------------------- EnhancedFSLKNet（修改：每层融合） --------------------
class EnhancedFSLKNet(nn.Module):
    """增强版FSLKNet: 使用多尺度 edge_net 并在每个 dfm block 后融合对应 edge 特征"""

    def __init__(self, scale_factor=4, num_dfm_blocks=8, edge_channels=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_dfm_blocks = num_dfm_blocks

        # 浅层特征提取 - 增加到128通道
        self.shallow_extract = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU()
        )

        # 多尺度边缘特征提取网络：输出 list 长度为 num_dfm_blocks
        self.edge_net = EnhancedEdgeNet(in_channels=3, out_channels=edge_channels, num_scales=num_dfm_blocks)

        # 深层特征提取 (增强版DFMB模块堆叠)
        self.dfm_blocks = nn.ModuleList([DFMB(128) for _ in range(num_dfm_blocks)])

        # 为每个 dfm block 分配一个 EIF block 用以融合对应尺度的 edge 特征
        self.eif_blocks = nn.ModuleList([EnhancedEIFBlock(128, edge_channels, 128) for _ in range(num_dfm_blocks)])

        # 增强版图像重建模块，加入注意力机制
        upsample_blocks = int(math.log2(scale_factor))
        if 2 ** upsample_blocks != scale_factor:
            raise ValueError(f"scale_factor must be a power of 2, got {scale_factor}")

        modules = []
        in_ch = 128

        # 对于每次 PixelShuffle(2) 需要将通道扩展 4 倍
        for i in range(upsample_blocks):
            modules.extend([
                nn.Conv2d(in_ch, in_ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                CBAM(in_ch, reduction=16)  # 添加注意力机制
            ])

        # 最终输出层
        modules.extend([
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1)
        ])

        self.reconstruction = nn.Sequential(*modules)

    def forward(self, x):
        # 提取浅层特征
        shallow_feat = self.shallow_extract(x)

        # 提取多尺度边缘特征（list length num_dfm_blocks）
        edge_feats = self.edge_net(x)

        # 深层特征提取与逐层边缘融合
        deep_feat = shallow_feat
        for i, dfm_block in enumerate(self.dfm_blocks):
            # 双路径特征调制
            dfm_feat = dfm_block(deep_feat)
            # 在每一层使用对应的 edge 特征进行融合
            # edge_feats[i] 应与 dfm_feat 空间尺寸匹配
            deep_feat = self.eif_blocks[i](dfm_feat, edge_feats[i])

        # 全局残差连接
        deep_feat = deep_feat + shallow_feat

        # 图像重建
        return self.reconstruction(deep_feat)


# -------------------- VGG 感知损失模块（保持） --------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, requires_grad=False):
        super().__init__()
        self.device = device
        # 选择若干层作为 perceptual 特征
        self.vgg = None
        # 兼容不同 torchvision 版本的加载方法
        try:
            # modern torchvision API
            from torchvision.models import vgg19, VGG19_Weights
            self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except Exception:
            try:
                # older API
                self.vgg = models.vgg19(pretrained=True).features
            except Exception:
                # 若都不可用，使用 untrained vgg （功能仍可用，只是效果差）
                self.vgg = models.vgg19().features

        # 只取前几层（conv1_2, conv2_2, conv3_4 常用），这里选择到第16层左右
        self.selected_layers = ['3', '8', '15', '22']  # 对应 conv1_2, conv2_2, conv3_4, conv4_4 (索引视实现而略有差异)
        self.vgg = self.vgg.to(device).eval()
        if not requires_grad:
            for p in self.vgg.parameters():
                p.requires_grad = False

        # ImageNet normalization 用于 vgg 输入
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        """
        x, y: tensors in [0,1], shape (B,3,H,W)
        returns L1 distance between selected VGG features
        """
        # clamp & normalize
        x = torch.clamp(x, 0., 1.)
        y = torch.clamp(y, 0., 1.)
        # normalize
        x_in = (x - self.mean) / self.std
        y_in = (y - self.mean) / self.std

        loss = 0.0
        xi = x_in
        yi = y_in
        # iterate through vgg modules and accumulate at selected layers
        for name, module in self.vgg._modules.items():
            xi = module(xi)
            yi = module(yi)
            if name in self.selected_layers:
                loss += F.l1_loss(xi, yi)
        return loss


# -------------------- 训练配置 --------------------
# 我们仍保留原先定义的 train_transform/target_transform，但在 patch 模式下 dataset 会忽略它们并使用 ToTensor()
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 创建数据集：启用 patch 模式（patch_size=128, scale_factor=4），训练时随机裁剪，验证时中心裁剪
train_dataset = CircleDataset(train_folder, target_folder,
                              train_transform=train_transform,
                              target_transform=target_transform,
                              patch_size=128, scale_factor=4, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

val_dataset = CircleDataset(validation_folder, validation_target_folder,
                            train_transform=train_transform,
                            target_transform=target_transform,
                            patch_size=128, scale_factor=4, is_train=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 使用增强版FSLKNet模型（edge_net 会输出 num_dfm_blocks 个尺度），scale_factor 改为 4
model = EnhancedFSLKNet(scale_factor=4, num_dfm_blocks=8, edge_channels=64).to(device)

# 损失函数
criterion_l1 = nn.L1Loss().to(device)
criterion_mse = nn.MSELoss().to(device)
vgg_loss_module = VGGPerceptualLoss(device).to(device)  # 感知损失模块

# 优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

# -------------------- 训练循环（在你原来基础上增加 VGG loss 与监控指标） --------------------
best_val_loss = float('inf')
train_losses = []
val_losses = []

scaler = amp.GradScaler(enabled=(device.type == "cuda"))


def batch_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='none')
    # reduce per image
    mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_per_image = 10 * torch.log10(max_val ** 2 / (mse_per_image + 1e-10))
    return psnr_per_image.mean().item()


def batch_ssim(pred, target):
    # compute per-image SSIM via skimage (float in [0,1])
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()
    b = pred_np.shape[0]
    ssim_vals = []
    for i in range(b):
        # skimage expects H,W,C with channels last and floats
        p = np.transpose(pred_np[i], (1, 2, 0))
        t = np.transpose(targ_np[i], (1, 2, 0))
        try:
            val = ssim_np(p, t, data_range=1.0, multichannel=True)
        except TypeError:
            # older/newer skimage API compatibility
            val = ssim_np(p, t, data_range=1.0, channel_axis=2)
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


for epoch in range(epochs):
    # -------- 训练 --------
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0
    train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True)

    epoch_train_psnr = 0.0
    epoch_train_ssim = 0.0

    for inputs, targets in train_progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        use_cuda = (device.type == "cuda")

        # 前向
        with amp.autocast(device_type='cuda', enabled=use_cuda):
            outputs = model(inputs)
            l1_loss = criterion_l1(outputs, targets)
            mse_loss = criterion_mse(outputs, targets)

        # FFT损失在FP32下计算以提高数值稳定性（保持原实现）
        with torch.cuda.amp.autocast(enabled=False):
            fft_loss = high_frequency_loss(outputs.float(), targets.float(), radius=10)

            # 感知损失（使用 VGG，通常在 FP32 上更稳定）
            vgg_loss = vgg_loss_module(outputs, targets)

        # 固定权重组合（新增 VGG 权重）
        total_loss = L1_WEIGHT * l1_loss + FFT_WEIGHT * fft_loss + MSE_WEIGHT * mse_loss + VGG_WEIGHT * vgg_loss

        # backward
        scaler.scale(total_loss).backward()

        # 解除scale，进行梯度裁剪，防止爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        epoch_train_loss += float(total_loss.item())
        num_train_batches += 1

        # 监控训练 PSNR/SSIM（小批量）
        try:
            batch_ps = batch_psnr(outputs.detach(), targets.detach())
            batch_ss = batch_ssim(outputs.detach(), targets.detach())
            epoch_train_psnr += batch_ps
            epoch_train_ssim += batch_ss
        except Exception:
            # 若计算出错，不影响训练
            pass

        # 更新进度条信息
        if train_progress.n % 5 == 0 or train_progress.n == len(train_loader):
            train_progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                l1=f"{l1_loss.item():.4f}",
                fft=f"{fft_loss.item():.4f}",
                mse=f"{mse_loss.item():.4f}",
                vgg=f"{vgg_loss.item():.4f}"
            )

    # 训练集epoch平均loss
    avg_train_loss = epoch_train_loss / max(1, num_train_batches)
    train_losses.append(avg_train_loss)
    train_progress.close()

    avg_train_psnr = epoch_train_psnr / max(1, num_train_batches)
    avg_train_ssim = epoch_train_ssim / max(1, num_train_batches)

    # -------- 验证 --------
    model.eval()
    epoch_val_loss = 0.0
    num_val_batches = 0
    val_progress = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True)

    epoch_val_psnr = 0.0
    epoch_val_ssim = 0.0

    with torch.no_grad():
        for inputs, targets in val_progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp.autocast(device_type='cuda', enabled=use_cuda):
                outputs = model(inputs)
                val_l1 = criterion_l1(outputs, targets)
                val_mse = criterion_mse(outputs, targets)

            with torch.cuda.amp.autocast(enabled=False):
                val_fft = high_frequency_loss(outputs.float(), targets.float(), radius=10)
                val_vgg = vgg_loss_module(outputs, targets)

            total_val_loss = L1_WEIGHT * val_l1 + FFT_WEIGHT * val_fft + MSE_WEIGHT * val_mse + VGG_WEIGHT * val_vgg

            epoch_val_loss += float(total_val_loss.item())
            num_val_batches += 1

            # 监控指标
            try:
                val_ps = batch_psnr(outputs, targets)
                val_ss = batch_ssim(outputs, targets)
                epoch_val_psnr += val_ps
                epoch_val_ssim += val_ss
            except Exception:
                pass

            if val_progress.n % 5 == 0 or val_progress.n == len(val_loader):
                val_progress.set_postfix(val_loss=f"{total_val_loss.item():.4f}")

    # 验证集epoch平均loss
    avg_val_loss = epoch_val_loss / max(1, num_val_batches)
    val_losses.append(avg_val_loss)
    val_progress.close()

    avg_val_psnr = epoch_val_psnr / max(1, num_val_batches)
    avg_val_ssim = epoch_val_ssim / max(1, num_val_batches)

    # -------- 保存逻辑 --------
    if avg_val_loss <= best_val_loss:
        best_val_loss = avg_val_loss
        if epoch > 10:
            model_name = f"best_model_enhanced_fslknet_epoch{epoch + 1}_val{avg_val_loss:.4f}.pth"
            torch.save(model.state_dict(), model_name)
            print(f"Epoch {epoch + 1}: 保存最佳模型 {model_name}")

    if (epoch + 1) % 10 == 0:
        model_name = f"checkpoint_enhanced_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), model_name)
        print(f"Epoch {epoch + 1}: 保存检查点 {model_name}")

    # 学习率调度
    scheduler.step()

    # 控制台输出（包含监控指标）
    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
    print(f"Train PSNR: {avg_train_psnr:.4f} | Train SSIM: {avg_train_ssim:.4f}")
    print(f"Val   PSNR: {avg_val_psnr:.4f} | Val   SSIM: {avg_val_ssim:.4f}")
    print(f"损失权重: L1={L1_WEIGHT}, FFT={FFT_WEIGHT}, MSE={MSE_WEIGHT}, VGG={VGG_WEIGHT}")
