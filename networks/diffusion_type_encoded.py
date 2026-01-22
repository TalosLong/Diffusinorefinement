"""
Diffusion Refinement Module for CPUNet Segmentation

This module implements a conditional diffusion model for refining coarse segmentation results.
The diffusion model is conditioned on:
1. The original input image
2. The coarse segmentation mask (soft probability map) from CPUNet

Training:
- Forward diffusion noise is applied only to the ground truth mask
- Network learns to predict noise residual at multiple noise scales

Inference:
- Start from the coarse mask aligned to an intermediate diffusion step
- Use DDIM reverse sampling for progressive denoising
- Output refined segmentation with improved boundary quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings for diffusion models.
    
    Args:
        timesteps: 1D tensor of timesteps (batch_size,)
        embedding_dim: Dimension of the output embedding
        
    Returns:
        Tensor of shape (batch_size, embedding_dim)
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


class GaussianDiffusion:
    """
    Gaussian diffusion scheduler for forward and reverse process.
    
    Implements linear beta schedule and provides methods for:
    - q_sample: Forward diffusion (adding noise)
    - p_sample: Reverse diffusion (denoising)
    - DDIM sampling for faster inference
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        """
        Initialize diffusion scheduler.
        
        Args:
            num_timesteps: Total number of diffusion steps
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            schedule_type: Type of beta schedule ('linear' or 'cosine')
        """
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif schedule_type == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
            alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def to(self, device: torch.device) -> 'GaussianDiffusion':
        """Move all tensors to specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_start: Clean data x_0 (ground truth mask)
            t: Timesteps tensor (batch_size,)
            noise: Optional pre-sampled noise
            
        Returns:
            Tuple of (noisy_x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_noisy, noise
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t
    
    def ddim_sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        image_cond: torch.Tensor,
        coarse_mask_cond: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Perform one DDIM sampling step.
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict logits of x0
        with torch.no_grad():
            pred_logits = model(x_t, t_tensor, image_cond, coarse_mask_cond)
            
        # Convert logits to x0 in [-1, 1] range
        if pred_logits.shape[1] == 1:  # Binary
            pred_prob = torch.sigmoid(pred_logits)
        else:
            pred_prob = F.softmax(pred_logits, dim=1)
            
        # Scale to [-1, 1] for diffusion process
        pred_x0 = pred_prob * 2 - 1
        
        # Compute alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        
        # Derive implied noise
        # x_t = sqrt(alpha_t) * x0 + sqrt(1-alpha_t) * noise
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        
        # Avoid division by zero at t=0 (alpha_t=1)
        if t == 0:
            pred_noise = torch.zeros_like(x_t)
        else:
            pred_noise = (x_t - sqrt_alpha_t * pred_x0) / sqrt_one_minus_alpha_t
        
        # DDIM update
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * pred_noise
        
        if eta > 0 and t_prev > 0:
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
        else:
            x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple,
        image_cond: torch.Tensor,
        coarse_mask_cond: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        start_from_coarse: bool = True,
        start_timestep: Optional[int] = None,
        x_start: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cuda')
    ) -> torch.Tensor:
        """
        DDIM sampling for inference.
        
        Args:
            model: Denoising model
            shape: Shape of output (batch_size, channels, height, width)
            image_cond: Original image condition
            coarse_mask_cond: Coarse segmentation mask condition
            num_inference_steps: Number of DDIM steps
            eta: DDIM eta (0 for deterministic)
            start_from_coarse: Whether to start from coarse mask
            start_timestep: Starting timestep for refinement
            device: Device to run on
            
        Returns:
            Refined segmentation mask
        """
        # Create timestep schedule for DDIM
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[:num_inference_steps]
        timesteps = list(reversed(timesteps))
        
        # Initialize x_T
        if start_from_coarse and start_timestep is not None:
            # Align coarse mask to intermediate diffusion step
            # This adds noise to the coarse mask to match the noise level at start_timestep
            noise = torch.randn(shape, device=device)
            t_start = torch.full((shape[0],), start_timestep, device=device, dtype=torch.long)
            
            # Use provided x_start (scaled) if available, otherwise coarse_mask_cond
            start_latents = x_start if x_start is not None else coarse_mask_cond
            x = self.q_sample(start_latents, t_start, noise)[0]
            
            # Adjust timestep list to start from start_timestep
            timesteps = [t for t in timesteps if t <= start_timestep]
        else:
            x = torch.randn(shape, device=device)
        
        # DDIM reverse sampling
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            x = self.ddim_sample_step(model, x, t, t_prev, image_cond, coarse_mask_cond, eta)
        
        return x


class ConditionalResBlock(nn.Module):
    """
    Residual block with time and condition embedding for diffusion model.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for diffusion U-Net."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        
        return x + attn_out


class ResBlock(nn.Module):
    """Single residual block for encoder/decoder."""
    
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = ConditionalResBlock(in_ch, out_ch, time_emb_dim, dropout)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.block(x, t_emb)


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PFMM(nn.Module):
    """
    Prior Feature Modulation Module
    利用粗略概率图 P 作为加权因子，调制网络输入 Z
    """
    def __init__(self, in_channels, num_classes, out_channels=None):
        """
        Args:
            in_channels: 输入特征 Z 的通道数
            num_classes: 类别数 K (概率图 P 的通道数)
            out_channels: 输出通道数，默认与 in_channels 相同
        """
        super(PFMM, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        
        # 拼接后的通道数:  K * in_channels
        concat_channels = num_classes * in_channels
        
        # 深度可分离卷积 (Depthwise Separable Convolution)
        self.dwconv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(concat_channels, concat_channels, kernel_size=3, 
                      padding=1, groups=concat_channels, bias=False),
            # Pointwise convolution
            nn.Conv2d(concat_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # Layer Normalization (对应公式21中的 LN)
        self.ln1 = nn.LayerNorm(in_channels)
        
        # Conv-LayerNorm-ReLU block (对应公式22)
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.ln2 = nn.LayerNorm(self.out_channels)
        self.relu = nn. ReLU(inplace=True)
    
    def forward(self, Z, P):
        """
        Args:
            Z: 网络输入特征, shape:  (B, C, H, W)
            P: 粗略概率图, shape:  (B, K, H, W), K 为类别数
        Returns:
            Z''':  输出特征, shape: (B, out_channels, H, W)
        """
        B, C, H, W = Z.shape
        
        # 公式19:  按类别通道拆分 P
        # P_split:  list of K tensors, each with shape (B, 1, H, W)
        P_split = torch.split(P, 1, dim=1)  # [P1, P2, .. ., PK]
        
        # 公式20: 逐元素相乘后拼接
        # Z' = Concat(P1 ⊙ Z, P2 ⊙ Z, ..., PK ⊙ Z)
        weighted_features = [p * Z for p in P_split]  # 每个 Pk ⊙ Z:  (B, C, H, W)
        Z_prime = torch.cat(weighted_features, dim=1)  # (B, K*C, H, W)
        
        # 公式21: Z'' = LN(DWConv(Z') ⊕ Z)
        dwconv_out = self.dwconv(Z_prime)  # (B, C, H, W)
        Z_double_prime = dwconv_out + Z    # 残差连接
        
        # LayerNorm 需要调整维度:  (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        Z_double_prime = Z_double_prime.permute(0, 2, 3, 1)  # (B, H, W, C)
        Z_double_prime = self.ln1(Z_double_prime)
        Z_double_prime = Z_double_prime.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 公式22: Z''' = ReLU(LN(Conv(Z'')))
        Z_triple_prime = self. conv(Z_double_prime)  # (B, out_channels, H, W)
        Z_triple_prime = Z_triple_prime.permute(0, 2, 3, 1)  # (B, H, W, out_channels)
        Z_triple_prime = self.ln2(Z_triple_prime)
        Z_triple_prime = Z_triple_prime. permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        Z_triple_prime = self.relu(Z_triple_prime)
        
        return Z_triple_prime


class CFFM(nn.Module):
    """
    Conditional Feature Fusion Module
    将原始图像的低级特征作为条件信息注入去噪过程
    """
    def __init__(self, f_channels, x_channels, out_channels=None):
        """
        Args:
            f_channels: 特征图 F 的通道数 (来自 U-Net encoder 上一层)
            x_channels: 低级特征 Xlow 的通道数
            out_channels: 输出通道数，默认与 f_channels 相同
        """
        super(CFFM, self).__init__()
        
        self.f_channels = f_channels
        self.out_channels = out_channels if out_channels else f_channels
        
        # 拼接后的通道数
        concat_channels = f_channels + x_channels
        
        # DWConv-LN-ReLU block (对应公式24)
        self.dwconv_block = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(concat_channels, concat_channels, kernel_size=3,
                      padding=1, groups=concat_channels, bias=False),
            # Pointwise convolution - 将通道数映射回 f_channels
            nn.Conv2d(concat_channels, f_channels, kernel_size=1, bias=False)
        )
        
        # Layer Normalization for DWConv output
        self.ln1 = nn.LayerNorm(f_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Final Layer Normalization (对应公式25)
        self.ln2 = nn. LayerNorm(self.out_channels)
        
        # 如果输出通道数不同，需要投影层
        self.proj = None
        if self.out_channels != f_channels:
            self.proj = nn.Conv2d(f_channels, self.out_channels, kernel_size=1, bias=False)
    
    def forward(self, F, Xlow):
        """
        Args:
            F: U-Net encoder 上一层的特征图, shape: (B, C_f, H, W)
            Xlow: 原始图像的低级特征, shape: (B, C_x, H, W)
                  需要与 F 的分辨率匹配
        Returns:
            F''': 输出特征, shape: (B, out_channels, H, W)
        """
        # 公式23: F' = Concat(F, Xlow)
        F_prime = torch.cat([F, Xlow], dim=1)  # (B, C_f + C_x, H, W)
        
        # 公式24: F'' = ReLU(LN(DWConv(F')))
        F_double_prime = self.dwconv_block(F_prime)  # (B, C_f, H, W)
        
        # LayerNorm 需要调整维度
        F_double_prime = F_double_prime.permute(0, 2, 3, 1)  # (B, H, W, C_f)
        F_double_prime = self.ln1(F_double_prime)
        F_double_prime = F_double_prime. permute(0, 3, 1, 2)  # (B, C_f, H, W)
        F_double_prime = self.relu(F_double_prime)
        
        # 公式25: F''' = LN(F'' ⊕ F)
        F_triple_prime = F_double_prime + F  # 残差连接
        
        # 如果需要投影
        if self.proj is not None:
            F_triple_prime = self.proj(F_triple_prime)
        
        # Final LayerNorm
        F_triple_prime = F_triple_prime.permute(0, 2, 3, 1)
        F_triple_prime = self.ln2(F_triple_prime)
        F_triple_prime = F_triple_prime. permute(0, 3, 1, 2)
        
        return F_triple_prime


class SACM(nn.Module):
    """
    Spatial and Channel Attention Module
    将原始图像的高级特征注入到 U-Net encoder 的高级语义特征中
    基于通道注意力和空间注意力机制
    """
    def __init__(self, x_channels, s_channels, reduction_ratio=16):
        """
        Args:
            x_channels:  高级特征 Xhigh 的通道数
            s_channels: 语义特征 S 的通道数 (U-Net encoder 输出)
            reduction_ratio: MLP 中的通道缩减比例
        """
        super(SACM, self).__init__()
        
        self.x_channels = x_channels
        self.s_channels = s_channels
        
        # 对 Xhigh 的卷积层 (公式26, 27中的 Conv)
        self.conv_xhigh = nn.Conv2d(x_channels, s_channels, kernel_size=3, 
                                     padding=1, bias=False)
        
        # 全局池化层 (公式26, 27)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 共享 MLP 用于通道注意力 (公式28)
        reduced_channels = max(s_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn. Linear(s_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, s_channels, bias=False)
        )
        
        # 空间注意力的深度可分离卷积 (公式29)
        self.spatial_dwconv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(2 * s_channels, 2 * s_channels, kernel_size=7,
                      padding=3, groups=2 * s_channels, bias=False),
            # Pointwise convolution - 输出单通道空间注意力图
            nn.Conv2d(2 * s_channels, 1, kernel_size=1, bias=False)
        )
        
        # 对 S 的卷积层 (公式30)
        self.conv_s = nn.Conv2d(s_channels, s_channels, kernel_size=3, 
                                 padding=1, bias=False)
        
        # Sigmoid 激活
        self.sigmoid = nn. Sigmoid()
        
        # 最终的 Layer Normalization (公式31)
        self.ln = nn.LayerNorm(s_channels)
    
    def forward(self, S, Xhigh):
        """
        Args:
            S:  U-Net encoder 输出的高级语义特征, shape: (B, C_s, H, W)
            Xhigh: 原始图像的高级特征, shape: (B, C_x, H, W)
        Returns:
            S'':  融合后的输出特征, shape: (B, C_s, H, W)
        """
        B, C_s, H, W = S.shape
        
        # ============ 通道注意力 (Channel Attention) ============
        # 对 Xhigh 进行卷积
        X_conv = self.conv_xhigh(Xhigh)  # (B, C_s, H, W)
        
        # 公式26: X'_max = MaxPool(Conv(Xhigh))
        X_max = self. global_max_pool(X_conv)  # (B, C_s, 1, 1)
        X_max = X_max.view(B, C_s)  # (B, C_s)
        
        # 公式27: X'_avg = AvgPool(Conv(Xhigh))
        X_avg = self.global_avg_pool(X_conv)  # (B, C_s, 1, 1)
        X_avg = X_avg.view(B, C_s)  # (B, C_s)
        
        # 公式28: CA = Sigmoid(MLP(X'_max) + MLP(X'_avg))
        CA = self.sigmoid(self.mlp(X_max) + self.mlp(X_avg))  # (B, C_s)
        CA = CA.view(B, C_s, 1, 1)  # (B, C_s, 1, 1) 用于广播
        
        # ============ 空间注意力 (Spatial Attention) ============
        # 保留空间维度的池化特征
        X_max_spatial = torch.max(X_conv, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        X_max_spatial = X_max_spatial.expand(-1, C_s, -1, -1)  # (B, C_s, H, W)
        
        X_avg_spatial = torch.mean(X_conv, dim=1, keepdim=True)  # (B, 1, H, W)
        X_avg_spatial = X_avg_spatial.expand(-1, C_s, -1, -1)  # (B, C_s, H, W)
        
        # 公式29: SA = Sigmoid(DWConv(Concat(X'_max, X'_avg)))
        X_concat = torch.cat([X_max_spatial, X_avg_spatial], dim=1)  # (B, 2*C_s, H, W)
        SA = self.sigmoid(self.spatial_dwconv(X_concat))  # (B, 1, H, W)
        
        # ============ 特征融合 ============
        # 公式30: S' = Conv(S)
        S_prime = self.conv_s(S)  # (B, C_s, H, W)
        
        # 公式31: S'' = LN(S ⊕ (CA ⊙ S') ⊕ (SA ⊙ S'))
        CA_weighted = CA * S_prime  # 通道注意力加权:  (B, C_s, H, W)
        SA_weighted = SA * S_prime  # 空间注意力加权: (B, C_s, H, W)
        
        S_double_prime = S + CA_weighted + SA_weighted  # 残差连接
        
        # Layer Normalization
        S_double_prime = S_double_prime.permute(0, 2, 3, 1)  # (B, H, W, C_s)
        S_double_prime = self.ln(S_double_prime)
        S_double_prime = S_double_prime.permute(0, 3, 1, 2)  # (B, C_s, H, W)
        
        return S_double_prime


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
    
def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                    padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                    padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*16, cout=width*32, cmid=width*8, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*32, cout=width*32, cmid=width*8)) for i in range(2, block_units[3] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
    

class ConditionalDenoisingUNet(nn.Module):
    """
    Conditional U-Net for denoising in diffusion model.
    
    Conditioned on:
    1. Original image (concatenated to input)
    2. Coarse segmentation mask (concatenated to input)
    3. Timestep (via embedding)
    
    This is a simpler, more robust implementation that properly handles
    skip connections with correct channel dimensions.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # Noisy mask channels (1 for binary segmentation)
        image_channels: int = 3,  # Original image channels
        coarse_mask_channels: int = 1,  # Coarse mask channels (soft probability)
        out_channels: int = 1,  # Predicted noise channels
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,  # Number of segmentation classes
        input_resolution: int = 256,  # Input resolution (configurable)
    ):
        """
        Initialize conditional denoising U-Net.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        
        # For multi-class, we use num_classes channels instead of 1
        actual_in_channels = num_classes if num_classes > 2 else 1
        actual_out_channels = num_classes if num_classes > 2 else 1
        
        # Total input channels: noisy mask + image + coarse mask
        total_in_channels = actual_in_channels + image_channels + actual_in_channels
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(total_in_channels, base_channels, 3, padding=1)
        self.PFMM = PFMM(in_channels=base_channels, num_classes=num_classes)
        self.Lowencoder = ResNetV2(block_units=(3, 4, 9), width_factor=1)
        self.CFFM = CFFM(f_channels=base_channels, x_channels=64)
        self.cffms = nn.ModuleList([
            CFFM(f_channels=base_channels, x_channels=64) for ch in self.enc_channels[:-1]
        ])
        # ============== ENCODER ==============
        # We store each block's output channels and track skips
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_res = input_resolution
        in_ch = base_channels
        self.skip_channels = [base_channels]  # First skip is from init_conv
        
        for level_idx, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            
            # Res blocks at this level
            for block_idx in range(num_res_blocks):
                block_in = in_ch if block_idx == 0 else out_ch
                self.encoder_blocks.append(ResBlock(block_in, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                self.skip_channels.append(out_ch)
            
            # Attention at this level
            if current_res in attention_resolutions:
                self.encoder_attns.append(AttentionBlock(out_ch))
            else:
                self.encoder_attns.append(nn.Identity())
            
            # Downsample (except last level)
            if level_idx < len(channel_mult) - 1:
                self.downsamples.append(Downsample(out_ch))
                self.skip_channels.append(out_ch)
                current_res //= 2
        
        # ============== MIDDLE ==============
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim, dropout)
        
        # ============== DECODER ==============
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Reverse iterate through levels
        for level_idx, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            
            # Res blocks at this level (each takes a skip connection)
            for block_idx in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                block_in = in_ch + skip_ch
                self.decoder_blocks.append(ResBlock(block_in, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
            
            # Attention at this level
            if current_res in attention_resolutions:
                self.decoder_attns.append(AttentionBlock(out_ch))
            else:
                self.decoder_attns.append(nn.Identity())
            
            # Upsample (except last level)
            if level_idx < len(channel_mult) - 1:
                self.upsamples.append(Upsample(out_ch))
                current_res *= 2
        
        # Output
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, actual_out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        image_cond: torch.Tensor,
        coarse_mask_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy mask tensor (B, C, H, W)
            t: Timestep tensor (B,)
            image_cond: Original image condition (B, 3, H, W)
            coarse_mask_cond: Coarse segmentation mask (B, C, H, W)
            
        Returns:
            Predicted noise tensor (B, C, H, W)
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate inputs
        # x (8,1,256,256) image_cond (8,3,256,256) coarse_mask_cond (8,1,256,256)--- IGNORE ---
        # x = torch.cat([x, image_cond, coarse_mask_cond], dim=1)
        # 使用 PFMM 模块调制输入特征
        x = self.PFMM(x, coarse_mask_cond)  

        # 提取原始图像的低级特征
        _ , imagefeature = self.Lowencoder(image_cond)  # (B, 64, H, W)

        # 融合低级特征到初始特征
        
        # x (8,5,256,256) --- IGNORE ---
        # Initial convolution
        h = self.init_conv(x)
        skips = [h]
        
        # ============== ENCODER ==============
        block_idx = 0
        attn_idx = 0
        down_idx = 0
        
        for level_idx in range(len(self.channel_mult)):
            # Res blocks
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx += 1
            
            # Attention
            h = self.encoder_attns[attn_idx](h)
            attn_idx += 1
            
            # Downsample
            if level_idx < len(self.channel_mult) - 1:
                h = self.downsamples[down_idx](h)
                skips.append(h)
                down_idx += 1
        
        # ============== MIDDLE ==============
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # ============== DECODER ==============
        block_idx = 0
        attn_idx = 0
        up_idx = 0
        
        for level_idx in range(len(self.channel_mult)):
            # Res blocks (with skip connections)
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            # Attention
            h = self.decoder_attns[attn_idx](h)
            attn_idx += 1
            
            # Upsample
            if level_idx < len(self.channel_mult) - 1:
                h = self.upsamples[up_idx](h)
                up_idx += 1
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


class DiffusionRefiner(nn.Module):
    """
    Complete Diffusion Refinement model.
    
    Combines:
    1. Frozen CPUNet for coarse segmentation
    2. Conditional diffusion model for refinement
    """
    
    def __init__(
        self,
        cpunet: nn.Module,
        diffusion_config: Optional[dict] = None,
        num_timesteps: int = 1000,
        num_classes: int = 2,
        freeze_cpunet: bool = True
    ):
        """
        Initialize DiffusionRefiner.
        
        Args:
            cpunet: Pre-trained CPUNet segmentation model
            diffusion_config: Configuration for diffusion model
            num_timesteps: Number of diffusion timesteps
            num_classes: Number of segmentation classes
            freeze_cpunet: Whether to freeze CPUNet weights
        """
        super().__init__()
        
        self.cpunet = cpunet
        self.num_classes = num_classes
        
        # Freeze CPUNet if required
        if freeze_cpunet:
            for param in self.cpunet.parameters():
                param.requires_grad = False
            self.cpunet.eval()
        
        # Initialize diffusion scheduler
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            schedule_type="linear"
        )
        
        # Initialize denoising network
        default_config = {
            'in_channels': 1,
            'image_channels': 3,
            'coarse_mask_channels': 1,
            'out_channels': 1,
            'base_channels': 64,
            'channel_mult': (1, 2, 4, 8),
            'num_res_blocks': 2,
            'attention_resolutions': (16, 8),
            'time_emb_dim': 256,
            'dropout': 0.1,
            'num_classes': num_classes
        }
        
        if diffusion_config is not None:
            default_config.update(diffusion_config)
        
        self.denoising_net = ConditionalDenoisingUNet(**default_config)
        
        self.num_timesteps = num_timesteps
    
    def get_coarse_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get coarse segmentation mask from CPUNet.
        
        Args:
            image: Input image (B, C, H, W)
            
        Returns:
            Soft probability mask (B, num_classes, H, W) or (B, 1, H, W) for binary
        """
        with torch.no_grad():
            self.cpunet.eval()
            logits = self.cpunet(image)  # (B, num_classes, H, W)
            probs = F.softmax(logits, dim=1)
            
            if self.num_classes == 2:
                # For binary segmentation, use foreground probability
                return probs[:, 1:2, :, :]
            else:
                return probs
    
    def training_step(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Training step for diffusion refinement.
        
        Args:
            image: Input image (B, C, H, W)
            gt_mask: Ground truth mask (B, H, W) or (B, 1, H, W)
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Ensure diffusion scheduler is on correct device
        self.diffusion.to(device)
        
        # Get coarse mask from frozen CPUNet
        coarse_mask = self.get_coarse_mask(image)
        
        # --- 1. Condition Augmentation ---
        # Randomly perturb coarse mask to simulate inference likelihood
        # This makes the refiner more robust to bad coarse predictions
        if self.training:
            # 50% chance to apply augmentation
            if torch.rand(1).item() < 0.5:
                # Randomly erode or dilate (using max pooling)
                if torch.rand(1).item() < 0.5:
                    # Dilation: max_pool expands high values (1s)
                    coarse_mask = F.max_pool2d(coarse_mask, kernel_size=3, stride=1, padding=1)
                else:
                    # Erosion: -max_pool(-x) shrinks high values
                    coarse_mask = -F.max_pool2d(-coarse_mask, kernel_size=3, stride=1, padding=1)
                
                # Add small Gaussian noise to probabilities helps escape local optima
                noise_perturb = torch.randn_like(coarse_mask) * 0.05
                coarse_mask = torch.clamp(coarse_mask + noise_perturb, 0, 1)

        # Prepare ground truth mask for diffusion
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)  # (B, 1, H, W)
        
        if self.num_classes == 2:
            # Binary mask: convert to float [0, 1] then scale to [-1, 1]
            x_start = gt_mask.float() * 2 - 1
        else:
            # Multi-class: one-hot encode and scale
            x_start = F.one_hot(gt_mask.long().squeeze(1), self.num_classes)
            x_start = x_start.permute(0, 3, 1, 2).float() * 2 - 1
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Forward diffusion: add noise to ground truth
        x_noisy, noise = self.diffusion.q_sample(x_start, t)
        
        # Predict logits of x0
        pred_logits = self.denoising_net(x_noisy, t, image, coarse_mask)
        
        # --- 2. Loss Calculation ---
        
        if self.num_classes == 2:
            # Binary Case
            # BCEWithLogitsLoss expects float targets
            loss_ce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask.float())
            pred_prob = torch.sigmoid(pred_logits)
            
            target_prob = gt_mask.float()
        else:
            # Multi-class Case
            # CrossEntropyLoss expects class indices as targets (long)
            loss_ce = F.cross_entropy(pred_logits, gt_mask.long().squeeze(1))
            pred_prob = F.softmax(pred_logits, dim=1)
            
            # For dice, we need one-hot target
            target_prob = F.one_hot(gt_mask.long().squeeze(1), self.num_classes).permute(0, 3, 1, 2).float()
            
        # Dice Loss calculation
        smooth = 1e-5
        intersection = (pred_prob * target_prob).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target_prob.sum(dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (union + smooth)
        loss_dice = 1.0 - dice_score.mean()
        
        # Combined loss
        loss = loss_ce + loss_dice
        
        metrics = {
            'loss': loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_dice': loss_dice.item(),
            'dice_score': dice_score.mean().item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 50,
        start_timestep: Optional[int] = None,
        eta: float = 0.0,
        return_coarse: bool = False
    ) -> torch.Tensor:
        """
        Inference: refine segmentation using diffusion.
        
        Args:
            image: Input image (B, C, H, W)
            num_inference_steps: Number of DDIM steps
            start_timestep: Starting timestep for refinement (default: 70% of total)
            eta: DDIM eta parameter
            return_coarse: Whether to also return coarse mask
            
        Returns:
            Refined segmentation mask (probabilities)
        """
        device = image.device
        batch_size = image.shape[0]
        h, w = image.shape[2:]
        
        # Ensure diffusion scheduler is on correct device
        self.diffusion.to(device)
        
        # Get coarse mask from CPUNet
        coarse_mask = self.get_coarse_mask(image)
        
        # Default start timestep: 70% of total (moderately noisy)
        if start_timestep is None:
            start_timestep = int(0.7 * self.num_timesteps)
        
        # Determine shape
        if self.num_classes == 2:
            shape = (batch_size, 1, h, w)
        else:
            shape = (batch_size, self.num_classes, h, w)
        
        # Scale coarse mask to [-1, 1] for diffusion
        coarse_mask_scaled = coarse_mask * 2 - 1
        
        # Run DDIM sampling starting from coarse mask
        refined = self.diffusion.ddim_sample(
            model=self.denoising_net,
            shape=shape,
            image_cond=image,
            coarse_mask_cond=coarse_mask,
            num_inference_steps=num_inference_steps,
            eta=eta,
            start_from_coarse=True,
            start_timestep=start_timestep,
            x_start=coarse_mask_scaled,
            device=device
        )
        
        # Scale back to [0, 1]
        refined = (refined + 1) / 2
        refined = torch.clamp(refined, 0, 1)
        
        if return_coarse:
            return refined, coarse_mask
        return refined
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            image: Input image (B, C, H, W)
            
        Returns:
            Refined segmentation mask
        """
        return self.inference(image)


def create_diffusion_refiner(
    cpunet: nn.Module,
    num_classes: int = 2,
    num_timesteps: int = 1000,
    base_channels: int = 64,
    freeze_cpunet: bool = True
) -> DiffusionRefiner:
    """
    Factory function to create a DiffusionRefiner model.
    
    Args:
        cpunet: Pre-trained CPUNet model
        num_classes: Number of segmentation classes
        num_timesteps: Number of diffusion timesteps
        base_channels: Base channels for denoising U-Net
        freeze_cpunet: Whether to freeze CPUNet weights
        
    Returns:
        DiffusionRefiner model
    """
    diffusion_config = {
        'base_channels': base_channels,
        'num_classes': num_classes
    }
    
    return DiffusionRefiner(
        cpunet=cpunet,
        diffusion_config=diffusion_config,
        num_timesteps=num_timesteps,
        num_classes=num_classes,
        freeze_cpunet=freeze_cpunet
    )
