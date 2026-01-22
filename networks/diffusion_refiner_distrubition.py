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
from typing import Optional, Tuple, List, Union


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
        eta: float = 0.0,
        cpunet_features: Optional[List[torch.Tensor]] = None,
        high_level_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform one DDIM sampling step.
        
        Args:
            model: Denoising model
            x_t: Current noisy sample
            t: Current timestep
            t_prev: Previous timestep (target)
            image_cond: Original image condition
            coarse_mask_cond: Coarse segmentation mask condition
            eta: DDIM eta parameter (0 for deterministic sampling)
            
        Returns:
            Denoised sample at t_prev
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            pred_noise = model(x_t, t_tensor, image_cond, coarse_mask_cond, cpunet_features, high_level_feat)
        
        # Compute alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip to valid range
        
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
        device: torch.device = torch.device('cuda'),
        cpunet_features: Optional[List[torch.Tensor]] = None,
        high_level_feat: Optional[torch.Tensor] = None
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
            x = self.ddim_sample_step(model, x, t, t_prev, image_cond, coarse_mask_cond, eta, cpunet_features, high_level_feat)
        
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

        # CHANGE 1: Alpha parameter for guided fusion
        self.param_alpha = nn.Parameter(torch.tensor(0.0))
        
        # CHANGE 2: CPUNet Feature Adaptors (Improved)
        # Apply GroupNorm -> SiLU -> Conv.
        # Initialize last conv to Zero to ensure identity mapping at start of training.
        self.cpunet_adaptors = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, 64), nn.SiLU(),
                nn.Conv2d(64, base_channels, 1)
            ),
            nn.Sequential(
                nn.GroupNorm(32, 256), nn.SiLU(),
                nn.Conv2d(256, base_channels * 2, 1)
            ),
            nn.Sequential(
                nn.GroupNorm(32, 512), nn.SiLU(),
                nn.Conv2d(512, base_channels * 4, 1)
            )
        ])
        
        # Zero-initialize the last convolution in each adaptor
        for adaptor in self.cpunet_adaptors:
            nn.init.zeros_(adaptor[-1].weight)
            nn.init.zeros_(adaptor[-1].bias)

        # CHANGE 3: High Level Attention Adaptor (Improved)
        # High Level Feat (S16, 1024ch) -> Diffusion Bottleneck (S8, 512ch)
        self.high_level_attn_conv = nn.Sequential(
            nn.GroupNorm(32, 1024),
            nn.SiLU(),
            nn.Conv2d(1024, base_channels * 8, 1)
        )
        # Zero-initialize
        nn.init.zeros_(self.high_level_attn_conv[-1].weight)
        nn.init.zeros_(self.high_level_attn_conv[-1].bias)
        
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
        coarse_mask_cond: torch.Tensor,
        cpunet_features: Optional[List[torch.Tensor]] = None,
        high_level_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy mask tensor (B, C, H, W)
            t: Timestep tensor (B,)
            image_cond: Original image condition (B, 3, H, W)
            coarse_mask_cond: Coarse segmentation mask (B, C, H, W)
            cpunet_features: List of CPUNet encoder features
            high_level_feat: CPUNet high level feature (S16)
            
        Returns:
            Predicted noise tensor (B, C, H, W)
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Concatenate inputs 直接拼接-》Z_guided = Z * (1 + α * P)

        x = torch.cat([x, image_cond, coarse_mask_cond], dim=1)

        
        # Initial convolution
        h = self.init_conv(x)

        # CHANGE 1: Guided Fusion Z_guided = Z * (1 + alpha * P)
        # alpha is self.param_alpha. P is coarse_mask_cond.
        h = h * (1.0 + self.param_alpha * coarse_mask_cond)

        skips = [h]
        
        # ============== ENCODER ==============
        block_idx = 0
        attn_idx = 0
        down_idx = 0
        
        for level_idx in range(len(self.channel_mult)):
            
            # CHANGE 2: Encoder Feature Fusion
            # F = F + Conv(low_level_feat). Applied at start of level.
            if cpunet_features is not None:
                feat = None
                # Mapping: Level 1 (S2) -> cpunet[2] (S2)
                # Level 2 (S4) -> cpunet[1] (S4)
                # Level 3 (S8) -> cpunet[0] (S8)
                
                if level_idx == 1:
                    feat = self.cpunet_adaptors[0](cpunet_features[2])
                elif level_idx == 2:
                    feat = self.cpunet_adaptors[1](cpunet_features[1])
                elif level_idx == 3:
                    feat = self.cpunet_adaptors[2](cpunet_features[0])
                
                if feat is not None:
                    # Verify shape relative to h
                    if feat.shape[-2:] != h.shape[-2:]:
                         feat = F.interpolate(feat, size=h.shape[-2:], mode='bilinear', align_corners=False)
                    h = h + feat

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

        # CHANGE 3: Bottleneck Attention
        # Z = Z + Z * tanh(Conv(X_high))
        if high_level_feat is not None:
            # high_level_feat is 1024ch (S16). h is 512ch (S8).
            x_high = self.high_level_attn_conv(high_level_feat) # 1024->512
            # Upsample S16 to S8
            x_high = F.interpolate(x_high, size=h.shape[-2:], mode='bilinear', align_corners=False)
            attn = torch.tanh(x_high)
            h = h + h * attn

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
    
    def get_coarse_mask_and_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Get coarse segmentation mask and features from CPUNet.
        """
        with torch.no_grad():
            self.cpunet.eval()
            # Updated to request features
            logits, high_level_feat, features = self.cpunet(image, return_features=True)
            probs = F.softmax(logits, dim=1)
            
            mask = probs
            if self.num_classes == 2:
                # For binary segmentation, use foreground probability
                mask = probs[:, 1:2, :, :]
            
            return mask, features, high_level_feat

    def get_coarse_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get coarse segmentation mask from CPUNet.
        
        Args:
            image: Input image (B, C, H, W)
            
        Returns:
            Soft probability mask (B, num_classes, H, W) or (B, 1, H, W) for binary
        """
        mask, _, _ = self.get_coarse_mask_and_features(image)
        return mask
    
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
        
        # Get coarse mask AND features from CPUNet
        coarse_mask, cpunet_features, high_level_feat = self.get_coarse_mask_and_features(image)
        
        # --- 1. Condition Augmentation ---
        # Randomly perturb coarse mask to simulate inference likelihood
        # CRITICAL: We must aggressively perturb the coarse mask during training.
        # Since CPUNet is trained on this data, its predictions are likely very good.
        # If we don't perturb, the diffusion model learns an identity mapping 
        # (just copying coarse_mask) and fails to refine boundaries at inference time.
        if self.training:
            # Increase perturbation probability to 90%
            if torch.rand(1).item() < 0.9:
                
                # Randomize kernel size for morphological ops (3, 5, or 7)
                # Larger kernels simulate larger boundary errors
                k_size = int(torch.randint(1, 4, (1,)).item()) * 2 + 1
                padding = k_size // 2
                
                # Randomly choose distrubtion type
                op_type = torch.rand(1).item()
                
                if op_type < 0.4:
                    # Dilation: Expand the mask (simulate over-segmentation)
                    coarse_mask = F.max_pool2d(coarse_mask, kernel_size=k_size, stride=1, padding=padding)
                elif op_type < 0.8:
                    # Erosion: Shrink the mask (simulate under-segmentation)
                    # Implemented as negating, max_pool (dilate background), then negating back
                    coarse_mask = -F.max_pool2d(-coarse_mask, kernel_size=k_size, stride=1, padding=padding)
                else:
                    # Random affine (shift/scale) could be added here, 
                    # but strong noise + morph is usually sufficient.
                    pass
                
                # Add stronger Gaussian noise to probabilities to simulate uncertainty
                # Range 0.05 to 0.2
                noise_scale = 0.05 + torch.rand(1).item() * 0.15
                noise_perturb = torch.randn_like(coarse_mask) * noise_scale
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
        
        # Predict noise
        # Note: Scale coarse mask to [-1, 1] for condition injection to match other inputs
        pred_noise = self.denoising_net(
            x_noisy, t, image, coarse_mask * 2 - 1,
            cpunet_features=cpunet_features,
            high_level_feat=high_level_feat
        )
        
        # --- 2. Loss Calculation ---
        
        # Standard MSE loss for noise prediction
        loss_noise = F.mse_loss(pred_noise, noise)
        
        # Boundary-aware Loss (Dice on predicted x0)
        # We calculate pred_x0 inside the loop
        pred_x0 = self.diffusion.predict_x0_from_noise(x_noisy, t, pred_noise)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Convert [-1, 1] back to [0, 1] implies probabilities
        pred_prob = (pred_x0 + 1) / 2
        
        # Target probabilities (same transformation)
        target_prob = (x_start + 1) / 2
        
        # Dice Loss calculation
        smooth = 1e-5
        intersection = (pred_prob * target_prob).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target_prob.sum(dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (union + smooth)
        loss_dice = 1.0 - dice_score.mean()
        
        # Weighted sum: 
        # loss_dice focuses on global structure and boundaries
        # loss_noise focuses on texture and local consistency
        loss = loss_noise + 1.0 * loss_dice
        
        metrics = {
            'loss': loss.item(),
            'loss_noise': loss_noise.item(),
            'loss_dice': loss_dice.item(),
            'dice_score': dice_score.mean().item(),
            'noise_mean': noise.mean().item(),
            'noise_std': noise.std().item(),
            'pred_noise_mean': pred_noise.mean().item(),
            'pred_noise_std': pred_noise.std().item()
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
        
        # Get coarse mask AND features
        coarse_mask, cpunet_features, high_level_feat = self.get_coarse_mask_and_features(image)
        
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
            coarse_mask_cond=coarse_mask_scaled, # Use scaled mask for condition
            num_inference_steps=num_inference_steps,
            eta=eta,
            start_from_coarse=True,
            start_timestep=start_timestep,
            x_start=coarse_mask_scaled,
            device=device,
            # Pass features used in conditioning
            cpunet_features=cpunet_features,
            high_level_feat=high_level_feat
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
