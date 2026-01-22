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


class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


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
        self.use_dsconv_levels = {0, 1}
        
        # Initial convolution
        self.init_conv = nn.Conv2d(total_in_channels, base_channels, 3, padding=1)
        
        # ============== ENCODER ==============
        # We store each block's output channels and track skips
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.dsconvs = nn.ModuleDict()
        self.downsamples = nn.ModuleList()
        
        current_res = input_resolution
        in_ch = base_channels
        self.skip_channels = [base_channels]  # First skip is from init_conv
        

        for level_idx, mult in enumerate(channel_mult):
            if level_idx in self.use_dsconv_levels:
                ch = base_channels * mult
                self.dsconvs[str(level_idx)] = DSConv(
                    in_ch=ch,
                    out_ch=ch,
                    kernel_size=7,        # 强烈不建议 >9
                    extend_scope=1,
                    morph=0,              # 先用一种方向，稳
                    if_offset=True,
                    device="cuda"         # 或 torch.device
                )

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
        x = torch.cat([x, image_cond, coarse_mask_cond], dim=1)
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
        # ===== DSConv enhancement (only early levels) =====
                if str(level_idx) in self.dsconvs:
                    h_ds = self.dsconvs[str(level_idx)](h)
                    h = h + h_ds   # residual fusion（最稳）
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
