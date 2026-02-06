"""
Diffusion Refinement Model for Medical Image Segmentation

This module implements a denoising diffusion probabilistic model (DDPM) for
refining the coarse segmentation masks produced by CPUNet. The diffusion
process helps to smooth boundaries and reduce noise in the segmentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of timesteps
        embedding_dim: Dimension of the embedding

    Returns:
        Tensor of timestep embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    """

    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super(ResBlock, self).__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep embedding
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """

    def __init__(self, channels, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        # Attention
        scale = head_dim**-0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.view(b, c, h, w)
        out = self.proj(out)

        return x + out


class UNetDiffusion(nn.Module):
    """
    U-Net architecture for the diffusion model.

    This network predicts the noise added to the segmentation mask
    during the diffusion process.
    """

    def __init__(
        self,
        in_channels=2,  # Concatenate image and mask
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        time_channels=256,
    ):
        super(UNetDiffusion, self).__init__()

        self.time_channels = time_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            nn.SiLU(),
            nn.Linear(time_channels * 4, time_channels),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mult):
            out_channels_i = base_channels * mult

            for _ in range(num_res_blocks):
                block = ResBlock(now_channels, out_channels_i, time_channels, dropout)
                self.down_blocks.append(block)
                now_channels = out_channels_i
                channels.append(now_channels)

            if i != len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(now_channels, now_channels, 3, 2, 1))
                channels.append(now_channels)

        # Middle
        self.middle_block1 = ResBlock(now_channels, now_channels, time_channels, dropout)
        self.middle_attn = AttentionBlock(now_channels)
        self.middle_block2 = ResBlock(now_channels, now_channels, time_channels, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels_i = base_channels * mult

            for j in range(num_res_blocks + 1):
                in_ch = channels.pop() + now_channels
                block = ResBlock(in_ch, out_channels_i, time_channels, dropout)
                self.up_blocks.append(block)
                now_channels = out_channels_i

            if i != 0:
                self.up_samples.append(
                    nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1)
                )

        # Final output
        self.final_norm = nn.GroupNorm(8, now_channels)
        self.final_conv = nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Forward pass of the diffusion U-Net.

        Args:
            x: Input tensor (noisy mask concatenated with image)
            t: Timestep tensor

        Returns:
            Predicted noise
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_channels)
        t_emb = self.time_mlp(t_emb)

        # Initial conv
        h = self.init_conv(x)
        hs = [h]

        # Encoder
        block_idx = 0
        for i in range(len(self.down_samples) + 1):
            for _ in range(2):  # num_res_blocks
                h = self.down_blocks[block_idx](h, t_emb)
                hs.append(h)
                block_idx += 1

            if i < len(self.down_samples):
                h = self.down_samples[i](h)
                hs.append(h)

        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        # Decoder
        block_idx = 0
        for i in range(len(self.up_samples) + 1):
            for _ in range(3):  # num_res_blocks + 1
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, t_emb)
                block_idx += 1

            if i < len(self.up_samples):
                h = self.up_samples[i](h)

        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for mask refinement.

    This class handles the forward diffusion process (adding noise)
    and the reverse process (denoising) for refining segmentation masks.
    """

    def __init__(
        self,
        model,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        super(GaussianDiffusion, self).__init__()

        self.model = model
        self.num_timesteps = num_timesteps

        # Define beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos((x / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to the input at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_losses(self, x_0, condition, t, noise=None):
        """
        Calculate the loss for training.

        Args:
            x_0: Clean segmentation mask
            condition: Original image (used as conditioning)
            t: Timestep
            noise: Optional noise tensor

        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0, t, noise)
        x_input = torch.cat([x_noisy, condition], dim=1)

        predicted_noise = self.model(x_input, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, condition, t):
        """
        Reverse diffusion: denoise at timestep t.
        """
        x_input = torch.cat([x, condition], dim=1)
        predicted_noise = self.model(x_input, t)

        beta = self.betas[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Mean prediction
        model_mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha)

        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, condition, shape, num_inference_steps=50):
        """
        Generate a refined segmentation mask.

        Args:
            condition: Original image for conditioning
            shape: Shape of the output mask
            num_inference_steps: Number of denoising steps (for faster sampling)

        Returns:
            Refined segmentation mask with probability values in [0, 1].
            Apply thresholding (e.g., > 0.5) to obtain binary mask.
        """
        device = condition.device
        batch_size = condition.shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Calculate step size for faster sampling
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, condition, t_batch)

        return torch.sigmoid(x)


class DiffusionRefinement(nn.Module):
    """
    Complete diffusion refinement module that takes coarse segmentation
    and refines it using the diffusion process.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        num_timesteps=1000,
        num_inference_steps=50,
    ):
        super(DiffusionRefinement, self).__init__()

        self.num_inference_steps = num_inference_steps

        # Create the denoising network
        self.denoiser = UNetDiffusion(
            in_channels=in_channels + 1,  # mask + image
            out_channels=1,
            base_channels=base_channels,
        )

        # Create the diffusion process
        self.diffusion = GaussianDiffusion(
            model=self.denoiser,
            num_timesteps=num_timesteps,
        )

    def forward(self, coarse_mask, image):
        """
        Refine the coarse segmentation mask.

        Args:
            coarse_mask: Coarse segmentation from CPUNet
            image: Original input image

        Returns:
            Refined segmentation mask
        """
        # During training, we use the diffusion loss
        if self.training:
            batch_size = image.shape[0]
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=image.device)
            return self.diffusion.p_losses(coarse_mask, image, t)

        # During inference, we refine the mask
        else:
            return self.refine(coarse_mask, image)

    @torch.no_grad()
    def refine(self, coarse_mask, image):
        """
        Refine a coarse segmentation mask using the diffusion process.

        Instead of starting from pure noise, we start from the coarse mask
        with some added noise and denoise it.

        Returns:
            Refined segmentation mask with probability values in [0, 1].
            Apply thresholding (e.g., > 0.5) to obtain binary mask.
        """
        device = image.device
        batch_size = image.shape[0]

        # Add some noise to the coarse mask and denoise
        # This refines the boundaries while preserving the overall structure
        t_start = self.diffusion.num_timesteps // 4  # Start from a mild noise level

        # Add noise to the coarse mask
        t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(coarse_mask)
        noisy_mask = self.diffusion.q_sample(coarse_mask, t, noise)

        # Denoise step by step
        step_size = max(1, t_start // self.num_inference_steps)
        timesteps = list(range(0, t_start, step_size))[::-1]

        x = noisy_mask
        for t_val in timesteps:
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            x = self.diffusion.p_sample(x, image, t_batch)

        return torch.sigmoid(x)


if __name__ == "__main__":
    # Test the diffusion refinement model
    device = "cpu"

    # Create model
    model = DiffusionRefinement(in_channels=1, base_channels=32, num_timesteps=100)
    model = model.to(device)
    model.eval()

    # Test input
    image = torch.randn(1, 1, 128, 128, device=device)
    coarse_mask = torch.randn(1, 1, 128, 128, device=device).sigmoid()

    # Test inference
    with torch.no_grad():
        refined_mask = model(coarse_mask, image)

    print(f"Image shape: {image.shape}")
    print(f"Coarse mask shape: {coarse_mask.shape}")
    print(f"Refined mask shape: {refined_mask.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
