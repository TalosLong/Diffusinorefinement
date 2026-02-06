"""
CPUNet: Coarse Prediction U-Net for Medical Image Segmentation

This module implements a U-Net based architecture for initial coarse segmentation
of medical images. The network uses skip connections and multi-scale feature
extraction to produce accurate segmentation masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers, batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block with max pooling followed by convolutional block.
    """

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with transposed convolution and skip connection.
    """

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features during upsampling.
    """

    def __init__(self, gate_channels, in_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Handle size mismatch
        if g1.size() != x1.size():
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CPUNet(nn.Module):
    """
    Coarse Prediction U-Net (CPUNet) for medical image segmentation.

    This network takes medical images as input and produces initial coarse
    segmentation masks. The architecture follows the U-Net design with
    encoder-decoder structure and skip connections.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for binary segmentation)
        base_channels (int): Number of channels in the first layer (default: 64)
        num_blocks (int): Number of encoder/decoder blocks (default: 4)
        use_attention (bool): Whether to use attention gates (default: True)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        num_blocks=4,
        use_attention=True,
    ):
        super(CPUNet, self).__init__()

        self.num_blocks = num_blocks
        self.use_attention = use_attention

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = base_channels * (2**i)
            out_ch = base_channels * (2 ** (i + 1))
            self.encoders.append(DownBlock(in_ch, out_ch))

        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        for i in range(num_blocks - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2**i)
            self.decoders.append(UpBlock(in_ch, out_ch))
            if use_attention:
                # For decoder at level i:
                # - gate signal (x) has in_ch channels (before upsampling)
                # - skip connection has out_ch channels
                gate_ch = in_ch
                skip_ch = out_ch
                inter_ch = max(skip_ch // 2, 8)
                self.attention_gates.append(AttentionGate(gate_ch, skip_ch, inter_ch))

        # Final output convolution
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Segmentation mask of shape (B, out_channels, H, W)
        """
        # Store skip connections
        skips = []

        # Initial convolution
        x = self.init_conv(x)
        skips.append(x)

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Remove the last skip (bottleneck) as it's not used
        skips.pop()
        skips = skips[::-1]  # Reverse for decoder

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skips[i]
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            x = decoder(x, skip)

        # Final output
        x = self.final_conv(x)

        return torch.sigmoid(x)

    def get_features(self, x):
        """
        Extract multi-scale features from the encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            list: List of feature tensors at different scales
        """
        features = []

        x = self.init_conv(x)
        features.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)

        return features


class CPUNetLite(nn.Module):
    """
    Lightweight version of CPUNet for faster inference.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(CPUNetLite, self).__init__()

        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)

        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        b = self.bottleneck(e3)

        d3 = self.decoder3(b, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        out = self.final(d1)
        return torch.sigmoid(out)


if __name__ == "__main__":
    # Test the model
    model = CPUNet(in_channels=1, out_channels=1, base_channels=64, num_blocks=4)
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
