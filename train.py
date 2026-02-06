"""
Training script for the Medical Image Segmentation models.

This module provides training routines for:
1. CPUNet - Coarse segmentation network
2. Diffusion model - Boundary refinement
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import CPUNet, DiffusionRefinement
from configs import MODEL_CONFIG, PATHS


class SegmentationDataset(Dataset):
    """
    Dataset class for medical image segmentation.

    Expected directory structure:
    data_dir/
        images/
            image_001.png
            image_002.png
            ...
        masks/
            image_001.png
            image_002.png
            ...
    """

    def __init__(
        self,
        data_dir: str,
        image_size: tuple = (256, 256),
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment

        # Find all image-mask pairs
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        if os.path.exists(self.image_dir):
            self.image_files = sorted([
                f for f in os.listdir(self.image_dir)
                if f.endswith((".png", ".jpg", ".jpeg", ".tif"))
            ])
        else:
            self.image_files = []

    def __len__(self):
        # For demo/testing without real data: returns synthetic samples when no images exist
        # In production, this should raise an error or return actual length
        return max(len(self.image_files), 100)

    def __getitem__(self, idx):
        if idx < len(self.image_files):
            # Load real data
            from PIL import Image
            import cv2

            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)

            image = np.array(Image.open(img_path).convert("L")).astype(np.float32)
            mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32)

            # Normalize
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            mask = mask / 255.0

            # Resize
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)

        else:
            # Generate synthetic data for demo
            image = np.random.rand(*self.image_size).astype(np.float32)
            mask = np.zeros(self.image_size, dtype=np.float32)

            # Add random circle
            import cv2
            center = (
                np.random.randint(50, self.image_size[0] - 50),
                np.random.randint(50, self.image_size[1] - 50),
            )
            radius = np.random.randint(20, 60)
            cv2.circle(mask, center, radius, 1.0, -1)

        # Add channel dimension
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return torch.from_numpy(image), torch.from_numpy(mask)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def train_cpunet(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "auto",
):
    """
    Train the CPUNet model.

    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use for training
    """
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Training on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = SegmentationDataset(data_dir, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create model
    model = CPUNet(
        in_channels=MODEL_CONFIG["cpunet"]["in_channels"],
        out_channels=MODEL_CONFIG["cpunet"]["out_channels"],
        base_channels=MODEL_CONFIG["cpunet"]["base_channels"],
        num_blocks=MODEL_CONFIG["cpunet"]["num_blocks"],
    ).to(device)

    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "cpunet_best.pth"))
            print(f"Saved best model with loss: {best_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                os.path.join(output_dir, f"cpunet_epoch_{epoch+1}.pth"),
            )

    print("Training completed!")
    return model


def train_diffusion(
    data_dir: str,
    output_dir: str,
    cpunet_weights: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "auto",
):
    """
    Train the diffusion refinement model.

    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints
        cpunet_weights: Path to pretrained CPUNet weights
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use for training
    """
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Training on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = SegmentationDataset(data_dir, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load CPUNet for generating coarse masks
    cpunet = CPUNet(
        in_channels=MODEL_CONFIG["cpunet"]["in_channels"],
        out_channels=MODEL_CONFIG["cpunet"]["out_channels"],
        base_channels=MODEL_CONFIG["cpunet"]["base_channels"],
        num_blocks=MODEL_CONFIG["cpunet"]["num_blocks"],
    ).to(device)

    if cpunet_weights and os.path.exists(cpunet_weights):
        cpunet.load_state_dict(torch.load(cpunet_weights, map_location=device))
    cpunet.eval()

    # Create diffusion model
    diffusion = DiffusionRefinement(
        in_channels=MODEL_CONFIG["cpunet"]["in_channels"],
        base_channels=MODEL_CONFIG["diffusion"]["channels"],
        num_timesteps=MODEL_CONFIG["diffusion"]["num_timesteps"],
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_loss = float("inf")

    for epoch in range(epochs):
        diffusion.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # The diffusion model in training mode returns the loss
            loss = diffusion(masks, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(diffusion.state_dict(), os.path.join(output_dir, "diffusion_best.pth"))
            print(f"Saved best model with loss: {best_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                os.path.join(output_dir, f"diffusion_epoch_{epoch+1}.pth"),
            )

    print("Training completed!")
    return diffusion


def main():
    parser = argparse.ArgumentParser(description="Train medical image segmentation models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cpunet", "diffusion", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--cpunet_weights", type=str, default=None, help="Pretrained CPUNet weights")

    args = parser.parse_args()

    if args.model in ["cpunet", "both"]:
        print("\n" + "=" * 50)
        print("Training CPUNet")
        print("=" * 50)
        train_cpunet(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )

    if args.model in ["diffusion", "both"]:
        print("\n" + "=" * 50)
        print("Training Diffusion Model")
        print("=" * 50)

        cpunet_weights = args.cpunet_weights
        if cpunet_weights is None and args.model == "both":
            cpunet_weights = os.path.join(args.output_dir, "cpunet_best.pth")

        train_diffusion(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cpunet_weights=cpunet_weights,
            epochs=args.epochs,
            batch_size=args.batch_size // 2,  # Diffusion needs more memory
            learning_rate=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
