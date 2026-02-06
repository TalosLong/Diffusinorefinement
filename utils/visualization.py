"""
Visualization utilities for medical image segmentation.

This module provides functions for visualizing segmentation results,
including overlays, comparisons, and analysis plots.
"""

from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_colormap(num_classes: int = 2) -> np.ndarray:
    """
    Create a colormap for segmentation visualization.

    Args:
        num_classes: Number of classes including background

    Returns:
        Colormap array of shape (num_classes, 3)
    """
    colors = np.array(
        [
            [0, 0, 0],  # Background - black
            [255, 0, 0],  # Class 1 - red
            [0, 255, 0],  # Class 2 - green
            [0, 0, 255],  # Class 3 - blue
            [255, 255, 0],  # Class 4 - yellow
            [255, 0, 255],  # Class 5 - magenta
            [0, 255, 255],  # Class 6 - cyan
        ],
        dtype=np.uint8,
    )
    return colors[:num_classes]


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create an overlay of segmentation mask on the original image.

    Args:
        image: Original image (grayscale or RGB)
        mask: Binary segmentation mask
        color: Color for the mask overlay (R, G, B)
        alpha: Transparency of the overlay (0-1)

    Returns:
        RGB image with overlay
    """
    # Ensure image is 2D or 3D
    while image.ndim > 3:
        image = image.squeeze()
    while mask.ndim > 2:
        mask = mask.squeeze()

    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    # Normalize to 0-255 if needed
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)

    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)

    # Resize mask if needed
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored mask
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color

    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored_mask, alpha, 0)

    # Keep original where mask is 0
    overlay = np.where(mask[:, :, np.newaxis] > 0, overlay, image_rgb)

    return overlay


def draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw contour of segmentation mask on the image.

    Args:
        image: Original image (grayscale or RGB)
        mask: Binary segmentation mask
        color: Contour color (R, G, B)
        thickness: Contour thickness

    Returns:
        Image with contour drawn
    """
    # Ensure proper dimensions
    while image.ndim > 3:
        image = image.squeeze()
    while mask.ndim > 2:
        mask = mask.squeeze()

    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    # Normalize to 0-255 if needed
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)

    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)

    # Resize mask if needed
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    result = image_rgb.copy()
    cv2.drawContours(result, contours, -1, color, thickness)

    return result


def visualize_results(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    refined_mask: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """
    Create a visualization comparing coarse and refined segmentation.

    Args:
        image: Original input image
        coarse_mask: Coarse segmentation from CPUNet
        refined_mask: Refined segmentation from diffusion model
        ground_truth: Optional ground truth mask
        save_path: Optional path to save the visualization
        show: Whether to display the plot

    Returns:
        Visualization image as numpy array
    """
    # Ensure proper dimensions
    while image.ndim > 2:
        image = image.squeeze()
    while coarse_mask.ndim > 2:
        coarse_mask = coarse_mask.squeeze()
    while refined_mask.ndim > 2:
        refined_mask = refined_mask.squeeze()

    num_plots = 4 if ground_truth is None else 5

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Coarse segmentation overlay
    coarse_overlay = create_overlay(image, coarse_mask, color=(255, 165, 0), alpha=0.4)
    axes[1].imshow(coarse_overlay)
    axes[1].set_title("Coarse Segmentation")
    axes[1].axis("off")

    # Refined segmentation overlay
    refined_overlay = create_overlay(image, refined_mask, color=(0, 255, 0), alpha=0.4)
    axes[2].imshow(refined_overlay)
    axes[2].set_title("Refined Segmentation")
    axes[2].axis("off")

    # Comparison with contours
    comparison = draw_contour(image, coarse_mask, color=(255, 165, 0), thickness=1)
    comparison = draw_contour(comparison, refined_mask, color=(0, 255, 0), thickness=1)
    axes[3].imshow(comparison)
    axes[3].set_title("Comparison\n(Orange: Coarse, Green: Refined)")
    axes[3].axis("off")

    if ground_truth is not None:
        while ground_truth.ndim > 2:
            ground_truth = ground_truth.squeeze()
        gt_overlay = create_overlay(image, ground_truth, color=(0, 0, 255), alpha=0.4)
        axes[4].imshow(gt_overlay)
        axes[4].set_title("Ground Truth")
        axes[4].axis("off")

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    viz_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return viz_array


def create_comparison_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a grid visualization of multiple image-mask pairs.

    Args:
        images: List of input images
        masks: List of corresponding masks
        titles: Optional list of titles
        cols: Number of columns in the grid
        save_path: Optional path to save the visualization

    Returns:
        Grid visualization as numpy array
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for i in range(n):
        img = images[i]
        mask = masks[i]

        while img.ndim > 2:
            img = img.squeeze()
        while mask.ndim > 2:
            mask = mask.squeeze()

        overlay = create_overlay(img, mask, alpha=0.4)
        axes[i].imshow(overlay)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")

    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    viz_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    return viz_array


def visualize_diffusion_process(
    image: np.ndarray,
    masks_at_timesteps: List[np.ndarray],
    timesteps: List[int],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize the diffusion refinement process at different timesteps.

    Args:
        image: Original input image
        masks_at_timesteps: List of masks at different denoising steps
        timesteps: List of corresponding timestep values
        save_path: Optional path to save the visualization

    Returns:
        Visualization as numpy array
    """
    n = len(masks_at_timesteps)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))

    while image.ndim > 2:
        image = image.squeeze()

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Masks at different timesteps
    for i, (mask, t) in enumerate(zip(masks_at_timesteps, timesteps)):
        while mask.ndim > 2:
            mask = mask.squeeze()
        overlay = create_overlay(image, mask, alpha=0.4)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"t={t}")
        axes[i + 1].axis("off")

    plt.tight_layout()

    fig.canvas.draw()
    viz_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
    return viz_array


def save_result(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    overlay: bool = True,
) -> None:
    """
    Save segmentation result to file.

    Args:
        image: Original image
        mask: Segmentation mask
        output_path: Path to save the result
        overlay: Whether to save as overlay or just the mask
    """
    while image.ndim > 2:
        image = image.squeeze()
    while mask.ndim > 2:
        mask = mask.squeeze()

    if overlay:
        result = create_overlay(image, mask, alpha=0.4)
    else:
        result = (mask * 255).astype(np.uint8)

    Image.fromarray(result).save(output_path)


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)

    # Create dummy data
    image = np.random.rand(256, 256).astype(np.float32)
    coarse_mask = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(coarse_mask, (128, 128), 60, 1.0, -1)
    refined_mask = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(refined_mask, (128, 128), 50, 1.0, -1)

    # Test overlay
    overlay = create_overlay(image, coarse_mask)
    print(f"Overlay shape: {overlay.shape}")

    # Test visualization (without showing)
    viz = visualize_results(image, coarse_mask, refined_mask, show=False)
    print(f"Visualization shape: {viz.shape}")
