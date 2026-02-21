"""
Postprocessing utilities for segmentation masks.

This module provides functions for refining and cleaning up segmentation
masks produced by the neural network.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


def threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply threshold to convert probability map to binary mask.

    Args:
        mask: Probability mask (values between 0 and 1)
        threshold: Threshold value

    Returns:
        Binary mask
    """
    return (mask > threshold).astype(np.uint8)


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 100,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Remove small connected components from the mask.

    Args:
        mask: Binary mask
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for connected component analysis (4 or 8)

    Returns:
        Cleaned binary mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=connectivity,
    )

    # Create output mask
    output = np.zeros_like(mask)

    # Keep only components larger than min_size
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 1

    return output.astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in the binary mask.

    Args:
        mask: Binary mask

    Returns:
        Mask with holes filled
    """
    filled = ndimage.binary_fill_holes(mask)
    return filled.astype(np.uint8)


def apply_morphological_ops(
    mask: np.ndarray,
    operations: List[Tuple[str, int]] = None,
) -> np.ndarray:
    """
    Apply morphological operations to refine the mask.

    Args:
        mask: Binary mask
        operations: List of (operation_name, kernel_size) tuples
                   Operations: 'erosion', 'dilation', 'opening', 'closing'

    Returns:
        Refined mask
    """
    if operations is None:
        operations = [("closing", 5), ("opening", 3)]

    result = mask.copy().astype(np.uint8)

    for op, kernel_size in operations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op == "erosion":
            result = cv2.erode(result, kernel, iterations=1)
        elif op == "dilation":
            result = cv2.dilate(result, kernel, iterations=1)
        elif op == "opening":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "closing":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {op}")

    return result


def smooth_boundaries(
    mask: np.ndarray,
    sigma: float = 1.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Smooth mask boundaries using Gaussian blur.

    Args:
        mask: Binary mask
        sigma: Gaussian blur sigma
        threshold: Threshold to rebinarize after blurring

    Returns:
        Mask with smoothed boundaries
    """
    # Convert to float and blur
    mask_float = mask.astype(np.float32)
    # Ensure odd kernel size (required by OpenCV): bitwise OR with 1 makes odd
    kernel_size = int(sigma * 4) | 1
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), sigma)

    # Rebinarize
    return (blurred > threshold).astype(np.uint8)


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize mask to target size using nearest neighbor interpolation.

    Args:
        mask: Binary mask
        target_size: Target size as (height, width)

    Returns:
        Resized mask
    """
    return cv2.resize(
        mask.astype(np.uint8),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def compute_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract contours from binary mask.

    Args:
        mask: Binary mask

    Returns:
        List of contour arrays
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def smooth_contours(
    contours: List[np.ndarray],
    epsilon_factor: float = 0.001,
) -> List[np.ndarray]:
    """
    Smooth contours using Douglas-Peucker algorithm.

    Args:
        contours: List of contour arrays
        epsilon_factor: Approximation accuracy factor

    Returns:
        Smoothed contours
    """
    smoothed = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smoothed.append(approx)
    return smoothed


def contours_to_mask(
    contours: List[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert contours back to binary mask.

    Args:
        contours: List of contour arrays
        shape: Output mask shape (height, width)

    Returns:
        Binary mask
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    """
    Compute segmentation metrics.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        Dictionary of metrics (dice, iou, precision, recall)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    # Dice coefficient
    dice = 2 * intersection / (pred_sum + gt_sum + 1e-8)

    # IoU (Jaccard index)
    iou = intersection / (union + 1e-8)

    # Precision and Recall
    precision = intersection / (pred_sum + 1e-8)
    recall = intersection / (gt_sum + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def postprocess_mask(
    mask: np.ndarray,
    original_size: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
    min_object_size: int = 0,
    apply_morphology: bool = True,
    fill_holes_flag: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Complete postprocessing pipeline for segmentation mask.

    Args:
        mask: Raw probability mask from the model
        original_size: Original image size to resize back to
        threshold: Threshold for binarization
        min_object_size: Minimum size of objects to keep
        apply_morphology: Whether to apply morphological operations
        fill_holes_flag: Whether to fill holes
        smooth: Whether to smooth boundaries

    Returns:
        Postprocessed binary mask
    """
    # Remove batch and channel dimensions if present
    while mask.ndim > 2:
        mask = mask.squeeze()

    # Threshold
    binary_mask = threshold_mask(mask, threshold)

    # Apply morphological operations
    if apply_morphology:
        binary_mask = apply_morphological_ops(binary_mask)

    # Fill holes
    if fill_holes_flag:
        binary_mask = fill_holes(binary_mask)

    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_object_size)

    # Smooth boundaries
    if smooth:
        binary_mask = smooth_boundaries(binary_mask)

    # Resize to original size
    if original_size is not None:
        binary_mask = resize_mask(binary_mask, original_size)

    return binary_mask


if __name__ == "__main__":
    # Test postprocessing
    # Create a dummy mask with noise
    np.random.seed(42)
    dummy_mask = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(dummy_mask, (128, 128), 50, 1.0, -1)
    # Add noise
    dummy_mask += np.random.rand(256, 256) * 0.3
    dummy_mask = np.clip(dummy_mask, 0, 1)

    # Test postprocessing pipeline
    processed = postprocess_mask(dummy_mask, threshold=0.5)
    print(f"Input shape: {dummy_mask.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output unique values: {np.unique(processed)}")
