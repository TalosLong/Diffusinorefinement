"""
Preprocessing utilities for medical images.

This module provides functions for loading and preprocessing medical images
from various formats including DICOM, NIfTI, and common image formats.
"""

import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def load_image(path: str) -> np.ndarray:
    """
    Load a medical image from various formats.

    Args:
        path: Path to the image file

    Returns:
        numpy array of the image

    Raises:
        ValueError: If the file format is not supported
    """
    ext = os.path.splitext(path.lower())[1]

    if ext == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required for DICOM files. Install with: pip install pydicom")
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)
    elif ext in [".nii", ".gz"]:
        if sitk is None:
            raise ImportError("SimpleITK is required for NIfTI files. Install with: pip install SimpleITK")
        sitk_image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        # For 3D images, take the middle slice
        if image.ndim == 3:
            image = image[image.shape[0] // 2]
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        image = np.array(Image.open(path).convert("L")).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    return image


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Normalize image intensities.

    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore', 'percentile')
        clip_range: Optional tuple (min, max) to clip values before normalization

    Returns:
        Normalized image array
    """
    if clip_range is not None:
        image = np.clip(image, clip_range[0], clip_range[1])

    if method == "minmax":
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 1e-8:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 1e-8:
            image = (image - mean) / std
        else:
            image = image - mean
    elif method == "percentile":
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        if p99 - p1 > 1e-8:
            image = (image - p1) / (p99 - p1)
        else:
            image = np.zeros_like(image)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return image.astype(np.float32)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image array
        target_size: Target size as (height, width)
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')

    Returns:
        Resized image array
    """
    interp_methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }

    if interpolation not in interp_methods:
        raise ValueError(f"Unknown interpolation method: {interpolation}")

    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interp_methods[interpolation])
    return resized


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input image array (normalized to 0-1)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of the grid for local histogram equalization

    Returns:
        Enhanced image array
    """
    # Convert to uint8 for CLAHE
    image_uint8 = (image * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image_uint8)

    return enhanced.astype(np.float32) / 255.0


def denoise_image(
    image: np.ndarray,
    method: str = "bilateral",
    strength: float = 10.0,
) -> np.ndarray:
    """
    Apply denoising to the image.

    Args:
        image: Input image array (normalized to 0-1)
        method: Denoising method ('bilateral', 'gaussian', 'median')
        strength: Denoising strength

    Returns:
        Denoised image array
    """
    # Convert to uint8 for denoising
    image_uint8 = (image * 255).astype(np.uint8)

    if method == "bilateral":
        denoised = cv2.bilateralFilter(image_uint8, d=9, sigmaColor=strength * 5, sigmaSpace=strength * 5)
    elif method == "gaussian":
        # Ensure odd kernel size (required by OpenCV): bitwise OR with 1 makes odd
        kernel_size = int(strength) | 1
        denoised = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
    elif method == "median":
        # Ensure odd kernel size (required by OpenCV): bitwise OR with 1 makes odd
        kernel_size = int(strength) | 1
        denoised = cv2.medianBlur(image_uint8, kernel_size)
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoised.astype(np.float32) / 255.0


def preprocess_image(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    enhance_contrast: bool = True,
    denoise: bool = False,
    return_original_size: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Complete preprocessing pipeline for a single medical image.

    Args:
        image: Path to image file or numpy array
        target_size: Target size for resizing
        normalize: Whether to apply normalization
        enhance_contrast: Whether to apply CLAHE
        denoise: Whether to apply denoising
        return_original_size: Whether to return original image size

    Returns:
        Preprocessed image tensor ready for model input
        Optionally returns original size as well
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = load_image(image)

    original_size = image.shape[:2]

    # Normalize
    if normalize:
        image = normalize_image(image, method="percentile")

    # Denoise
    if denoise:
        image = denoise_image(image, method="bilateral")

    # Enhance contrast
    if enhance_contrast:
        image = apply_clahe(image)

    # Resize
    image = resize_image(image, target_size)

    # Add channel dimension if needed
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    # Add batch dimension
    image = image[np.newaxis, ...]

    if return_original_size:
        return image, original_size
    return image


def preprocess_batch(
    images: List[Union[str, np.ndarray]],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    enhance_contrast: bool = True,
    denoise: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Preprocess a batch of images.

    Args:
        images: List of image paths or numpy arrays
        target_size: Target size for resizing
        normalize: Whether to apply normalization
        enhance_contrast: Whether to apply CLAHE
        denoise: Whether to apply denoising

    Returns:
        Batch of preprocessed images and list of original sizes
    """
    processed_images = []
    original_sizes = []

    for img in images:
        processed, orig_size = preprocess_image(
            img,
            target_size=target_size,
            normalize=normalize,
            enhance_contrast=enhance_contrast,
            denoise=denoise,
            return_original_size=True,
        )
        processed_images.append(processed)
        original_sizes.append(orig_size)

    # Stack into batch
    batch = np.concatenate(processed_images, axis=0)

    return batch, original_sizes


if __name__ == "__main__":
    # Test preprocessing
    # Create a dummy image
    dummy_image = np.random.rand(512, 512).astype(np.float32) * 255

    # Test preprocessing pipeline
    processed = preprocess_image(dummy_image, target_size=(256, 256))
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
