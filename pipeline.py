"""
Main Pipeline for Medical Image Segmentation with Diffusion Refinement

This module provides the complete segmentation pipeline that combines:
1. Image preprocessing
2. Coarse segmentation using CPUNet
3. Boundary refinement using Diffusion model
4. Postprocessing and visualization
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models import CPUNet, DiffusionRefinement
from utils import (
    preprocess_image,
    preprocess_batch,
    postprocess_mask,
    visualize_results,
    create_overlay,
)
from configs import MODEL_CONFIG, IMAGE_CONFIG, PROCESSING_CONFIG


class SegmentationPipeline:
    """
    Complete segmentation pipeline with diffusion refinement.

    This class orchestrates the entire segmentation process from
    raw medical images to final refined segmentation masks.
    """

    def __init__(
        self,
        cpunet_weights: Optional[str] = None,
        diffusion_weights: Optional[str] = None,
        device: str = "auto",
        use_diffusion: bool = True,
        num_inference_steps: int = 50,
    ):
        """
        Initialize the segmentation pipeline.

        Args:
            cpunet_weights: Path to pretrained CPUNet weights
            diffusion_weights: Path to pretrained diffusion model weights
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            use_diffusion: Whether to use diffusion refinement
            num_inference_steps: Number of diffusion denoising steps
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_diffusion = use_diffusion
        self.num_inference_steps = num_inference_steps

        # Initialize models
        self._init_models(cpunet_weights, diffusion_weights)

        # Store configuration
        self.image_size = IMAGE_CONFIG["input_size"]

    def _init_models(
        self,
        cpunet_weights: Optional[str],
        diffusion_weights: Optional[str],
    ) -> None:
        """Initialize the neural network models."""
        # Initialize CPUNet
        self.cpunet = CPUNet(
            in_channels=MODEL_CONFIG["cpunet"]["in_channels"],
            out_channels=MODEL_CONFIG["cpunet"]["out_channels"],
            base_channels=MODEL_CONFIG["cpunet"]["base_channels"],
            num_blocks=MODEL_CONFIG["cpunet"]["num_blocks"],
        )

        # Load pretrained weights if provided
        if cpunet_weights and os.path.exists(cpunet_weights):
            state_dict = torch.load(cpunet_weights, map_location=self.device)
            self.cpunet.load_state_dict(state_dict)
            print(f"Loaded CPUNet weights from {cpunet_weights}")

        self.cpunet = self.cpunet.to(self.device)
        self.cpunet.eval()

        # Initialize Diffusion Refinement
        if self.use_diffusion:
            self.diffusion = DiffusionRefinement(
                in_channels=MODEL_CONFIG["cpunet"]["in_channels"],
                base_channels=MODEL_CONFIG["diffusion"]["channels"],
                num_timesteps=MODEL_CONFIG["diffusion"]["num_timesteps"],
                num_inference_steps=self.num_inference_steps,
            )

            # Load pretrained weights if provided
            if diffusion_weights and os.path.exists(diffusion_weights):
                state_dict = torch.load(diffusion_weights, map_location=self.device)
                self.diffusion.load_state_dict(state_dict)
                print(f"Loaded diffusion weights from {diffusion_weights}")

            self.diffusion = self.diffusion.to(self.device)
            self.diffusion.eval()
        else:
            self.diffusion = None

    @torch.no_grad()
    def segment_image(
        self,
        image: Union[str, np.ndarray],
        return_intermediate: bool = False,
        postprocess: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Segment a single medical image.

        Args:
            image: Path to image file or numpy array
            return_intermediate: Whether to return intermediate results
            postprocess: Whether to apply postprocessing

        Returns:
            Segmentation mask or dict with intermediate results
        """
        # Preprocess
        processed, original_size = preprocess_image(
            image,
            target_size=self.image_size,
            return_original_size=True,
        )

        # Convert to tensor
        input_tensor = torch.from_numpy(processed).float().to(self.device)

        # Coarse segmentation
        coarse_mask = self.cpunet(input_tensor)
        coarse_mask_np = coarse_mask.cpu().numpy()

        # Diffusion refinement
        if self.use_diffusion and self.diffusion is not None:
            refined_mask = self.diffusion.refine(coarse_mask, input_tensor)
            refined_mask_np = refined_mask.cpu().numpy()
        else:
            refined_mask_np = coarse_mask_np

        # Postprocess
        if postprocess:
            final_mask = postprocess_mask(
                refined_mask_np,
                original_size=original_size,
            )
        else:
            final_mask = refined_mask_np.squeeze()

        if return_intermediate:
            return {
                "input": processed.squeeze(),
                "coarse_mask": postprocess_mask(coarse_mask_np, original_size) if postprocess else coarse_mask_np.squeeze(),
                "refined_mask": final_mask,
                "original_size": original_size,
            }

        return final_mask

    @torch.no_grad()
    def segment_batch(
        self,
        images: List[Union[str, np.ndarray]],
        postprocess: bool = True,
    ) -> List[np.ndarray]:
        """
        Segment a batch of medical images.

        Args:
            images: List of image paths or numpy arrays
            postprocess: Whether to apply postprocessing

        Returns:
            List of segmentation masks
        """
        # Preprocess batch
        batch, original_sizes = preprocess_batch(images, target_size=self.image_size)

        # Convert to tensor
        input_tensor = torch.from_numpy(batch).float().to(self.device)

        # Coarse segmentation
        coarse_masks = self.cpunet(input_tensor)

        # Diffusion refinement
        if self.use_diffusion and self.diffusion is not None:
            refined_masks = self.diffusion.refine(coarse_masks, input_tensor)
        else:
            refined_masks = coarse_masks

        refined_masks_np = refined_masks.cpu().numpy()

        # Postprocess each mask
        results = []
        for i, mask in enumerate(refined_masks_np):
            if postprocess:
                processed_mask = postprocess_mask(mask, original_sizes[i])
            else:
                processed_mask = mask.squeeze()
            results.append(processed_mask)

        return results

    def visualize(
        self,
        image: Union[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Segment and visualize results for a single image.

        Args:
            image: Path to image file or numpy array
            save_path: Optional path to save visualization
            show: Whether to display the visualization

        Returns:
            Visualization image as numpy array
        """
        # Get intermediate results
        results = self.segment_image(image, return_intermediate=True, postprocess=False)

        # Create visualization
        viz = visualize_results(
            results["input"],
            results["coarse_mask"],
            results["refined_mask"],
            save_path=save_path,
            show=show,
        )

        return viz

    def get_overlay(
        self,
        image: Union[str, np.ndarray],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Get segmentation overlay on the original image.

        Args:
            image: Path to image file or numpy array
            alpha: Overlay transparency

        Returns:
            RGB overlay image
        """
        # Load image if path
        if isinstance(image, str):
            from utils.preprocessing import load_image
            image_np = load_image(image)
        else:
            image_np = image

        # Get segmentation
        mask = self.segment_image(image, postprocess=True)

        # Create overlay
        overlay = create_overlay(image_np, mask, alpha=alpha)

        return overlay


def create_demo_model():
    """
    Create a demonstration model without pretrained weights.

    Returns:
        SegmentationPipeline instance
    """
    return SegmentationPipeline(
        cpunet_weights=None,
        diffusion_weights=None,
        device="auto",
        use_diffusion=True,
    )


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Medical Image Segmentation Pipeline")
    print("=" * 50)

    # Create pipeline
    pipeline = create_demo_model()
    print(f"Device: {pipeline.device}")
    print(f"Using diffusion: {pipeline.use_diffusion}")

    # Create dummy input
    dummy_image = np.random.rand(512, 512).astype(np.float32)

    # Test segmentation
    result = pipeline.segment_image(dummy_image, return_intermediate=True)

    print(f"\nInput shape: {result['input'].shape}")
    print(f"Coarse mask shape: {result['coarse_mask'].shape}")
    print(f"Refined mask shape: {result['refined_mask'].shape}")
    print(f"Original size: {result['original_size']}")

    print("\nPipeline test completed successfully!")
