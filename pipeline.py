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
import time

# Use the models copied from CPUNet
from models import (
    CPUNet_TransUnet, 
    create_diffusion_refiner_WS_encoded,
    CONFIGS_ViT_seg
)

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
        num_inference_steps: int = 20, # Default changed to 20 to match inference_diffusion.py
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
        
        # Configuration
        self.image_size = IMAGE_CONFIG.get("input_size", (256, 256))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
            
        self.num_classes = 2 # Hardcoded for now, as in most inference scripts
        self.vit_name = 'R50-ViT-B_16'

        # Initialize models
        self._init_models(cpunet_weights, diffusion_weights)

    def _init_models(
        self,
        cpunet_weights: Optional[str],
        diffusion_weights: Optional[str],
    ) -> None:
        """Initialize the neural network models."""
        
        # --- Initialize CPUNet (TransUnet) ---
        config_vit = CONFIGS_ViT_seg[self.vit_name]
        config_vit.n_classes = self.num_classes
        config_vit.n_skip = 3
        config_vit.dim = 3
        config_vit.heads = 8
        config_vit.dim_head = 64
        config_vit.dropout = 0
        config_vit.num_patches = 1024
        config_vit.mlp_dim = 1024
        
        if self.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(self.image_size[0] / 16), 
                int(self.image_size[1] / 16)
            )
            
        self.cpunet = CPUNet_TransUnet(
            config_vit, 
            img_size=self.image_size[0], 
            num_classes=self.num_classes
        ).to(self.device)

        # Load pretrained weights for CPUNet
        if cpunet_weights and os.path.exists(cpunet_weights):
            checkpoint = torch.load(cpunet_weights, map_location=self.device)
            # Handle possible wrapping
            if 'module.' in list(checkpoint.keys())[0]:
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                checkpoint = new_state_dict
                
            self.cpunet.load_state_dict(checkpoint, strict=False)
            print(f"Loaded CPUNet weights from {cpunet_weights}")
        elif cpunet_weights:
             print(f"Warning: CPUNet weights not found at {cpunet_weights}")

        self.cpunet.eval()

        # --- Initialize Diffusion Refinement ---
        if self.use_diffusion:
            # Parameters from inference_diffusion.py defaults
            base_channels = 64 
            num_timesteps = 1000
            
            # Using the factory from diffusion_refine_WS_encoded
            self.diffusion = create_diffusion_refiner_WS_encoded(
                cpunet=self.cpunet,
                num_classes=self.num_classes,
                num_timesteps=num_timesteps,
                base_channels=base_channels,
                freeze_cpunet=True
            ).to(self.device)

            # Load pretrained weights for Diffusion
            if diffusion_weights and os.path.exists(diffusion_weights):
                checkpoint = torch.load(diffusion_weights, map_location=self.device)
                
                # Logic from inference_diffusion.py load_diffusion_model
                state_dict = checkpoint
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.diffusion.load_state_dict(state_dict, strict=False)
                print(f"Loaded diffusion weights from {diffusion_weights}")
            elif diffusion_weights:
                print(f"Warning: Diffusion weights not found at {diffusion_weights}")
            
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
        """
        # Preprocess
        # Note: preprocess_image should return (1, C, H, W) or (C, H, W)
        # Disable enhance_contrast (CLAHE) as it distorts the intensity distribution 
        # expected by the model (trained on standard normalized data)
        processed, original_size = preprocess_image(
            image,
            target_size=self.image_size,
            return_original_size=True,
            enhance_contrast=False, 
            normalize=True,
        )

        # Convert to tensor
        # Ensure correct shape for CPUNet/Diffusion (B, 3, H, W) usually
        if isinstance(processed, np.ndarray):
            input_tensor = torch.from_numpy(processed).float()
        else:
            input_tensor = processed.float()
            
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Ensure 3 channels for CPUNet/Diffusion (R50-ViT expects 3 channels)
        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
            
        input_tensor = input_tensor.to(self.device)

        # Run inference
        refined_mask_np = None
        coarse_mask_np = None

        if self.use_diffusion and self.diffusion is not None:
             # inference() returns refined mask. Can also return coarse.
             # Note: model.inference takes raw image, and internally calls CPUNet to get coarse mask
             refined_tensor, coarse_tensor = self.diffusion.inference(
                 input_tensor,
                 num_inference_steps=self.num_inference_steps,
                 return_coarse=True
             )
             refined_mask_np = refined_tensor.cpu().numpy()
             coarse_mask_np = coarse_tensor.cpu().numpy()
        else:
             # Use CPUNet directly
             logits = self.cpunet(input_tensor)
             probs = torch.softmax(logits, dim=1)
             # return foreground prob
             coarse_mask_np = probs[:, 1:2, :, :].cpu().numpy()
             refined_mask_np = coarse_mask_np

        # Postprocess
        # refined_mask_np shape is (1, 1, H, W) or (1, C, H, W)
        # postprocess_mask usually expects (H, W) or (C, H, W)
        
        refined_for_pp = refined_mask_np.squeeze()
        coarse_for_pp = coarse_mask_np.squeeze()

        if postprocess:
            final_mask = postprocess_mask(
                refined_for_pp,
                original_size=original_size,
                min_object_size=50, # Reduced from default 100
                apply_morphology=False # setup default is too aggressive
            )
            # If requesting intermediate, we might want to postprocess coarse too
            final_coarse = postprocess_mask(
                coarse_for_pp,
                original_size=original_size,
                 min_object_size=50,
                 apply_morphology=False
            )
        else:
            final_mask = refined_for_pp
            final_coarse = coarse_for_pp

        # Convert to uint8 mask images for visualization/output
        final_mask_img = (final_mask > 0).astype(np.uint8) * 255
        final_coarse_img = (final_coarse > 0).astype(np.uint8) * 255

        if return_intermediate:
            return {
                "input": processed.squeeze(),
                "coarse_mask": final_coarse_img,
                "refined_mask": final_mask_img,
                "original_size": original_size,
            }

        return final_mask_img

    @torch.no_grad()
    def segment_batch(
        self,
        images: List[Union[str, np.ndarray]],
        postprocess: bool = True,
    ) -> List[np.ndarray]:
        """
        Segment a batch of medical images.
        """
        batch, original_sizes = preprocess_batch(images, target_size=self.image_size)
        input_tensor = torch.from_numpy(batch).float().to(self.device)
        
        # Inference
        if self.use_diffusion and self.diffusion is not None:
             refined_tensor = self.diffusion.inference(
                 input_tensor,
                 num_inference_steps=self.num_inference_steps
             )
             refined_masks_np = refined_tensor.cpu().numpy()
        else:
             logits = self.cpunet(input_tensor)
             probs = torch.softmax(logits, dim=1)
             refined_masks_np = probs[:, 1:2, :, :].cpu().numpy()

        results = []
        for i, mask in enumerate(refined_masks_np):
            if postprocess:
                processed_mask = postprocess_mask(mask.squeeze(), original_sizes[i])
            else:
                processed_mask = mask.squeeze()
            processed_mask = (processed_mask > 0).astype(np.uint8) * 255
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
        """
        results = self.segment_image(image, return_intermediate=True, postprocess=False)

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
        """
        if isinstance(image, str):
            from utils.preprocessing import load_image
            image_np = load_image(image)
        else:
            image_np = image

        mask = self.segment_image(image, postprocess=True)
        overlay = create_overlay(image_np, mask, alpha=alpha)

        return overlay


def create_demo_model():
    """
    Create a demonstration model with default weights.
    Results will be valid if weights are found, otherwise random.
    """
    # Default paths from workspace
    cpunet_weights = "/root/CPUNet/model/TU_Synapse256/+SGDpolyepochmval0.0001_0.002test_CPUNet_CVC_best.pth"
    diffusion_weights = "/root/autodl-tmp/model/CVC/diffusion_refiner_WS_withSACM_encoded_Synapse_256_bs8_lr0.0001_timesteps1000_20260126_160320/diffusion_refiner_final.pth"
    
    if not os.path.exists(cpunet_weights):
        cpunet_weights = None
        print("Warning: Default CPUNet weights not found.")
        
    if not os.path.exists(diffusion_weights):
        diffusion_weights = None
        print("Warning: Default Diffusion weights not found.")
        
    return SegmentationPipeline(
        cpunet_weights=cpunet_weights,
        diffusion_weights=diffusion_weights,
        device="auto",
        use_diffusion=True,
    )


if __name__ == "__main__":
    print("Testing Medical Image Segmentation Pipeline")
    pipeline = create_demo_model()
    dummy_image = np.random.rand(512, 512).astype(np.float32)
    # Mocking preprocess to return 3 channels as expected by R50-ViT
    
    # Actually segment_image calls preprocess_image, we should check if that works.
    # Assuming standard pipeline works.
    try:
        result = pipeline.segment_image(dummy_image, return_intermediate=True)
        print("Success!")
    except Exception as e:
        print(f"Error during test: {e}")
