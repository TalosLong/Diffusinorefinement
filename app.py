"""
Web Interface for Medical Image Segmentation System

This module provides a Gradio-based web interface for the medical image
segmentation system with diffusion refinement.

Features:
- Single image segmentation
- Batch image processing
- Interactive result visualization
- Support for various medical image formats
"""

import os
import tempfile
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from pipeline import SegmentationPipeline, create_demo_model
from utils import create_overlay, visualize_results
from utils.preprocessing import load_image, normalize_image
from utils.postprocessing import compute_metrics
from configs import SUPPORTED_FORMATS


# Global pipeline instance
pipeline: Optional[SegmentationPipeline] = None


def initialize_pipeline():
    """Initialize the segmentation pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = create_demo_model()
    return pipeline


def process_single_image(
    image: np.ndarray,
    use_diffusion: bool = True,
    overlay_alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Process a single image and return results.

    Args:
        image: Input image from Gradio
        use_diffusion: Whether to use diffusion refinement
        overlay_alpha: Transparency for overlay visualization

    Returns:
        Tuple of (coarse_overlay, refined_overlay, comparison, info_text)
    """
    global pipeline

    # Initialize pipeline if needed
    if pipeline is None:
        pipeline = initialize_pipeline()

    # Update diffusion setting
    pipeline.use_diffusion = use_diffusion

    try:
        # Convert to grayscale if needed
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            image_gray = np.mean(image, axis=2).astype(np.float32)
        else:
            image_gray = image.astype(np.float32)

        # Normalize
        image_gray = normalize_image(image_gray, method="minmax")

        # Get intermediate results
        results = pipeline.segment_image(image_gray, return_intermediate=True, postprocess=True)

        # Create overlays
        coarse_overlay = create_overlay(
            image_gray, results["coarse_mask"],
            color=(255, 165, 0), alpha=overlay_alpha
        )

        refined_overlay = create_overlay(
            image_gray, results["refined_mask"],
            color=(0, 255, 0), alpha=overlay_alpha
        )

        # Create comparison visualization
        from utils.visualization import draw_contour
        comparison = draw_contour(image_gray, results["coarse_mask"], color=(255, 165, 0), thickness=2)
        comparison = draw_contour(comparison, results["refined_mask"], color=(0, 255, 0), thickness=2)

        # Generate info text
        coarse_area = np.sum(results["coarse_mask"] > 0)
        refined_area = np.sum(results["refined_mask"] > 0)
        total_pixels = results["refined_mask"].size

        info_text = f"""
## Segmentation Results

### Statistics
- **Original Image Size:** {results['original_size'][0]} x {results['original_size'][1]}
- **Coarse Segmentation Area:** {coarse_area:,} pixels ({100*coarse_area/total_pixels:.2f}%)
- **Refined Segmentation Area:** {refined_area:,} pixels ({100*refined_area/total_pixels:.2f}%)
- **Diffusion Refinement:** {'Enabled' if use_diffusion else 'Disabled'}

### Legend
- 🟠 **Orange Contour:** Coarse segmentation (CPUNet)
- 🟢 **Green Contour:** Refined segmentation (Diffusion)
        """

        return coarse_overlay, refined_overlay, comparison, info_text

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return blank, blank, blank, f"## Error\n{error_msg}"


def process_batch_images(
    files: List[tempfile.NamedTemporaryFile],
    use_diffusion: bool = True,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """
    Process a batch of images.

    Args:
        files: List of uploaded files
        use_diffusion: Whether to use diffusion refinement

    Returns:
        Tuple of (gallery_images, summary_text)
    """
    global pipeline

    if pipeline is None:
        pipeline = initialize_pipeline()

    pipeline.use_diffusion = use_diffusion

    results = []
    summaries = []

    for i, file in enumerate(files):
        try:
            # Load image
            if hasattr(file, 'name'):
                image = np.array(Image.open(file.name).convert('L'))
            else:
                image = np.array(Image.open(file).convert('L'))

            image = normalize_image(image.astype(np.float32))

            # Segment
            mask = pipeline.segment_image(image, postprocess=True)

            # Create overlay
            overlay = create_overlay(image, mask, color=(0, 255, 0), alpha=0.4)

            # Add to results
            results.append((overlay, f"Image {i+1}"))

            # Calculate statistics
            area = np.sum(mask > 0)
            total = mask.size
            summaries.append(f"- Image {i+1}: {area:,} pixels ({100*area/total:.2f}%)")

        except Exception as e:
            summaries.append(f"- Image {i+1}: Error - {str(e)}")

    summary_text = f"""
## Batch Processing Results

Processed {len(files)} images with {'diffusion refinement' if use_diffusion else 'coarse segmentation only'}.

### Segmentation Areas:
{chr(10).join(summaries)}
    """

    return results, summary_text


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Medical Image Segmentation System",
    ) as demo:
        gr.Markdown(
            """
            # 🏥 Medical Image Intelligent Segmentation System

            A deep learning-based medical image segmentation system that combines **CPUNet** for coarse segmentation
            with **Diffusion Refinement** for high-quality boundary optimization.

            ## Features
            - ✅ Support for CT, endoscopic, and other medical images
            - ✅ Single image or batch processing
            - ✅ Diffusion-based boundary refinement
            - ✅ Interactive result visualization
            """
        )

        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("📷 Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Medical Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            use_diffusion = gr.Checkbox(
                                label="Use Diffusion Refinement",
                                value=True,
                            )
                            overlay_alpha = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.4,
                                step=0.1,
                                label="Overlay Transparency",
                            )
                        segment_btn = gr.Button("🔍 Segment Image", variant="primary")

                    with gr.Column(scale=2):
                        with gr.Row():
                            coarse_output = gr.Image(label="Coarse Segmentation (CPUNet)")
                            refined_output = gr.Image(label="Refined Segmentation (Diffusion)")

                        comparison_output = gr.Image(label="Comparison (Orange: Coarse, Green: Refined)")
                        info_output = gr.Markdown(label="Results")

                segment_btn.click(
                    fn=process_single_image,
                    inputs=[input_image, use_diffusion, overlay_alpha],
                    outputs=[coarse_output, refined_output, comparison_output, info_output],
                )

            # Batch Processing Tab
            with gr.TabItem("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input = gr.File(
                            label="Upload Multiple Images",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        batch_diffusion = gr.Checkbox(
                            label="Use Diffusion Refinement",
                            value=True,
                        )
                        batch_btn = gr.Button("🔄 Process Batch", variant="primary")

                    with gr.Column(scale=2):
                        batch_gallery = gr.Gallery(
                            label="Segmentation Results",
                            columns=3,
                            height="auto",
                        )
                        batch_info = gr.Markdown(label="Summary")

                batch_btn.click(
                    fn=process_batch_images,
                    inputs=[batch_input, batch_diffusion],
                    outputs=[batch_gallery, batch_info],
                )

            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                    ## System Architecture

                    ```
                    Image Input → Preprocessing → CPUNet (Coarse Segmentation)
                                                      ↓
                              Result Visualization ← Postprocessing ← Diffusion Refinement
                    ```

                    ### Components

                    #### 1. Preprocessing
                    - Image loading (supports DICOM, NIfTI, PNG, JPEG, etc.)
                    - Intensity normalization
                    - Contrast enhancement (CLAHE)
                    - Resizing to model input size

                    #### 2. CPUNet (Coarse Prediction U-Net)
                    - U-Net based encoder-decoder architecture
                    - Attention gates for feature focusing
                    - Produces initial coarse segmentation

                    #### 3. Diffusion Refinement
                    - Denoising Diffusion Probabilistic Model (DDPM)
                    - Refines boundaries through iterative denoising
                    - Conditioned on original image for accuracy

                    #### 4. Postprocessing
                    - Morphological operations
                    - Small object removal
                    - Boundary smoothing
                    - Hole filling

                    ### Supported Formats
                    - **Medical:** DICOM (.dcm), NIfTI (.nii, .nii.gz)
                    - **Standard:** PNG, JPEG, BMP, TIFF

                    ### References
                    - U-Net: Convolutional Networks for Biomedical Image Segmentation
                    - Denoising Diffusion Probabilistic Models
                    - Attention U-Net for Medical Image Segmentation
                    """
                )

        gr.Markdown(
            """
            ---
            *Medical Image Intelligent Segmentation System with Diffusion Refinement*
            """
        )

    return demo


def main():
    """Launch the web interface."""
    # Initialize pipeline
    initialize_pipeline()

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
