# Diffusion Refinement for Medical Image Segmentation

基于扩散优化的医学图像智能分割系统 / Medical Image Intelligent Segmentation System with Diffusion Refinement

## 🎯 Overview

This system combines deep learning segmentation models with diffusion models to achieve high-precision automatic segmentation of target regions in medical images. The diffusion denoising mechanism further optimizes the quality of segmentation boundaries.

本系统利用深度学习分割模型与扩散模型相结合，实现对医学图像中目标区域的高精度自动分割，并通过扩散去噪机制进一步优化分割边界质量。

## ✨ Features

- 🏥 **Medical Image Support**: CT, endoscopic images, X-ray, and more
- 📷 **Flexible Input**: Single image or batch processing
- 🔬 **Two-Stage Segmentation**: Coarse segmentation + Diffusion refinement
- 🎨 **Interactive Visualization**: Web-based interface with real-time results
- 📊 **Multiple Formats**: DICOM, NIfTI, PNG, JPEG, TIFF, and more

## 🏗️ System Architecture

```
Image Input → Preprocessing → CPUNet (Coarse Segmentation)
                                    ↓
          Result Visualization ← Postprocessing ← Diffusion Refinement
```

### Components

1. **Preprocessing Module** (`utils/preprocessing.py`)
   - Image loading (DICOM, NIfTI, common formats)
   - Intensity normalization
   - Contrast enhancement (CLAHE)
   - Denoising

2. **CPUNet** (`models/cpunet.py`)
   - U-Net based encoder-decoder architecture
   - Attention gates for feature focusing
   - Multi-scale feature extraction
   - Produces initial coarse segmentation

3. **Diffusion Refinement** (`models/diffusion.py`)
   - Denoising Diffusion Probabilistic Model (DDPM)
   - Conditioned on original image
   - Iterative boundary refinement
   - Noise-based uncertainty modeling

4. **Postprocessing Module** (`utils/postprocessing.py`)
   - Morphological operations (opening, closing)
   - Small object removal
   - Hole filling
   - Boundary smoothing

5. **Visualization Module** (`utils/visualization.py`)
   - Overlay creation
   - Contour drawing
   - Comparison visualization
   - Result export

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Diffusinorefinement.git
cd Diffusinorefinement

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Web Interface

Launch the Gradio-based web interface:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### Python API

```python
from pipeline import SegmentationPipeline

# Initialize the pipeline
pipeline = SegmentationPipeline(
    cpunet_weights="checkpoints/cpunet_best.pth",      # Optional
    diffusion_weights="checkpoints/diffusion_best.pth", # Optional
    device="auto",
    use_diffusion=True,
)

# Segment a single image
mask = pipeline.segment_image("path/to/medical_image.png")

# Get intermediate results
results = pipeline.segment_image(
    "path/to/image.png",
    return_intermediate=True
)
# results contains: input, coarse_mask, refined_mask, original_size

# Segment batch of images
masks = pipeline.segment_batch([
    "image1.png",
    "image2.png",
    "image3.png"
])

# Visualize results
pipeline.visualize("path/to/image.png", save_path="result.png")
```

### Command Line

```bash
# Segment a single image
python -c "
from pipeline import SegmentationPipeline
pipeline = SegmentationPipeline()
mask = pipeline.segment_image('input.png')
from PIL import Image
import numpy as np
Image.fromarray((mask * 255).astype(np.uint8)).save('output_mask.png')
"
```

## 🎓 Training

### Data Preparation

Organize your data in the following structure:

```
data/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### Train Models

```bash
# Train both CPUNet and Diffusion model
python train.py --model both --data_dir ./data --output_dir ./checkpoints --epochs 100

# Train only CPUNet
python train.py --model cpunet --data_dir ./data --epochs 100

# Train only Diffusion model (requires pretrained CPUNet)
python train.py --model diffusion --data_dir ./data --cpunet_weights ./checkpoints/cpunet_best.pth
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model to train (cpunet/diffusion/both) | both |
| `--data_dir` | Data directory | ./data |
| `--output_dir` | Output directory | ./checkpoints |
| `--epochs` | Number of epochs | 100 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Learning rate | 1e-4 |
| `--device` | Device (auto/cuda/cpu) | auto |

## 📁 Project Structure

```
Diffusinorefinement/
├── app.py                 # Gradio web interface
├── pipeline.py            # Main segmentation pipeline
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── configs/
│   ├── __init__.py
│   └── config.py          # Configuration settings
├── models/
│   ├── __init__.py
│   ├── cpunet.py          # CPUNet model
│   └── diffusion.py       # Diffusion refinement model
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py   # Image preprocessing
│   ├── postprocessing.py  # Mask postprocessing
│   └── visualization.py   # Result visualization
├── data/                  # Training data (not included)
└── checkpoints/           # Saved models (not included)
```

## 🔧 Configuration

Edit `configs/config.py` to customize:

```python
# Model Configuration
MODEL_CONFIG = {
    "cpunet": {
        "in_channels": 1,      # Grayscale input
        "out_channels": 1,     # Binary segmentation
        "base_channels": 64,   # Base feature channels
        "num_blocks": 4,       # Encoder/decoder depth
    },
    "diffusion": {
        "num_timesteps": 1000, # Diffusion timesteps
        "beta_start": 0.0001,  # Noise schedule start
        "beta_end": 0.02,      # Noise schedule end
        "channels": 64,        # Model channels
    }
}

# Image Configuration
IMAGE_CONFIG = {
    "input_size": (256, 256),  # Model input size
}
```

## 📊 Supported Image Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| DICOM | .dcm | Medical imaging standard |
| NIfTI | .nii, .nii.gz | Neuroimaging format |
| PNG | .png | Lossless compression |
| JPEG | .jpg, .jpeg | Lossy compression |
| TIFF | .tif, .tiff | High quality images |
| BMP | .bmp | Bitmap images |

## 🔬 Technical Details

### CPUNet Architecture

- Encoder: 4 downsampling blocks with 64→128→256→512 channels
- Decoder: 4 upsampling blocks with skip connections
- Attention gates for feature refinement
- Output: Sigmoid activated probability map

### Diffusion Model

- Based on DDPM (Denoising Diffusion Probabilistic Models)
- 1000 timesteps with linear beta schedule
- Conditioned on original image for context
- U-Net denoiser with timestep embedding

### Refinement Process

1. Add controlled noise to coarse mask
2. Iteratively denoise conditioned on image
3. Noise level determines refinement strength
4. Result: smoother, more accurate boundaries

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{diffusion_refinement_segmentation,
  title = {Diffusion Refinement for Medical Image Segmentation},
  year = {2024},
  url = {https://github.com/yourusername/Diffusinorefinement}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- U-Net architecture from Ronneberger et al.
- DDPM from Ho et al.
- Attention U-Net from Oktay et al.