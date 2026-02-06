"""
Configuration settings for the Medical Image Segmentation System
"""

import os

# Model Configuration
MODEL_CONFIG = {
    "cpunet": {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 64,
        "num_blocks": 4,
    },
    "diffusion": {
        "num_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "channels": 64,
        "num_res_blocks": 2,
    }
}

# Image Configuration
IMAGE_CONFIG = {
    "input_size": (256, 256),
    "normalize_mean": 0.5,
    "normalize_std": 0.5,
}

# Processing Configuration
PROCESSING_CONFIG = {
    "batch_size": 1,
    "device": "cuda",  # or "cpu"
    "num_workers": 4,
}

# Paths Configuration
PATHS = {
    "models_dir": os.path.join(os.path.dirname(__file__), "..", "checkpoints"),
    "data_dir": os.path.join(os.path.dirname(__file__), "..", "data"),
    "output_dir": os.path.join(os.path.dirname(__file__), "..", "outputs"),
}

# Supported Image Formats
SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz"]
