"""
Inference script for Diffusion Refinement model.

This script performs inference using the trained diffusion refinement model:
1. Load the frozen CPUNet and trained diffusion model
2. Generate coarse segmentation from CPUNet
3. Refine segmentation using DDIM sampling

Usage:
    python inference_diffusion.py --cpunet_path <cpunet_path> --diffusion_path <diffusion_path> --input_path <input_path>
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.ndimage import zoom
import copy
from datasets.dataset_synapse import Synapse_dataset

# Import network components
# Note: The directory is 'network' (singular) in this repo
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.CPUNet import TransUnet_mlp as CPUNet
from networks.diffusion_refiner import DiffusionRefiner, create_diffusion_refiner
from utils import calculate_metric_percase


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Refinement Inference')
    
    # Model paths
    parser.add_argument('--cpunet_path', type=str, default='/root/CPUNet/model/TU_Synapse256/+SGDpolyepochmval0.0001_0.002test_CPUNet_CVC_best.pth',
                        help='path to pretrained CPUNet checkpoint')
    parser.add_argument('--diffusion_path', type=str, default='/root/autodl-tmp/model/diffusion_refiner_best.pth',
                        help='path to trained diffusion refinement model')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='vit model name for CPUNet')
    
    # Data paths
    parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/data/Synapse-CVC/test',
                        help='root dir for test data')
    parser.add_argument('--list_dir', type=str, default='/root/autodl-tmp/lists/lists_Synapse-CVC',
                        help='list dir')
    parser.add_argument('--dataset', type=str, default='Synapse',
                        help='dataset name')
    parser.add_argument('--output_dir', type=str, default='results/diffusion_refiner',
                        help='output directory for results')
    
    # Model configuration
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of segmentation classes')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='number of diffusion timesteps')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='base channels for diffusion U-Net')
    
    # Inference configuration
    parser.add_argument('--num_inference_steps', type=int, default=15,
                        help='number of DDIM sampling steps')
    parser.add_argument('--start_timestep', type=int, default=None,
                        help='starting timestep for refinement (default: 70% of num_timesteps)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0 for deterministic)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold for binary segmentation')
    
    # Other settings
    parser.add_argument('--save_images', action='store_true',
                        help='save prediction images')
    parser.add_argument('--compare_coarse', action='store_true',
                        help='compare with coarse segmentation')
    
    return parser.parse_args()


def load_cpunet(args, device):
    """Load pre-trained CPUNet model."""
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = 3
    config_vit.dim = 3
    config_vit.heads = 8
    config_vit.dim_head = 64
    config_vit.dropout = 0
    config_vit.num_patches = 1024
    config_vit.mlp_dim = 1024
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / 16), 
            int(args.img_size / 16)
        )
    
    # Create CPUNet model
    cpunet = CPUNet(
        config_vit, 
        img_size=args.img_size, 
        num_classes=args.num_classes
    ).to(device)
    
    # Load pre-trained weights
    if os.path.exists(args.cpunet_path):
        checkpoint = torch.load(args.cpunet_path, map_location=device)
        if 'module.' in list(checkpoint.keys())[0]:
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        cpunet.load_state_dict(checkpoint, strict=False)
        print(f"Loaded CPUNet from {args.cpunet_path}")
    else:
        raise FileNotFoundError(f"CPUNet checkpoint not found at {args.cpunet_path}")
    
    return cpunet


def load_diffusion_model(args, cpunet, device):
    """Load trained diffusion refinement model."""
    model = create_diffusion_refiner(
        cpunet=cpunet,
        num_classes=args.num_classes,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        freeze_cpunet=True
    ).to(device)
    
    if os.path.exists(args.diffusion_path):
        checkpoint = torch.load(args.diffusion_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel saved models
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded diffusion model from {args.diffusion_path}")
    else:
        raise FileNotFoundError(f"Diffusion model checkpoint not found at {args.diffusion_path}")
    
    return model


def inference_single_image(model, image, args, device):
    """
    Perform inference on a single image.
    
    Args:
        model: DiffusionRefiner model
        image: Input image tensor (1, C, H, W) or numpy array
        args: Arguments
        device: Device
        
    Returns:
        Tuple of (refined_mask, coarse_mask) as numpy arrays
    """
    model.eval()
    
    # Prepare input
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=0)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float().unsqueeze(0)
    
    image = image.to(device)
    
    # Resize if needed
    original_size = image.shape[2:]
    if image.shape[2] != args.img_size or image.shape[3] != args.img_size:
        image = F.interpolate(image, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
    
    # Inference
    with torch.no_grad():
        refined, coarse = model.inference(
            image,
            num_inference_steps=args.num_inference_steps,
            start_timestep=args.start_timestep,
            eta=args.eta,
            return_coarse=True
        )
    
    # Convert to numpy
    refined = refined.squeeze().cpu().numpy()
    coarse = coarse.squeeze().cpu().numpy()
    
    # Resize back to original size if needed
    if original_size[0] != args.img_size or original_size[1] != args.img_size:
        refined = zoom(refined, (original_size[0] / args.img_size, original_size[1] / args.img_size), order=1)
        coarse = zoom(coarse, (original_size[0] / args.img_size, original_size[1] / args.img_size), order=1)
    
    return refined, coarse


def test_single_volume_diffusion(image, label, model, args, device, test_save_path=None, case=None):
    """
    Test on a single volume with diffusion refinement.
    
    Args:
        image: Input image (1, C, H, W)
        label: Ground truth label (1, H, W)
        model: DiffusionRefiner model
        args: Arguments
        device: Device
        test_save_path: Path to save predictions
        case: Case name for saving
        
    Returns:
        Tuple of metrics for refined and coarse predictions
    """
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # Handle different image formats
    if image.ndim == 2:
        # Grayscale image [H, W] -> [3, H, W]
        image = np.stack([image] * 3, axis=0)
        _, x, y = image.shape
    elif image.ndim == 3:
        if image.shape[0] in [1, 3]:
            # Already in [C, H, W] format
            c, x, y = image.shape
            if c == 1:
                # Single channel -> 3 channels
                image = np.concatenate([image] * 3, axis=0)
        elif image.shape[2] in [1, 3]:
            # [H, W, C] format -> convert to [C, H, W]
            x, y, c = image.shape
            image = image.transpose(2, 0, 1)
            if c == 1:
                image = np.concatenate([image] * 3, axis=0)
        else:
            # Assume [C, H, W] format
            _, x, y = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Resize if needed
    if x != args.img_size or y != args.img_size:
        # image is now [3, H, W], resize spatial dimensions only
        image_resized = zoom(image, (1, args.img_size / x, args.img_size / y), order=3)
    else:
        image_resized = image
    
    # Convert to tensor [1, 3, H, W]
    input_tensor = torch.from_numpy(image_resized).unsqueeze(0).float().to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        refined_prob, coarse_prob = model.inference(
            input_tensor,
            num_inference_steps=args.num_inference_steps,
            start_timestep=args.start_timestep,
            eta=args.eta,
            return_coarse=True
        )
    
    # Convert to numpy
    refined_prob = refined_prob.squeeze().cpu().numpy()
    coarse_prob = coarse_prob.squeeze().cpu().numpy()
    
    # Apply threshold
    refined_pred = (refined_prob > args.threshold).astype(np.float32)
    coarse_pred = (coarse_prob > args.threshold).astype(np.float32)
    
    # Resize back
    if x != args.img_size or y != args.img_size:
        refined_pred = zoom(refined_pred, (x / args.img_size, y / args.img_size), order=0)
        coarse_pred = zoom(coarse_pred, (x / args.img_size, y / args.img_size), order=0)
    
    # Calculate metrics
    refined_metrics = []
    coarse_metrics = []
    
    for i in range(1, args.num_classes):
        refined_metrics.append(calculate_metric_percase(refined_pred == i, label == i))
        coarse_metrics.append(calculate_metric_percase(coarse_pred == i, label == i))
    
    # Save predictions
    if test_save_path is not None and case is not None:
        # Save refined prediction
        save_prediction(refined_pred, test_save_path, f'{case}_refined.png')
        # Save coarse prediction
        save_prediction(coarse_pred, test_save_path, f'{case}_coarse.png')
    
    return refined_metrics, coarse_metrics


def save_prediction(prediction, save_path, filename):
    """Save prediction as image."""
    pred_copy = copy.deepcopy(prediction)
    
    # Convert to RGB for visualization
    pred_copy[pred_copy == 1] = 255
    pred_copy[pred_copy == 2] = 128
    
    pred_img = Image.fromarray(np.uint8(pred_copy)).convert('L')
    pred_img.save(os.path.join(save_path, filename))


def inference_dataset(model, args, device):
    """Run inference on entire test dataset."""
    try:
        from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    except ImportError:
        print("Error: Could not import dataset_synapse.")
        print("Please ensure the datasets module is available.")
        print("Expected module path: datasets/dataset_synapse.py")
        return None, None
    
    from torch.utils.data import DataLoader
    
    # Load test dataset
    # test_dataset = Synapse_dataset(
    #     base_dir=args.root_path,
    #     list_dir=args.list_dir,
    #     split="test_vol",
    #     transform=transforms.Compose([
    #         RandomGenerator(output_size=[args.img_size, args.img_size])
    #     ])
    # )
    test_dataset = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Create output directory
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    refined_metrics_list = []
    coarse_metrics_list = []
    
    print(f"Running inference on {len(test_dataset)} samples...")
    
    for i_batch, sampled_batch in enumerate(test_loader):
        image = sampled_batch['image'].to(device)
        label = sampled_batch['label'].to(device)
        
        save_path = args.output_dir if args.save_images else None
        case_name = f'case_{i_batch:04d}'
        
        refined_metrics, coarse_metrics = test_single_volume_diffusion(
            image, label, model, args, device,
            test_save_path=save_path,
            case=case_name
        )
        
        refined_metrics_list.append(np.mean(refined_metrics, axis=0))
        coarse_metrics_list.append(np.mean(coarse_metrics, axis=0))
        
        if (i_batch + 1) % 10 == 0:
            print(f"Processed {i_batch + 1}/{len(test_dataset)} samples")
    
    # Calculate average metrics
    refined_avg = np.mean(refined_metrics_list, axis=0)
    coarse_avg = np.mean(coarse_metrics_list, axis=0)
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    metric_names = ['Dice', 'HD95', 'MCC', 'IoU', 'Acc', 'Sen', 'Spec']
    
    print("\nRefined (Diffusion) Results:")
    for name, value in zip(metric_names, refined_avg):
        print(f"  {name}: {value:.4f}")
    
    if args.compare_coarse:
        print("\nCoarse (CPUNet) Results:")
        for name, value in zip(metric_names, coarse_avg):
            print(f"  {name}: {value:.4f}")
        
        print("\nImprovement (Refined - Coarse):")
        for name, ref, coarse in zip(metric_names, refined_avg, coarse_avg):
            diff = ref - coarse
            sign = "+" if diff >= 0 else ""
            print(f"  {name}: {sign}{diff:.4f}")
    
    return refined_avg, coarse_avg


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset configuration (use command-line args as defaults)
    # Note: root_path and list_dir should be provided via command line
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '/root/autodl-tmp/data/Synapse-CVC/test',
            'list_dir': '/root/CPUNet/lists/lists_Synapse-CVC',
            'num_classes': 2,
        },
    }
    
    if args.dataset in dataset_config:
        dataset_name = args.dataset
        args.volume_path = dataset_config[dataset_name]['volume_path']
        args.num_classes = dataset_config[args.dataset]['num_classes']
        args.list_dir = dataset_config[dataset_name]['list_dir']
        args.Dataset = dataset_config[dataset_name]['Dataset']
    # Use the paths provided via command line arguments
    
    # Load models
    print("Loading CPUNet...")
    cpunet = load_cpunet(args, device)
    
    print("Loading Diffusion model...")
    model = load_diffusion_model(args, cpunet, device)
    
    # Run inference
    print(f"\nStarting inference with {args.num_inference_steps} DDIM steps...")
    if args.start_timestep:
        print(f"Starting from timestep {args.start_timestep}")
    else:
        print(f"Starting from default timestep (70% of {args.num_timesteps})")
    
    refined_metrics, coarse_metrics = inference_dataset(model, args, device)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'results.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("Diffusion Refinement Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"DDIM Steps: {args.num_inference_steps}\n")
        f.write(f"Start Timestep: {args.start_timestep or 'default (70%)'}\n")
        f.write(f"Eta: {args.eta}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        
        metric_names = ['Dice', 'HD95', 'MCC', 'IoU', 'Acc', 'Sen', 'Spec']
        
        f.write("Refined Results:\n")
        for name, value in zip(metric_names, refined_metrics):
            f.write(f"  {name}: {value:.4f}\n")
        
        f.write("\nCoarse Results:\n")
        for name, value in zip(metric_names, coarse_metrics):
            f.write(f"  {name}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
