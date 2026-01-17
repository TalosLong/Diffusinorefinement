"""
Training script for Diffusion Refinement model.

This script trains the diffusion refinement model using:
1. A frozen CPUNet (or TransUNet-s2mlp) as the coarse segmentation frontend
2. A conditional diffusion model that learns to refine the coarse segmentation

Usage:
    python train_diffusion_refiner.py --cpunet_path <path_to_pretrained_cpunet> --root_path <data_path>
"""

import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Import network components
# Note: The directory is 'network' (singular) in this repo
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.CPUNet import TransUnet_mlp as CPUNet
from networks.diffusion_refiner import DiffusionRefiner, create_diffusion_refiner

def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Refinement Model')
    
    # Data paths
    parser.add_argument('--root_path', type=str, 
                        default='/root/autodl-tmp/data/Synapse-CVC', help='root dir for data')
    parser.add_argument('--list_dir', type=str,
                        default='/root/CPUNet/lists/lists_Synapse-CVC', help='list dir')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='dataset name')
    
    # CPUNet configuration
    parser.add_argument('--cpunet_path', type=str, required=True,
                        help='path to pretrained CPUNet checkpoint')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='vit model name for CPUNet')
    
    # Training configuration
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of segmentation classes')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum training epochs')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    
    # Diffusion configuration
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='number of diffusion timesteps')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='base channels for diffusion U-Net')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log metrics every N iterations')
    
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
        # Handle DataParallel saved models
        if 'module.' in list(checkpoint.keys())[0]:
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        cpunet.load_state_dict(checkpoint, strict=False)
        logging.info(f"Loaded CPUNet from {args.cpunet_path}")
    else:
        logging.warning(f"CPUNet checkpoint not found at {args.cpunet_path}, using random initialization")
    
    return cpunet


def train_diffusion_refiner(args, model, train_loader, val_loader, snapshot_path):
    """Training loop for diffusion refinement model."""
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "diffusion_train.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Setup optimizer (only train diffusion model, CPUNet is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, 
        lr=args.base_lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    max_iterations = args.max_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max_iterations,
        eta_min=1e-6
    )
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    
    # Training
    iter_num = 0
    best_val_loss = float('inf')
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable:,}")
    
    iterator = tqdm(range(args.max_epochs), ncols=70)
    
    for epoch_num in iterator:
        model.train()
        # Keep CPUNet in eval mode
        model.cpunet.eval()
        
        epoch_loss = 0.0
        epoch_samples = 0
        
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            
            # Training step
            loss, metrics = model.training_step(image_batch, label_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item() * image_batch.shape[0]
            epoch_samples += image_batch.shape[0]
            iter_num += 1
            
            # Logging
            if iter_num % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss.item(), iter_num)
                writer.add_scalar('train/lr', current_lr, iter_num)
                
                logging.info(
                    f'Epoch [{epoch_num+1}/{args.max_epochs}] '
                    f'Iter [{iter_num}] '
                    f'Loss: {loss.item():.6f} '
                    f'LR: {current_lr:.8f}'
                )
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / epoch_samples
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch_num)
        logging.info(f'Epoch [{epoch_num+1}] Average Loss: {avg_epoch_loss:.6f}')
        
        # Validation
        if val_loader is not None:
            val_loss = validate(model, val_loader)
            writer.add_scalar('val/loss', val_loss, epoch_num)
            logging.info(f'Epoch [{epoch_num+1}] Validation Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join("/root/autodl-tmp/model", 'diffusion_refiner_best.pth')
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, save_path)
                logging.info(f'Best model saved to {save_path}')
        
        # Periodic checkpoint
        if (epoch_num + 1) % args.save_interval == 0:
            save_path = os.path.join("/root/autodl-tmp/model", f'diffusion_refiner_epoch_{epoch_num+1}.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, save_path)
            logging.info(f'Checkpoint saved to {save_path}')
    
    # Save final model
    save_path = os.path.join("/root/autodl-tmp/model", 'diffusion_refiner_final.pth')
    torch.save({
        'epoch': args.max_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, save_path)
    logging.info(f'Final model saved to {save_path}')
    
    writer.close()
    return best_val_loss


def validate(model, val_loader):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            
            # Compute loss (in eval mode, we still use training_step for loss computation)
            loss, _ = model.training_step(image_batch, label_batch)
            
            total_loss += loss.item() * image_batch.shape[0]
            total_samples += image_batch.shape[0]
    
    return total_loss / total_samples


def main():
    args = parse_args()
    
    # Set up deterministic training
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset configuration (use command-line args as defaults, or override if needed)
    # Note: These paths are examples. Override with --root_path and --list_dir args
    dataset_config = {
        'Synapse': {
            'num_classes': 2,
        },
    }
    
    if args.dataset in dataset_config:
        args.num_classes = dataset_config[args.dataset]['num_classes']
    # root_path and list_dir should be provided via command line arguments
    
    # Create snapshot directory
    snapshot_path = f"model/diffusion_refiner_{args.dataset}_{args.img_size}"
    snapshot_path += f"_bs{args.batch_size}_lr{args.base_lr}"
    snapshot_path += f"_timesteps{args.num_timesteps}"
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # Load data - import with error handling
    try:
        from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    except ImportError:
        logging.error("Could not import dataset_synapse. Please ensure the datasets module is available.")
        logging.error("Expected module path: datasets/dataset_synapse.py")
        sys.exit(1)
    
    train_dataset = Synapse_dataset(
        base_dir=args.root_path + "/train",
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )
    
    # Split train/val
    total_size = len(train_dataset)
    val_size = total_size // 10
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logging.info(f"Train set size: {len(train_dataset)}")
    logging.info(f"Val set size: {len(val_dataset)}")
    
    # Load CPUNet
    cpunet = load_cpunet(args, device)
    
    # Create DiffusionRefiner model
    model = create_diffusion_refiner(
        cpunet=cpunet,
        num_classes=args.num_classes,
        num_timesteps=args.num_timesteps,
        base_channels=args.base_channels,
        freeze_cpunet=True
    ).to(device)
    
    # Multi-GPU support
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Train
    train_diffusion_refiner(args, model, train_loader, val_loader, snapshot_path)


if __name__ == '__main__':
    main()
