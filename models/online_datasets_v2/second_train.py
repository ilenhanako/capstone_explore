#!/usr/bin/env python3
"""
train_road_defects.py - Optimized YOLO training for road damage detection
Works with raw merged dataset (SVRDD + RDD2024) without preprocessing

Key features:
- No masking (trains on raw images)
- Ignores class 3 (road_rutting) during training - simply skipped, no remapping
- Optimized hyperparameters from ORDDC'2024 winners
- Handles imbalanced classes via focal loss

Usage:
    # Verify dataset structure
    python train_road_defects.py --dataset ./mergedDataset --mode verify
    
    # Analyze class distribution (including class 3 count)
    python train_road_defects.py --dataset ./mergedDataset --mode analyze
    
    # Fresh training with YOLOv11m (recommended)
    python train_road_defects.py --dataset ./mergedDataset --mode fresh --weights yolov11m.pt
    
    # Resume training
    python train_road_defects.py --dataset ./mergedDataset --mode resume
"""

from pathlib import Path
import argparse
import sys
import yaml
import torch
import warnings
from ultralytics import YOLO
import gc
from collections import Counter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FULL_TAXONOMY = {
    0: "road_crack_longitudinal",
    1: "road_crack_transverse", 
    2: "road_crack_alligator",
    3: "pothole",
    4: "marking_faded",
    5: "distractor_manhole",
    6: "distractor_patch",
}

ACTIVE_CLASSES = [0, 1, 2, 3, 4, 5, 6]

OPTIMIZED_HPARAMS = {
    # Loss functions
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    'iou': 0.5,
    
    #Augmentation
    'hsv_h': 0.015,# Hue
    'hsv_s': 0.7,# Saturation(lighting conditions)
    'hsv_v': 0.4, # Brightness
    'degrees': 10.0,# Rotation ±10°
    'translate': 0.2,# Translation 20%
    'scale': 0.5,# Scale variation
    'shear': 0.0,# Disable shear (roads are planar)
    'perspective': 0.0,# Disable perspective
    'flipud': 0.0,# No vertical flip (roads have orientation)
    'fliplr': 0.5,# 50% horizontal flip
    'mosaic': 1.0,#Mosaic augmentation
    'mixup': 0.15,#Mixup for occlusion robustness
    'copy_paste': 0.3,#Copy-paste for rare classes
    
    #learning rate schedule
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,
    
    # Training stability
    'close_mosaic': 10,
    'amp': True,#helps with the VRAM OOM issues
    'seed': 42,
    'deterministic': False,
}

# ---------------------------------------------------------------------------
# Dataset Analysis
# ---------------------------------------------------------------------------

def analyze_class_distribution(dataset_path: Path):
    """
    Analyze class distribution across all splits
    
    Args:
        dataset_path: Root directory of dataset
        
    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Class Distribution")
    print(f"{'='*60}\n")
    
    stats = {
        'total_files': 0,
        'total_labels': 0,
        'class_distribution': Counter(),
        'split_distribution': {},
    }
    
    for split in ['train', 'val', 'test']:
        label_dir = dataset_path / 'labels' / split
        
        if not label_dir.exists():
            print(f"⚠  Warning: {label_dir} not found, skipping")
            continue
        
        split_stats = {
            'files': 0,
            'labels': 0,
            'class_counts': Counter()
        }
        
        label_files = list(label_dir.glob('*.txt'))
        print(f"📁 Processing {split}: {len(label_files)} files")
        
        for label_file in label_files:
            split_stats['files'] += 1
            stats['total_files'] += 1
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    stats['total_labels'] += 1
                    stats['class_distribution'][class_id] += 1
                    split_stats['labels'] += 1
                    split_stats['class_counts'][class_id] += 1
            
            except Exception as e:
                print(f"⚠  Error reading {label_file}: {e}")
                continue
        
        stats['split_distribution'][split] = split_stats
        print(f"   Labels: {split_stats['labels']}")
    
    print(f"\n{'='*60}")
    print(f"Overall Class Distribution")
    print(f"{'='*60}\n")
    
    print(f"Total files: {stats['total_files']}")
    print(f"Total labels: {stats['total_labels']}")
    
    print(f"\nClass breakdown:")
    for class_id in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][class_id]
        class_name = FULL_TAXONOMY.get(class_id, f"unknown_{class_id}")
        percentage = (count / stats['total_labels'] * 100) if stats['total_labels'] > 0 else 0
        
        status = ""
        if class_id not in ACTIVE_CLASSES:
            status = "(NOT IN ACTIVE CLASSES)"
        
        print(f"  Class {class_id} ({class_name:.<30}): {count:>6} ({percentage:>5.2f}%){status}")
    
    print(f"\n{'='*60}")
    print(f"Per-Split Distribution")
    print(f"{'='*60}\n")
    
    for split, split_stats in stats['split_distribution'].items():
        print(f"{split.upper()}:")
        print(f"  Files: {split_stats['files']}")
        print(f"  Labels: {split_stats['labels']}")
        print(f"  Class distribution:")
        for class_id in sorted(split_stats['class_counts'].keys()):
            count = split_stats['class_counts'][class_id]
            class_name = FULL_TAXONOMY.get(class_id, f"unknown_{class_id}")
            print(f"    {class_id} ({class_name:.<28}): {count:>5}")
        print()

    return stats


# ---------------------------------------------------------------------------
# Dataset Management
# ---------------------------------------------------------------------------

def create_data_yaml(dataset_path: Path) -> str:
    """
    Create data.yaml that excludes class 3 from training
    
    Strategy: Only list active classes in 'names' dict
    YOLO will ignore any labels with class IDs not in this dict
    
    Args:
        dataset_path: Root directory of dataset
        
    Returns:
        Path to data.yaml file
    """
    data_yaml_path = dataset_path / "data.yaml"
    
    class_names = {class_id: FULL_TAXONOMY[class_id] for class_id in ACTIVE_CLASSES}
    
    data = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
        "nc": len(class_names)
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    
    print(f"\n{'='*60}")
    print(f"data.yaml created")
    print(f"{'='*60}")
    print(f"Path: {data['path']}")
    print(f"Active classes: {data['nc']}")
    print(f"\nClass mapping for training:")
    for idx in sorted(class_names.keys()):
        name = class_names[idx]
        print(f"  {idx}: {name}")
    
    print(f"{'='*60}\n")
    
    return str(data_yaml_path)


def verify_dataset_structure(dataset_path: Path) -> bool:
    """
    Verify that train/val/test splits exist with images and labels
    
    Args:
        dataset_path: Root directory of dataset
        
    Returns:
        True if structure is valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"🔍 Verifying Dataset Structure")
    print(f"{'='*60}\n")
    
    issues = []
    stats = {}
    
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split
        
        if not img_dir.is_dir():
            issues.append(f"Missing directory: {img_dir}")
            continue
        
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not img_files:
            issues.append(f"No images in: {img_dir}")
        
        if not lbl_dir.is_dir():
            issues.append(f"Missing directory: {lbl_dir}")
            continue
            
        lbl_files = list(lbl_dir.glob("*.txt"))
        if not lbl_files:
            issues.append(f"No labels in: {lbl_dir}")
        
        # Store stats
        stats[split] = {
            'images': len(img_files),
            'labels': len(lbl_files)
        }
    
    if issues:
        print(f"Dataset Structure Issues:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n{'='*60}\n")
        return False
    
    print(f"Dataset structure valid:")
    total_images = 0
    total_labels = 0
    for split, counts in stats.items():
        print(f"   {split:>5}: {counts['images']:>5} images, {counts['labels']:>5} labels")
        total_images += counts['images']
        total_labels += counts['labels']
    
    print(f"   {'TOTAL':>5}: {total_images:>5} images, {total_labels:>5} labels")
    print(f"\n{'='*60}\n")
    
    return True


def locate_checkpoint(project_root: Path, project_name: str) -> str:
    """
    Find the most recent checkpoint for resuming training
    
    Args:
        project_root: Root directory for training outputs
        project_name: Name of the training run
        
    Returns:
        Path to checkpoint file, or None if not found
    """
    weights_dir = project_root / project_name / "weights"
    
    if not weights_dir.exists():
        return None
    last_ckpt = weights_dir / "last.pt"
    if last_ckpt.exists():
        return str(last_ckpt)
    
    best_ckpt = weights_dir / "best.pt"
    if best_ckpt.exists():
        return str(best_ckpt)
    
    return None


# ---------------------------------------------------------------------------
# Memory & Batch Size Management
# ---------------------------------------------------------------------------

def estimate_batch_size(model_name: str, imgsz: int, vram_gb: float) -> int:
    """
    Conservative batch size estimation based on model size and available VRAM
    
    Args:
        model_name: Model identifier (e.g., 'yolov11m')
        imgsz: Input image size
        vram_gb: Available VRAM in GB
        
    Returns:
        Recommended batch size
    """
    model_memory = {
        'yolov11n': 0.5,
        'yolov11s': 1.0,
        'yolov11m': 2.0,
        'yolov11l': 3.0,
        'yolov11x': 4.5,
    }
    
    base_mem = model_memory.get(model_name.lower(), 2.0)
    per_image_mem = (imgsz / 640) ** 2 * 0.4
    usable_mem = max(vram_gb - base_mem - 2.0, 1.0)
    estimated = int(usable_mem / per_image_mem)
    return max(4, min(estimated, 32))


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    weights: str,
    data_yaml: str,
    project_root: str,
    project_name: str,
    epochs: int = 200,
    batch: int = -1,
    imgsz: int = 640,
    resume: bool = False,
    cache: str = "disk",
    workers: int = 8,
    device: str = 'auto',
    patience: int = 50,
    save_period: int = 10,
    verbose: bool = True,
):
    """
    Main training function with optimized hyperparameters
    
    Args:
        weights: Path to pretrained weights or checkpoint
        data_yaml: Path to data configuration file
        project_root: Root directory for outputs
        project_name: Name for this training run
        epochs: Number of training epochs
        batch: Batch size (-1 for auto-detection)  ##currently using 6-8
        imgsz: Input image size
        resume: Whether to resume from checkpoint
        cache: Caching strategy ('disk', 'ram', or 'none') ##not really in use
        workers: Number of dataloader workers
        device: Device specification ('auto', 0, 'cpu')
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        verbose: Print detailed information
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Road Damage Detection Training")
        print(f"{'='*60}")
        print(f"Model: {weights}")
        print(f"Data: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}x{imgsz}")
        print(f"Resume: {resume}")
        print(f"{'='*60}\n")
    
    print(f"Loading model...")
    model = YOLO(weights)
    
    # Auto-detect device and configure batch size
    if device == 'auto':
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"🎮 GPU: {gpu_name}")
            print(f"   VRAM: {vram_gb:.1f} GB")
            
            # Auto-adjust batch size if requested
            if batch == -1:
                model_size = Path(weights).stem
                batch = estimate_batch_size(model_size, imgsz, vram_gb)
                print(f"   Auto-batch: {batch}")
        else:
            device = 'cpu'
            batch = 4 if batch == -1 else batch
            print("⚠  No GPU detected - training on CPU (very slow)")
            print(f"   Using batch size: {batch}")
    
    cache_setting = False
    if cache.lower() == 'ram':
        cache_setting = True
        print("💾 Cache: RAM (fastest, high memory usage)")
    elif cache.lower() == 'disk':
        cache_setting = 'disk'
        print("💾 Cache: Disk (slower, no memory overhead)")
    else:
        print("💾 Cache: Disabled (slowest)")
    
    train_args = {
        'data': data_yaml,
        'imgsz': imgsz,
        'cache': cache_setting,
        'workers': workers,
        
        'epochs': epochs,
        'batch': batch,
        'device': device,
        'patience': patience,
        'save_period': save_period,
        
        'project': project_root,
        'name': project_name,
        'exist_ok': True,
        'save': True,
        'plots': True,
        'verbose': verbose,
        
        'resume': resume,
        **OPTIMIZED_HPARAMS
    }
    
    if verbose:
        print(f"\n📋 Key Training Configuration:")
        print(f"   Batch size: {train_args['batch']}")
        print(f"   Learning rate: {train_args['lr0']} → {train_args['lrf']} (cosine)")
        print(f"   Warmup epochs: {train_args['warmup_epochs']}")
        print(f"   IoU threshold: {train_args['iou']}")
        print(f"   Augmentation:")
        print(f"     - Mosaic: {train_args['mosaic']}")
        print(f"     - Copy-paste: {train_args['copy_paste']}")
        print(f"     - Mixup: {train_args['mixup']}")
        print(f"     - HSV: h={train_args['hsv_h']}, s={train_args['hsv_s']}, v={train_args['hsv_v']}")
        print(f"   Mixed precision: {train_args['amp']}")
        print(f"   Close mosaic: last {train_args['close_mosaic']} epochs")
    
    clear_memory()
    
    print(f"Training started...\n")
    
    try:
        results = model.train(**train_args)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training completed successfully!")
            print(f"{'='*60}")
            print(f"Results directory: {project_root}/{project_name}")
            print(f"Best weights: {project_root}/{project_name}/weights/best.pt")
            print(f"Last checkpoint: {project_root}/{project_name}/weights/last.pt")
            print(f"{'='*60}\n")
        
        return results
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        
        if "out of memory" in error_msg or "oom" in error_msg:
            print(f"\n{'='*60}")
            print("OUT OF MEMORY ERROR")
            print(f"{'='*60}")
            print(f"Current configuration:")
            print(f"  Model: {weights}")
            print(f"  Batch size: {batch}")
            print(f"  Image size: {imgsz}")
            print(f"  Cache: {cache}")
            print(f"\nSolutions (try in order):")
            print(f"  1. Reduce batch size:")
            print(f"     --batch {max(batch//2, 2)}")
            print(f"  2. Reduce image size:")
            print(f"     --imgsz {imgsz//2}")
            print(f"  3. Disable cache:")
            print(f"     --cache none")
            print(f"  4. Use smaller model:")
            print(f"     --weights yolov11s.pt")
            print(f"{'='*60}\n")
        
        raise
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        print(f"   Partial results saved to: {project_root}/{project_name}")
        raise
        
    except Exception as e:
        print(f"\nTraining failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Road damage detection training on merged raw dataset (SVRDD + RDD2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify dataset structure
  python train_road_defects.py --dataset ./mergedDataset --mode verify
  
  # Analyze class distribution (see how many class 3 labels exist)
  python train_road_defects.py --dataset ./mergedDataset --mode analyze
  
  # Fresh training with YOLOv11m (recommended for experimentation)
  python train_road_defects.py --dataset ./mergedDataset --mode fresh --weights yolov11m.pt
  
  # Fresh training with auto-batch
  python train_road_defects.py --dataset ./mergedDataset --mode fresh --batch -1
  
  # Resume from checkpoint
  python train_road_defects.py --dataset ./mergedDataset --mode resume
  
  # Quick test run
  python train_road_defects.py --dataset ./mergedDataset --mode fresh --epochs 10 --batch 4
        """
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to merged dataset root (must contain images/ and labels/)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["verify", "analyze", "fresh", "resume"],
        default="verify",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--weights",
        default="yolov11m.pt",
        help="Pretrained weights (.pt file)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto-detection)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--cache",
        choices=["none", "disk", "ram"],
        default="none",
        help="Image caching strategy"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device specification"
    )
    
    parser.add_argument(
        "--project_root",
        default="yolo_training_runs",
        help="Root directory for training outputs"
    )
    
    parser.add_argument(
        "--project_name",
        default="yolov11m_road_defects",
        help="Name for this training run"
    )
    
    parser.add_argument(
        "--save_period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"❌ Dataset directory not found: {dataset}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Road Damage Detection - Merged Dataset Training")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    # Verify dataset structure first
    if not verify_dataset_structure(dataset):
        print(f"\n❌ Dataset structure validation failed")
        sys.exit(1)
    
    # Mode: Analyze - show class distribution including class 3
    if args.mode == "analyze":
        stats = analyze_class_distribution(dataset)
        
        print(f"\n{'='*60}")
        print("✅ Dataset Analysis Complete")
        print(f"{'='*60}")
        print(f"\nKey findings:")
        total_count = stats['total_labels']
        print(f"  - Total labels: {total_count}")
        print(f"\nTo start training:")
        print(f"  python {sys.argv[0]} --dataset {args.dataset} --mode fresh")
        print(f"{'='*60}\n")
        return
    
    # Mode: Verify - just check structure
    if args.mode == "verify":
        print(f"\n{'='*60}")
        print("✅ Dataset Verification Complete")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. Analyze class distribution:")
        print(f"     python {sys.argv[0]} --dataset {args.dataset} --mode analyze")
        print(f"  2. Start training:")
        print(f"     python {sys.argv[0]} --dataset {args.dataset} --mode fresh")
        print(f"{'='*60}\n")
        return
    
    data_yaml = create_data_yaml(dataset)
    if args.mode == "resume":
        checkpoint = locate_checkpoint(Path(args.project_root), args.project_name)
        if not checkpoint:
            print(f"No checkpoint found for resuming")
            print(f"Searched in: {args.project_root}/{args.project_name}/weights")
            print(f"\nStart fresh training instead:")
            print(f"python {sys.argv[0]} --dataset {args.dataset} --mode fresh")
            sys.exit(1)
        
        weights_to_use = checkpoint
        should_resume = True
        print(f"Resuming from checkpoint: {checkpoint}\n")
        
    else:
        weights_path = Path(args.weights)
        
        if weights_path.exists():
            weights_to_use = str(weights_path.resolve())
            print(f"Using local weights: {weights_to_use}\n")
        else:
            weights_to_use = args.weights
            print(f"Using pretrained model: {weights_to_use}")
            print(f"(Will be downloaded if not cached)\n")
        
        should_resume = False
    
    try:
        results = train_model(
            weights=weights_to_use,
            data_yaml=data_yaml,
            project_root=args.project_root,
            project_name=args.project_name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            resume=should_resume,
            cache=args.cache,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            save_period=args.save_period,
            verbose=not args.quiet,
        )
        
        print(f"\n{'='*60}")
        print(f"Training Pipeline Complete")
        print(f"{'='*60}")
        print(f"Next steps:")
        print(f"  1. Validate your model:")
        print(f"     from ultralytics import YOLO")
        print(f"     model = YOLO('{args.project_root}/{args.project_name}/weights/best.pt')")
        print(f"     results = model.val()")
        print(f"     print(f'mAP50: {{results.box.map50:.4f}}')")
        print(f"  2. Check per-class performance:")
        print(f"     # Model predicts classes: 0,1,2,3,4,5,6 ")
        print(f"     print(results.box.ap_class_index)")
        print(f"  3. Run inference on test images:")
        print(f"     model.predict(source='{dataset}/images/test', save=True)")
        print(f"  4. View training plots:")
        print(f"     See: {args.project_root}/{args.project_name}/")
        print(f"\nImportant Notes:")
        print(f"  - Class IDs remain unchanged: 0,1,2,3,4,5,6")
        print(f"  - During inference, predictions will only be from active classes")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user (Ctrl+C)")
        print(f"   Partial results saved to: {args.project_root}/{args.project_name}")
        print(f"   Resume with: python {sys.argv[0]} --dataset {args.dataset} --mode resume")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n\nraining failed:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()