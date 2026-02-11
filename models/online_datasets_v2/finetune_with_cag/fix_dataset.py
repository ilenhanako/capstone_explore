#!/usr/bin/env python3
"""
Split YOLO dataset into train/val/test with stratification.
KEEPS negative samples (images without annotations).

Changes:
- Does NOT skip empty label files
- Treats empty labels as "negative" class
- Ensures negatives are distributed across splits
"""

import argparse
import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Path to original dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Training set ratio (default: 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--stratify", action="store_true", help="Stratify split by class")
    parser.add_argument("--copy", action="store_true", default=True, help="Copy files (default)")
    parser.add_argument("--no-copy", dest="copy", action="store_false", help="Create symlinks instead")
    return parser.parse_args()


def parse_label_file(label_path):
    """
    Parse YOLO label file and return list of class IDs.
    Returns empty list for empty files (negative samples).
    """
    classes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    classes.append(class_id)
    except Exception as e:
        print(f"Warning: Could not parse {label_path}: {e}")
    
    return classes


def collect_dataset_info(images_dir, labels_dir):
    """
    Collect information about all images and their classes.
    INCLUDES images with empty labels (negatives).
    """
    dataset_info = {}
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    negatives_count = 0
    
    for img_path in image_files:
        stem = img_path.stem
        
        # Find corresponding label file
        label_path = Path(labels_dir) / f"{stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file for {img_path.name}, skipping...")
            continue
        
        # Parse classes in this image
        classes = parse_label_file(label_path)
        
        # KEEP IMAGES WITH EMPTY LABELS (negatives)
        if not classes:
            # Treat as negative sample (class = -1)
            dataset_info[stem] = {
                'image_path': img_path,
                'label_path': label_path,
                'classes': [],
                'primary_class': -1,  # Special marker for negatives
                'is_negative': True
            }
            negatives_count += 1
        else:
            # Normal sample with annotations
            dataset_info[stem] = {
                'image_path': img_path,
                'label_path': label_path,
                'classes': classes,
                'primary_class': classes[0],  # Use first class for stratification
                'is_negative': False
            }
    
    print(f"Valid samples (with labels): {len(dataset_info)}")
    print(f"  Positive samples (with defects): {len(dataset_info) - negatives_count}")
    print(f"  Negative samples (no defects): {negatives_count}")
    
    if negatives_count > 0:
        print(f"\n✅ Keeping {negatives_count} negative samples for training")
        print(f"   Negatives are CRITICAL for teaching model what is NOT a defect!")
    
    return dataset_info


def stratified_split(dataset_info, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Perform stratified split including negative samples.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group images by their primary class (including negatives as class -1)
    class_to_images = defaultdict(list)
    for stem, info in dataset_info.items():
        primary_class = info['primary_class']
        class_to_images[primary_class].append(stem)
    
    # Print class distribution
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION IN DATASET")
    print("="*70)
    
    # Sort classes, with negatives (-1) last
    sorted_classes = sorted([c for c in class_to_images.keys() if c != -1])
    if -1 in class_to_images:
        sorted_classes.append(-1)
    
    for class_id in sorted_classes:
        count = len(class_to_images[class_id])
        if class_id == -1:
            print(f"Negatives (no defects): {count:4d} images")
        else:
            print(f"Class {class_id:2d}: {count:4d} images")
    print("="*70 + "\n")
    
    train_set = []
    val_set = []
    test_set = []
    
    # Split each class separately (including negatives)
    for class_id, image_stems in class_to_images.items():
        n_samples = len(image_stems)
        
        # Shuffle
        random.shuffle(image_stems)
        
        class_name = "Negatives (no defects)" if class_id == -1 else f"Class {class_id}"
        
        # Handle rare classes (< 10 samples)
        if n_samples < 10:
            print(f"⚠️  {class_name} has only {n_samples} samples")
            
            if n_samples == 1:
                train_set.extend(image_stems)
                print(f"   → train: 1, val: 0, test: 0")
            elif n_samples == 2:
                train_set.append(image_stems[0])
                val_set.append(image_stems[1])
                print(f"   → train: 1, val: 1, test: 0")
            elif n_samples <= 5:
                train_set.append(image_stems[0])
                val_set.append(image_stems[1])
                if len(image_stems) > 2:
                    test_set.extend(image_stems[2:])
                print(f"   → train: 1, val: 1, test: {len(image_stems)-2}")
            else:
                # 6-9 samples: at least 1 in each split
                train_set.append(image_stems[0])
                val_set.append(image_stems[1])
                test_set.append(image_stems[2])
                
                remaining = image_stems[3:]
                n_train_add = int(len(remaining) * train_ratio)
                n_val_add = int(len(remaining) * val_ratio)
                
                train_set.extend(remaining[:n_train_add])
                val_set.extend(remaining[n_train_add:n_train_add + n_val_add])
                test_set.extend(remaining[n_train_add + n_val_add:])
                
                print(f"   → train: {1+n_train_add}, val: {1+n_val_add}, test: {1+len(remaining)-n_train_add-n_val_add}")
        
        else:
            # Normal stratified split for classes with >= 10 samples
            n_train = max(1, int(n_samples * train_ratio))
            n_val = max(1, int(n_samples * val_ratio))
            n_test = max(1, n_samples - n_train - n_val)
            
            train_set.extend(image_stems[:n_train])
            val_set.extend(image_stems[n_train:n_train + n_val])
            test_set.extend(image_stems[n_train + n_val:])
            
            print(f"{class_name:<25} train: {n_train:3d}, val: {n_val:3d}, test: {n_test:3d}")
    
    return train_set, val_set, test_set


def copy_or_link_files(file_list, dataset_info, output_dir, split_name, copy=True):
    """Copy or symlink images and labels to output directory"""
    images_out = Path(output_dir) / "images" / split_name
    labels_out = Path(output_dir) / "labels" / split_name
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    for stem in file_list:
        info = dataset_info[stem]
        
        src_image = info['image_path']
        src_label = info['label_path']
        
        dst_image = images_out / src_image.name
        dst_label = labels_out / src_label.name
        
        if copy:
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_label, dst_label)
        else:
            os.symlink(src_image.absolute(), dst_image)
            os.symlink(src_label.absolute(), dst_label)
    
    print(f"✅ {split_name:5s}: {len(file_list):4d} images → {images_out}")


def verify_split(dataset_info, train_set, val_set, test_set):
    """Verify split and print statistics"""
    def get_class_distribution(image_stems):
        class_counts = defaultdict(int)
        negative_count = 0
        
        for stem in image_stems:
            info = dataset_info[stem]
            if info['is_negative']:
                negative_count += 1
            else:
                for class_id in info['classes']:
                    class_counts[class_id] += 1
        
        return class_counts, negative_count
    
    print("\n" + "="*70)
    print("SPLIT VERIFICATION")
    print("="*70)
    
    train_dist, train_neg = get_class_distribution(train_set)
    val_dist, val_neg = get_class_distribution(val_set)
    test_dist, test_neg = get_class_distribution(test_set)
    
    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))
    
    print(f"\n{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 70)
    
    # Print per-class distribution
    for class_id in all_classes:
        train_count = train_dist.get(class_id, 0)
        val_count = val_dist.get(class_id, 0)
        test_count = test_dist.get(class_id, 0)
        total = train_count + val_count + test_count
        
        print(f"Class {class_id:<13} {train_count:>8} {val_count:>8} {test_count:>8} {total:>8}")
    
    # Print negative samples
    print("-" * 70)
    print(f"{'NEGATIVES (clean)':<20} {train_neg:>8} {val_neg:>8} {test_neg:>8} {train_neg+val_neg+test_neg:>8}")
    print("-" * 70)
    print(f"{'IMAGES':<20} {len(train_set):>8} {len(val_set):>8} {len(test_set):>8} {len(train_set)+len(val_set)+len(test_set):>8}")
    
    # Summary
    total_negatives = train_neg + val_neg + test_neg
    total_images = len(train_set) + len(val_set) + len(test_set)
    negative_ratio = total_negatives / total_images * 100
    
    print("\n📊 Negative Sample Summary:")
    print(f"   Total negatives: {total_negatives}")
    print(f"   Negative ratio:  {negative_ratio:.1f}% of dataset")
    
    if negative_ratio > 5:
        print(f"   ✅ Good negative ratio (>5%) - helps reduce false positives")
    elif negative_ratio > 0:
        print(f"   ⚠️  Low negative ratio ({negative_ratio:.1f}%) - consider adding more")
    else:
        print(f"   ❌ No negative samples - model may have high false positive rate")
    
    print("="*70 + "\n")


def create_data_yaml(output_dir, original_data_yaml=None):
    """Create data.yaml for the split dataset"""
    output_yaml = Path(output_dir) / "data.yaml"
    
    # Try to read original data.yaml
    class_names = None
    nc = None
    
    if original_data_yaml and Path(original_data_yaml).exists():
        try:
            with open(original_data_yaml, 'r') as f:
                original_data = yaml.safe_load(f)
                class_names = original_data.get('names', None)
                nc = original_data.get('nc', None)
        except Exception as e:
            print(f"Warning: Could not read original data.yaml: {e}")
    
    # Create new data.yaml
    data_config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
    }
    
    if nc is not None:
        data_config['nc'] = nc
    
    if class_names:
        data_config['names'] = class_names
    else:
        print("⚠️  Warning: Could not find class names. Please update data.yaml manually.")
    
    with open(output_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created data.yaml: {output_yaml}")
    
    # Print contents
    print("\n" + "="*70)
    print("DATA.YAML CONTENTS")
    print("="*70)
    with open(output_yaml, 'r') as f:
        print(f.read())
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return
    
    print("="*70)
    print("YOLO DATASET SPLITTING (KEEPS NEGATIVES)")
    print("="*70)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Split ratios:     train={args.train_ratio:.1%}, val={args.val_ratio:.1%}, test={args.test_ratio:.1%}")
    print(f"Stratified:       {args.stratify}")
    print(f"Copy files:       {args.copy}")
    print(f"Random seed:      {args.seed}")
    print("="*70 + "\n")
    
    # Locate images and labels directories
    input_path = Path(args.input)
    
    possible_image_dirs = [
        input_path / "images" / "train",
        input_path / "images",
        input_path / "train" / "images",
        input_path,
    ]
    
    possible_label_dirs = [
        input_path / "labels" / "train",
        input_path / "labels",
        input_path / "train" / "labels",
        input_path,
    ]
    
    images_dir = None
    labels_dir = None
    
    for img_dir in possible_image_dirs:
        if img_dir.exists() and list(img_dir.glob("*.jpg")):
            images_dir = img_dir
            break
    
    for lbl_dir in possible_label_dirs:
        if lbl_dir.exists() and list(lbl_dir.glob("*.txt")):
            labels_dir = lbl_dir
            break
    
    if not images_dir or not labels_dir:
        print("❌ Error: Could not find images/ or labels/ directories")
        return
    
    print(f"📂 Images directory: {images_dir}")
    print(f"📂 Labels directory: {labels_dir}\n")
    
    # Collect dataset information (INCLUDES NEGATIVES)
    dataset_info = collect_dataset_info(images_dir, labels_dir)
    
    if not dataset_info:
        print("❌ Error: No valid samples found")
        return
    
    # Perform split
    if args.stratify:
        print("\nPerforming stratified split (including negatives)...")
        train_set, val_set, test_set = stratified_split(
            dataset_info,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
    else:
        print("\nPerforming random split (including negatives)...")
        all_stems = list(dataset_info.keys())
        random.seed(args.seed)
        random.shuffle(all_stems)
        
        n = len(all_stems)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        
        train_set = all_stems[:n_train]
        val_set = all_stems[n_train:n_train + n_val]
        test_set = all_stems[n_train + n_val:]
    
    # Verify split
    verify_split(dataset_info, train_set, val_set, test_set)
    
    # Copy/link files
    print("Copying files to output directory...\n")
    copy_or_link_files(train_set, dataset_info, args.output, "train", args.copy)
    copy_or_link_files(val_set, dataset_info, args.output, "val", args.copy)
    copy_or_link_files(test_set, dataset_info, args.output, "test", args.copy)
    
    # Create data.yaml
    original_yaml = input_path / "data.yaml"
    create_data_yaml(args.output, original_yaml)
    
    print("="*70)
    print("✅ DATASET SPLIT COMPLETE")
    print("="*70)
    print(f"Output directory: {args.output}")
    print(f"Train: {len(train_set)} images")
    print(f"Val:   {len(val_set)} images")
    print(f"Test:  {len(test_set)} images")
    print(f"Total: {len(train_set) + len(val_set) + len(test_set)} images")
    
    # Count negatives in each split
    train_neg = sum(1 for s in train_set if dataset_info[s]['is_negative'])
    val_neg = sum(1 for s in val_set if dataset_info[s]['is_negative'])
    test_neg = sum(1 for s in test_set if dataset_info[s]['is_negative'])
    
    print(f"\n📊 Negative samples distribution:")
    print(f"   Train negatives: {train_neg}")
    print(f"   Val negatives:   {val_neg}")
    print(f"   Test negatives:  {test_neg}")
    print(f"   Total negatives: {train_neg + val_neg + test_neg}")
    
    print("\n💡 Next steps:")
    print(f"   1. Verify split: ls -R {args.output}")
    print(f"   2. Check data.yaml: cat {args.output}/data.yaml")
    print(f"   3. Train model: python transfer_learning.py --data {args.output}/data.yaml")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()