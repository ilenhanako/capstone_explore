#!/usr/bin/env python3
"""
Remove empty label files before uploading to Colab.
They will be recreated in Colab based on orphaned images.
"""

from pathlib import Path
import shutil

def remove_empty_labels(dataset_path):
    """Remove all empty .txt files from labels directories"""
    
    dataset_path = Path(dataset_path)
    removed_count = 0
    
    # Check all label directories
    label_dirs = [
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
        dataset_path / "labels" / "test",
    ]
    
    print("="*70)
    print("REMOVING EMPTY LABEL FILES")
    print("="*70)
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        
        print(f"\n📂 Checking: {label_dir}")
        
        empty_labels = []
        
        for label_file in label_dir.glob("*.txt"):
            # Check if file is empty
            if label_file.stat().st_size == 0:
                empty_labels.append(label_file)
        
        if empty_labels:
            print(f"   Found {len(empty_labels)} empty label files")
            
            for label_file in empty_labels:
                print(f"   Removing: {label_file.name}")
                label_file.unlink()
                removed_count += 1
        else:
            print(f"   No empty labels found")
    
    print("\n" + "="*70)
    print(f"✅ REMOVED {removed_count} EMPTY LABEL FILES")
    print("="*70)
    print("\nThese will be recreated as placeholders in Colab")
    print("to mark negative samples (images without defects)")
    
    return removed_count


def create_removal_report(dataset_path):
    """Create a report of images that will need empty labels recreated"""
    
    dataset_path = Path(dataset_path)
    
    report = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        # Find images without labels
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # If label doesn't exist, this is a negative sample
            if not label_file.exists():
                report[split].append(img_file.stem)
        
        # Find images with empty labels (will be removed)
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size == 0:
                report[split].append(label_file.stem)
    
    # Save report
    report_file = dataset_path / "negative_samples_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("NEGATIVE SAMPLES (Images without defects)\n")
        f.write("="*70 + "\n\n")
        
        for split in ['train', 'val', 'test']:
            f.write(f"{split.upper()}:\n")
            f.write(f"  Count: {len(report[split])}\n")
            
            if report[split]:
                f.write(f"  Files:\n")
                for stem in sorted(report[split]):
                    f.write(f"    - {stem}\n")
            
            f.write("\n")
    
    print(f"\n📄 Created report: {report_file}")
    print(f"   Train negatives: {len(report['train'])}")
    print(f"   Val negatives:   {len(report['val'])}")
    print(f"   Test negatives:  {len(report['test'])}")
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python remove_empty_labels.py <dataset_path>")
        print("Example: python remove_empty_labels.py training_dataset_split")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Create report first
    print("\n📊 Creating negative samples report...")
    report = create_removal_report(dataset_path)
    
    # Remove empty labels
    removed = remove_empty_labels(dataset_path)
    
    print("\n💡 Next steps:")
    print("   1. ZIP the dataset: zip -r dataset.zip training_dataset_split/")
    print("   2. Upload to Google Drive or Colab")
    print("   3. Run the recreation script in Colab (see below)")