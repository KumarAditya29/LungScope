import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessor import ChestXRayDataset, DataPreprocessor
from data.nih_labels import NIHLabelsParser

def build_chest_xray_loaders(
    data_dir="data/raw",
    batch_size=32,
    num_workers=4,
    max_samples=1000,
    train_split=0.8,
    image_size=224,
    seed=42
):
    """
    Build train and validation DataLoaders for chest X-ray classification
    
    Args:
        data_dir: Directory containing raw data
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        max_samples: Maximum samples to use (for testing)
        train_split: Fraction of data for training
        image_size: Size to resize images to
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, num_classes, class_names)
    """
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("🔄 Building Chest X-ray DataLoaders...")
    
    # Parse labels
    parser = NIHLabelsParser(data_dir)
    image_paths, labels, metadata = parser.parse_labels(max_samples=max_samples)
    
    if len(image_paths) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    # Create preprocessor with updated transforms
    preprocessor = DataPreprocessor()
    
    # Update transforms to use specified image size
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Training transforms (with augmentation)
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        # Use Affine instead of ShiftScaleRotate to avoid warning
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create full dataset first
    full_dataset = ChestXRayDataset(
        image_paths=image_paths,
        labels=labels,
        transform=train_transform  # Will be updated for val set
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    print(f"📊 Dataset split: {train_size} train, {val_size} validation")
    
    # Create train and validation datasets
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Update transforms for each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
        drop_last=False
    )
    
    # Get class information
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from configs.training_config import DISEASE_CLASSES
    num_classes = len(DISEASE_CLASSES)
    class_names = DISEASE_CLASSES
    
    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Image size: {image_size}x{image_size}")
    
    return train_loader, val_loader, num_classes, class_names

def test_loaders():
    """Test the DataLoader creation"""
    print("🧪 Testing DataLoader creation...")
    
    try:
        train_loader, val_loader, num_classes, class_names = build_chest_xray_loaders(
            max_samples=100,
            batch_size=8,
            num_workers=0  # Avoid multiprocessing issues in testing
        )
        
        # Test a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"✅ Test batch {batch_idx}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Images dtype: {images.dtype}")
            print(f"   Labels dtype: {labels.dtype}")
            print(f"   Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   Positive labels in batch: {labels.sum().item()}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        print("🎉 DataLoader test passed!")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        raise e

if __name__ == "__main__":
    test_loaders()