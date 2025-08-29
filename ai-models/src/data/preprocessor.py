import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images"""
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        
        return image

class DataPreprocessor:
    """Data preprocessing utilities"""
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def get_train_transforms(self):
        """Training data augmentation"""
        return A.Compose([
            A.Resize(256, 256),
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_val_transforms(self):
        """Validation data preprocessing"""
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    print("✅ Data preprocessor created successfully!")
    print(f"📁 Raw data directory: {preprocessor.raw_dir}")
    print(f"📁 Processed data directory: {preprocessor.processed_dir}")