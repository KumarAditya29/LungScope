#!/usr/bin/env python3
"""Test script to verify AI models setup"""

import sys
from pathlib import Path

# Add both root directory and src to path
sys.path.append(str(Path(__file__).parent.parent))  # Add ai-models root
sys.path.append(str(Path(__file__).parent.parent / "src"))  # Add src directory

import torch
import numpy as np
from models.ddpm import UNet, DDPM
from data.preprocessor import DataPreprocessor
from configs.training_config import ddpm_config, DISEASE_CLASSES

def test_pytorch_setup():
    """Test PyTorch installation and GPU availability"""
    print("🔧 Testing PyTorch Setup...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")  # Apple Silicon GPU
    print("✅ PyTorch setup OK!\n")

def test_ddpm_model():
    """Test DDPM model creation and forward pass"""
    print("🧠 Testing DDPM Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Create model
    model = UNet(in_channels=3, out_channels=3)
    ddpm = DDPM(model, device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    with torch.no_grad():
        noise_pred = model(x, t)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {noise_pred.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ DDPM model test passed!\n")

def test_data_preprocessor():
    """Test data preprocessing utilities"""
    print("📊 Testing Data Preprocessor...")
    
    preprocessor = DataPreprocessor()
    
    # Test transforms
    train_transform = preprocessor.get_train_transforms()
    val_transform = preprocessor.get_val_transforms()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test transforms
    train_result = train_transform(image=dummy_image)
    val_result = val_transform(image=dummy_image)
    
    print(f"   Train transform output shape: {train_result['image'].shape}")
    print(f"   Val transform output shape: {val_result['image'].shape}")
    print("✅ Data preprocessor test passed!\n")

def test_disease_classes():
    """Test disease classes configuration"""
    print("🏥 Testing Disease Classes...")
    print(f"   Number of disease classes: {len(DISEASE_CLASSES)}")
    print(f"   Disease classes: {DISEASE_CLASSES[:5]}... (showing first 5)")
    print("✅ Disease classes loaded!\n")

def main():
    """Run all tests"""
    print("🚀 LungScope AI Models Setup Test\n")
    print("=" * 50)
    
    try:
        test_pytorch_setup()
        test_ddpm_model()
        test_data_preprocessor()
        test_disease_classes()
        
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your AI models environment is ready for development!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("🔧 Please check your setup and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
