#!/usr/bin/env python3
"""
Smoke test training script for chest X-ray classification
Runs a minimal training loop to verify the full pipeline works
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import time

# Add both src and root to path
sys.path.append(str(Path(__file__).parent.parent))  # Add ai-models root
sys.path.append(str(Path(__file__).parent.parent / "src"))  # Add src

from data.build_loaders import build_chest_xray_loaders
from configs.training_config import DISEASE_CLASSES


class ChestXrayClassifier(nn.Module):
    """Simple ResNet-based classifier for chest X-rays"""
    
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=False)
        
        # Replace final layer for multi-label classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Batch': f'{batch_idx+1}/{len(train_loader)}'
        })
        
        # Early stopping for smoke test
        if batch_idx >= 10:  # Only train on first 10 batches for smoke test
            print(f"🔄 Smoke test: stopping after {batch_idx+1} batches")
            break
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Batch': f'{batch_idx+1}/{len(val_loader)}'
            })
            
            # Early stopping for smoke test
            if batch_idx >= 5:  # Only validate on first 5 batches
                print(f"🔄 Smoke test: stopping after {batch_idx+1} batches")
                break
    
    return total_loss / num_batches

def main():
    """Main training function"""
    print("🚀 Starting Chest X-ray Classifier Smoke Test\n")
    
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 1,  # Just 1 epoch for smoke test
        'max_samples': 200,  # Small dataset for testing
        'image_size': 224,
        'num_workers': 2,
        'save_dir': 'models/checkpoints',
    }
    
    print("⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔥 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔥 Using CUDA")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    print(f"   Device: {device}\n")
    
    # Build data loaders
    print("📊 Building data loaders...")
    try:
        train_loader, val_loader, num_classes, class_names = build_chest_xray_loaders(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_samples=config['max_samples'],
            image_size=config['image_size']
        )
        print("✅ Data loaders created successfully!\n")
        
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        return
    
    # Create model
    print("🧠 Creating model...")
    model = ChestXrayClassifier(num_classes=num_classes, pretrained=False)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("✅ Model created successfully!\n")
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    print("🎯 Training components ready!")
    print(f"   Loss function: {criterion.__class__.__name__}")
    print(f"   Optimizer: {optimizer.__class__.__name__}")
    print(f"   Learning rate: {config['learning_rate']}\n")
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("🚀 Starting training...\n")
    
    for epoch in range(config['num_epochs']):
        print(f"📈 Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_time = time.time() - start_time
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} (Time: {train_time:.1f}s)")
        print(f"   Val Loss: {val_loss:.4f} (Time: {val_time:.1f}s)")
        print()
    
    # Save model
    save_path = save_dir / "smoke_test_classifier.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'num_classes': num_classes,
        'class_names': class_names,
    }, save_path)
    
    print(f"💾 Model saved to: {save_path}")
    
    # Test inference
    print("\n🔍 Testing inference...")
    model.eval()
    
    with torch.no_grad():
        # Get a batch from validation loader
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            
            print(f"   Input shape: {images.shape}")
            print(f"   Output shape: {outputs.shape}")
            print(f"   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            
            # Show predictions for first sample
            first_sample_probs = probabilities[0].cpu().numpy()
            first_sample_labels = labels[0].cpu().numpy()
            
            print(f"\n   First sample predictions:")
            for i, (disease, prob, true_label) in enumerate(zip(class_names, first_sample_probs, first_sample_labels)):
                if prob > 0.1 or true_label > 0:  # Show only interesting predictions
                    print(f"     {disease}: {prob:.3f} (true: {int(true_label)})")
            
            break
    
    print("\n🎉 Smoke test completed successfully!")
    print("✅ All components are working:")
    print("   - Data loading ✓")
    print("   - Model forward pass ✓") 
    print("   - Loss computation ✓")
    print("   - Backpropagation ✓")
    print("   - Device handling (MPS) ✓")
    print("   - Model saving ✓")
    print("   - Inference ✓")
    
    print(f"\n🚀 Ready for full training! Scale up by:")
    print("   - Increasing max_samples (e.g., 10000)")
    print("   - Adding more epochs (e.g., 20-50)")
    print("   - Adding learning rate scheduling")
    print("   - Adding metrics (AUROC, etc.)")
    print("   - Adding W&B logging")

if __name__ == "__main__":
    main()