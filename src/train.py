"""
Training script for hurricane wind speed estimation.

Usage:
    python src/train.py --data data/hurricane_data.h5 --epochs 50
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms

import sys
import os
# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.process import HurricaneH5Dataset
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train hurricane wind speed model')
    parser.add_argument('--data', type=str, default='data/hurricane_data.h5', help='Path to H5 file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to save model')
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate MAE
    import numpy as np
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return avg_loss, mae


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading data from {args.data}...")
    full_dataset = HurricaneH5Dataset(args.data)
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}, Validation: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(dropout_rate=0.5, device=str(device))
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.2f} kt | "
              f"LR: {current_lr:.2e}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE (knots)')
    plt.title('Validation MAE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=150)
    print(f"\nTraining curves saved to {args.save_dir}/training_curves.png")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best model saved to: {args.save_dir}/best_model.pth")
    print(f"Best validation MAE: {min(history['val_mae']):.2f} knots")
    print("="*50)


if __name__ == '__main__':
    main()

