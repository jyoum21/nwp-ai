"""
ResNet-50 model for hurricane wind speed estimation.

Handles single-channel (infrared) satellite imagery input.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from data.process import HurricaneH5Dataset


class HurricaneWindCNN(nn.Module):
    """
    ResNet-50 adapted for single-channel satellite imagery regression.
    
    Input: [batch_size, 1, 301, 301] - Single channel IR image
    Output: [batch_size, 1] - Wind speed estimate
    
    Architecture:
        - Modified first conv layer (1 channel → 64 filters)
        - ResNet-50 backbone (pretrained on ImageNet)
        - Custom regression head with dropout
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Args:
            pretrained: Load ImageNet pretrained weights (adapted for 1-channel)
            dropout_rate: Dropout probability in regression head
            freeze_backbone: Freeze backbone layers for transfer learning
        """
        super().__init__()
        
        # Load ResNet-50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Modify first conv layer: 3 channels → 1 channel
        # Keep the same output (64 filters, 7x7 kernel, stride 2, padding 3)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize new conv layer using pretrained weights (sum across RGB channels)
        if pretrained:
            with torch.no_grad():
                # Average the 3-channel weights to create 1-channel weights
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # Replace classifier with regression head
        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Initialize regression head
        self._init_regression_head()
    
    def _init_regression_head(self):
        """Initialize the regression head with Kaiming initialization."""
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _freeze_backbone(self):
        """Freeze all backbone layers except the regression head."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self, num_layers: int = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layer groups to unfreeze from the end.
                       None = unfreeze all. Options: 1-4 for ResNet layer groups.
        """
        if num_layers is None:
            # Unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # ResNet-50 has: conv1, bn1, layer1, layer2, layer3, layer4, fc
            # Unfreeze from the end
            layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
            layers_to_unfreeze = layer_names[:num_layers]
            
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze) or 'fc' in name:
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 1, H, W]
        
        Returns:
            Wind speed predictions of shape [batch_size, 1]
        """
        return self.backbone(x)
    
    def get_num_params(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def build_model(
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    device: str = None
) -> HurricaneWindCNN:
    """
    Factory function to build and configure the model.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Freeze backbone for transfer learning
        device: Device to place model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        Configured HurricaneWindCNN model
    """
    model = HurricaneWindCNN(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    params = model.get_num_params()
    print(f"Model initialized on {device}")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    
    return model


if __name__ == '__main__':
    # Quick test with real data
    model = build_model(pretrained=True, freeze_backbone=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load real data from dataset
    data_path = 'data/hurricane_data.h5'
    dataset = HurricaneH5Dataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch of real data
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    print(f"\nLoaded {len(dataset)} samples from {data_path}")
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"\nBatch predictions vs actual wind speeds:")
    print("-" * 50)
    for i in range(len(predictions)):
        pred = predictions[i].item()
        actual = labels[i].item()
        error = abs(pred - actual)
        print(f"  Sample {i+1}: Predicted {pred:.1f} kt | Actual {actual:.1f} kt | Error {error:.1f} kt")
    
    # Clean up
    dataset.close()
