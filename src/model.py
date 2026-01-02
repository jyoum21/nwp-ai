import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from data.process import HurricaneH5Dataset


class HurricaneWindCNN(nn.Module):
    
    def __init__(self, dropout_rate: float = 0.5, freeze_backbone: bool = False):
        super().__init__()
        
        # load resnet-50 with random weights
        self.backbone = models.resnet50(weights=None)
        
        # first conv layer
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, # single channel input (BW vs RGB)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # kaiming initialization (fan_in is PyTorch default and preserves forward pass variance)
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_in', nonlinearity='relu')
        
        # optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # replace classifier with regression head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # initialize regression head
        self._init_regression_head()
    
    # initialize regression head with kaiming initialization
    def _init_regression_head(self):
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    # freeze all backbone layers except the regression head
    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    # unfreeze backbone layers
    def unfreeze_backbone(self, num_layers: int = None):
        if num_layers is None:
            # unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # resnet-50 has: conv1, bn1, layer1, layer2, layer3, layer4, fc
            # unfreeze from the end
            layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
            layers_to_unfreeze = layer_names[:num_layers]
            
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze) or 'fc' in name:
                    param.requires_grad = True
    
    # forward pass, input [batch_size, 1, H, W]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x) # returns wind speed prediction [batch_size, 1]
    
    # get total and trainable parameters
    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

# factory function to build and configure model
def build_model(dropout_rate: float = 0.5, freeze_backbone: bool = False, device: str = None) -> HurricaneWindCNN:
    model = HurricaneWindCNN(
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
   print("no need to run model.py as a script! it is used in train.py.")
