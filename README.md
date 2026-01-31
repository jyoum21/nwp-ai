# nwp-ai

## **WIP**

## End goal:
Input a satellite image of a tropical system, and nwp-ai will return an estimated wind speed and pressure, as well as (de)intensification predictions.

## Model architecture:
**ResNet-50 (randomly initialized):**
- Modified input conv: 1 channel (b/w instead of rgb) → 64 filters (7×7, stride 2)
- Standard residual blocks: layer1 (3) → layer2 (4) → layer3 (6) → layer4 (3)
- Global average pooling → 2048-dim feature vector

**Regression Head:**
Dropout(0.5) → Linear(2048, 512) → ReLU
    → Dropout(0.25) → Linear(512, 128) → ReLU
    → Linear(128, 1)

## I/O

- **Input:** `[batch, 1, H, W]` grayscale satellite image
- **Output:** `[batch, 1]` predicted wind speed (kt)
