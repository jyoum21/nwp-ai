from data.process import HurricaneH5Dataset
import torch

# 1. Initialize the dataset
ds = HurricaneH5Dataset('data/hurricane_data.h5')

# 2. Grab the first sample (this triggers __getitem__)
# This is where the math (X - min) / (max - min) happens!
image_tensor, label = ds[0]

print("--- Verification ---")
print(f"Original Data Type: {type(image_tensor)}")
print(f"Shape: {image_tensor.shape}") # Should be [1, 301, 301]

# 3. Check the values
print(f"Max Value: {image_tensor.max().item():.4f}") # Should be <= 1.0
print(f"Min Value: {image_tensor.min().item():.4f}") # Should be >= 0.0

