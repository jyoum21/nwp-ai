"""
Inference script for hurricane wind speed prediction.

Usage:
    # Predict from H5 file (shows first N samples)
    python src/predict.py --model outputs/best_model.pth --data data/hurricane_data.h5
    
    # Predict on a specific index in the H5 file
    python src/predict.py --model outputs/best_model.pth --data data/hurricane_data.h5 --index 5
    
    # Predict from a numpy file
    python src/predict.py --model outputs/best_model.pth --image path/to/image.npy
    
    # Predict from a regular image file (PNG, JPG, etc.)
    python src/predict.py --model outputs/best_model.pth --image path/to/hurricane.png
"""

import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HurricaneWindCNN


# Physical constraints (must match process.py)
MIN_TEMP = 150.0
MAX_TEMP = 340.0


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = HurricaneWindCNN()
    
    # weights_only=False needed for PyTorch 2.6+ when checkpoint contains numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'val_mae' in checkpoint:
        print(f"  Validation MAE: {checkpoint['val_mae']:.2f} kt")
    
    return model


def load_image_file(image_path):
    """
    Load and preprocess a regular image file (PNG, JPG, etc.).
    
    Args:
        image_path: Path to image file
    
    Returns:
        numpy array of shape [H, W] in Kelvin range (150-340)
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Resize to 301x301 if needed (model expects this size)
    if img_array.shape != (301, 301):
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((301, 301), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil, dtype=np.float32)
    
    # Scale pixel values (0-255) to Kelvin range (150-340)
    # This assumes the image represents brightness temperature
    # If your image is already in Kelvin, you can skip this scaling
    img_array = (img_array / 255.0) * (MAX_TEMP - MIN_TEMP) + MIN_TEMP
    
    return img_array


def preprocess_image(image):
    """
    Preprocess a raw image (in Kelvin) to model input format.
    
    Args:
        image: numpy array of shape [H, W] with values in Kelvin
    
    Returns:
        torch tensor of shape [1, 1, H, W] normalized to [0, 1]
    """
    # Normalize to [0, 1]
    normalized = (image - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
    tensor = torch.from_numpy(normalized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor


def predict_single(model, image, device):
    """
    Predict wind speed for a single image.
    
    Args:
        model: Trained model
        image: numpy array [H, W] in Kelvin, or [1, H, W] normalized
        device: torch device
    
    Returns:
        Predicted wind speed in knots
    """
    model.eval()
    
    # Check if already preprocessed (has 3 dims and values in [0,1])
    if len(image.shape) == 2:
        tensor = preprocess_image(image)
    else:
        # Assume already preprocessed [1, H, W]
        tensor = torch.from_numpy(image).float().unsqueeze(0)
    
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    return output.item()


def predict_from_h5(model, h5_path, device, indices=None, show_plot=True, output_dir='outputs'):
    """
    Predict on samples from an H5 file.
    
    Args:
        model: Trained model
        h5_path: Path to H5 file
        device: torch device
        indices: List of indices to predict (None = first 10)
        show_plot: Whether to display visualization
        output_dir: Directory to save visualization
    """
    with h5py.File(h5_path, 'r') as f:
        total = f.attrs['total_samples']
        
        if indices is None:
            indices = list(range(min(10, total)))
        
        print(f"\nPredicting on {len(indices)} samples from {h5_path}")
        print("-" * 60)
        print(f"{'Index':<8} {'Predicted':>12} {'Actual':>12} {'Error':>12}")
        print("-" * 60)
        
        predictions = []
        actuals = []
        images_to_show = []
        
        for idx in indices:
            # Load and preprocess image
            raw_image = f['images/data'][idx]
            actual_wind = f['metadata/wind_speeds'][idx]
            
            # Predict
            pred_wind = predict_single(model, raw_image, device)
            error = pred_wind - actual_wind
            
            predictions.append(pred_wind)
            actuals.append(actual_wind)
            images_to_show.append(raw_image)
            
            print(f"{idx:<8} {pred_wind:>10.1f} kt {actual_wind:>10.1f} kt {error:>+10.1f} kt")
        
        # Summary statistics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        errors = np.abs(predictions - actuals)
        
        print("-" * 60)
        print(f"MAE: {errors.mean():.2f} kt")
        print(f"RMSE: {np.sqrt((errors**2).mean()):.2f} kt")
        
        # Visualization
        if show_plot and len(indices) <= 10:
            n_images = len(indices)
            fig, axes = plt.subplots(2, min(5, n_images), figsize=(15, 6))
            
            if n_images <= 5:
                axes = axes.reshape(2, -1)
            
            for i, (img, pred, actual) in enumerate(zip(images_to_show[:5], predictions[:5], actuals[:5])):
                row = i // 5
                col = i % 5
                
                if n_images > 5:
                    ax = axes[row, col]
                else:
                    ax = axes[0, i] if n_images > 1 else axes[0]
                
                ax.imshow(img, cmap='inferno')
                ax.set_title(f'Pred: {pred:.1f} kt\nActual: {actual:.1f} kt', fontsize=10)
                ax.axis('off')
            
            # Hide empty subplots
            for i in range(n_images, 5):
                if n_images > 1:
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'predictions_visualization.png')
            plt.savefig(output_path, dpi=150)
            print(f"\nVisualization saved to {output_path}")
            plt.show()


def predict_from_numpy(model, npy_path, device):
    """
    Predict on a single numpy image file.
    
    Args:
        model: Trained model
        npy_path: Path to .npy file containing image
        device: torch device
    """
    image = np.load(npy_path)
    print(f"\nLoaded image from {npy_path}")
    print(f"  Shape: {image.shape}")
    print(f"  Range: {image.min():.1f} - {image.max():.1f}")
    
    pred_wind = predict_single(model, image, device)
    print(f"\nPredicted wind speed: {pred_wind:.1f} knots")
    
    # Show the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='inferno')
    plt.colorbar(label='Brightness Temperature (K)')
    plt.title(f'Predicted Wind Speed: {pred_wind:.1f} kt')
    plt.savefig('single_prediction.png', dpi=150)
    print("Image saved to single_prediction.png")
    plt.show()
    
    return pred_wind


def predict_from_image_file(model, image_path, device):
    """
    Predict on a regular image file (PNG, JPG, etc.).
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: torch device
    """
    print(f"\nLoading image from {image_path}")
    
    # Load and preprocess image
    image = load_image_file(image_path)
    
    print(f"  Shape: {image.shape}")
    print(f"  Temperature range: {image.min():.1f}K - {image.max():.1f}K")
    
    # Predict
    pred_wind = predict_single(model, image, device)
    print(f"\n{'='*60}")
    print(f"Predicted wind speed: {pred_wind:.1f} knots")
    print(f"{'='*60}")
    
    # Show the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='inferno')
    plt.colorbar(label='Brightness Temperature (K)')
    plt.title(f'Hurricane Image\nPredicted Wind Speed: {pred_wind:.1f} knots', fontsize=14)
    plt.axis('off')
    
    output_path = 'prediction_from_image.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.show()
    
    return pred_wind


def main():
    parser = argparse.ArgumentParser(description='Predict hurricane wind speeds')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None, help='Path to H5 data file')
    parser.add_argument('--image', type=str, default=None, help='Path to image file (.npy, .png, .jpg, etc.)')
    parser.add_argument('--index', type=int, default=None, help='Specific index in H5 file')
    parser.add_argument('--n', type=int, default=10, help='Number of samples to predict')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    # Run prediction
    if args.image:
        # Check file extension to determine how to load
        ext = os.path.splitext(args.image)[1].lower()
        if ext == '.npy':
            predict_from_numpy(model, args.image, device)
        else:
            # Regular image file (PNG, JPG, etc.)
            predict_from_image_file(model, args.image, device)
    elif args.data:
        if args.index is not None:
            indices = [args.index]
        else:
            indices = list(range(args.n))
        predict_from_h5(model, args.data, device, indices, show_plot=not args.no_plot, output_dir=args.output_dir)
    else:
        print("Error: Provide either --data or --image")


if __name__ == '__main__':
    main()

