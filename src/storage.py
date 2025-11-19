import h5py
import matplotlib.pyplot as plt
import numpy as np

def _decode_string(value):
    """Helper function to decode h5py string values"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.bytes_):
        return value.decode('utf-8')
    return str(value)

def plot_first_timestep_from_four_storms(db_path='hurricane_data.h5'):
    """
    Extract the first timestep from the first four unique storms in hurricane_data.h5
    and plot them using matplotlib subplots.
    """
    # Open the HDF5 file
    with h5py.File(db_path, 'r') as f:
        # Get the images dataset and metadata
        images = f['images/data']
        storm_names = f['metadata/storm_names'][:]
        years = f['metadata/years'][:]
        
        # Find first occurrence of each unique storm (by name and year)
        seen_storms = {}
        storm_indices = []
        
        for i in range(len(storm_names)):
            storm_name = _decode_string(storm_names[i])
            year = int(years[i])
            storm_key = (year, storm_name)
            
            if storm_key not in seen_storms:
                seen_storms[storm_key] = i
                storm_indices.append(i)
                
                if len(storm_indices) >= 4:
                    break
        
        if len(storm_indices) < 4:
            print(f"Warning: Only found {len(storm_indices)} unique storms, but need 4")
        
        # Extract the first timestep for each of the first 4 storms
        selected_images = []
        selected_info = []
        
        for idx in storm_indices[:4]:
            image = images[idx]
            storm_name = _decode_string(storm_names[idx])
            year = int(years[idx])
            selected_images.append(image)
            selected_info.append((year, storm_name, idx))
        
        # Create subplots: 2 rows, 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('First Timestep from First Four Storms', fontsize=16)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Plot each image
        for i, (image, (year, storm_name, idx)) in enumerate(zip(selected_images, selected_info)):
            ax = axes_flat[i]
            
            # Print the image data to terminal
            print(f"\n{'='*60}")
            print(f"Storm {i+1}: {storm_name} ({year}) - images/data[{idx}]:")
            print(f"{'='*60}")
            print(image)
            print(f"Shape: {image.shape}")
            print(f"Min: {np.min(image)}, Max: {np.max(image)}, Mean: {np.mean(image):.2f}")
            
            # Plot the image
            im = ax.imshow(image, cmap='viridis', origin='upper')
            ax.set_title(f'{storm_name} ({year})')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    plot_first_timestep_from_four_storms()

