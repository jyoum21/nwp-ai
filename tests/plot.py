import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def _decode_string(value):
    """Helper function to decode h5py string values"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.bytes_):
        return value.decode('utf-8')
    return str(value)

def animate_first_storm(db_path='data/hurricane_data.h5', dt=0.2):
    """
    Animate all timesteps from the first storm in data/hurricane_data.h5.
    
    Args:
        db_path: Path to the HDF5 database file
        dt: Time interval between frames in seconds (default: 0.2)
    """
    # Open the HDF5 file
    with h5py.File(db_path, 'r') as f:
        # Get the images dataset and metadata
        images = f['images/data']
        storm_names = f['metadata/storm_names'][:]
        years = f['metadata/years'][:]
        dates = f['metadata/dates'][:]
        times = f['metadata/times'][:]
        
        # Find the first unique storm (by name and year)
        first_storm_key = None
        first_storm_indices = []
        
        for i in range(len(storm_names)):
            storm_name = _decode_string(storm_names[i])
            year = int(years[i])
            storm_key = (year, storm_name)
            
            if first_storm_key is None:
                first_storm_key = storm_key
                first_storm_indices.append(i)
            elif storm_key == first_storm_key:
                first_storm_indices.append(i)
            else:
                # We've moved to a different storm, stop collecting
                break
        
        if not first_storm_indices:
            print("No storms found in database!")
            return
        
        # Get storm info
        first_idx = first_storm_indices[0]
        storm_name = _decode_string(storm_names[first_idx])
        year = int(years[first_idx])
        
        print(f"Found first storm: {storm_name} ({year})")
        print(f"Total timesteps: {len(first_storm_indices)}")
        
        # Extract all timesteps for the first storm
        storm_images = []
        storm_info = []
        for idx in first_storm_indices:
            image = images[idx]
            storm_images.append(image)
            storm_info.append({
                'date': int(dates[idx]),
                'time': int(times[idx]),
                'index': idx
            })
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize the plot with the first image
        im = ax.imshow(storm_images[0], cmap='viridis', origin='upper', animated=True)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Title will be updated in animation
        title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                            ha='center', va='bottom', fontsize=14, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def animate(frame):
            """Update function for animation"""
            idx = frame % len(storm_images)
            image = storm_images[idx]
            info = storm_info[idx]
            
            # Update image
            im.set_array(image)
            
            # Update title
            title_text.set_text(f'{storm_name} ({year}) - Timestep {idx+1}/{len(storm_images)}\n'
                              f'Date: {info["date"]}, Time: {info["time"]}')
            
            # Print to terminal
            print(f"\nTimestep {idx+1}/{len(storm_images)} - Date: {info['date']}, Time: {info['time']}")
            print(f"  Min: {np.min(image):.2f}, Max: {np.max(image):.2f}, Mean: {np.mean(image):.2f}")
            
            return [im, title_text]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(storm_images), 
                                     interval=dt*1000, blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

if __name__ == '__main__':
    animate_first_storm()

