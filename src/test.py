import h5py
import numpy as np

filename = 'data/hurricane_data.h5'

def _decode_string(value):
    """Helper function to decode h5py string values"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.bytes_):
        return value.decode('utf-8')
    return str(value)

with h5py.File(filename, 'r') as f:
    print("=" * 60)
    print("HDF5 File Structure")
    print("=" * 60)
    print(f"Top-level groups: {list(f.keys())}")
    print()
    
    # Print global attributes
    print("Global Attributes:")
    for key in f.attrs.keys():
        print(f"  {key}: {f.attrs[key]}")
    print()
    
    # Access images group
    if 'images' in f:
        print("Images Group:")
        print(f"  Datasets: {list(f['images'].keys())}")
        images_data = f['images/data']
        print(f"  Images shape: {images_data.shape}")
        print(f"  Images dtype: {images_data.dtype}")
        if images_data.shape[0] > 0:
            print(f"  First image shape: {images_data[0].shape}")
            print(f"  First image min: {np.min(images_data[0])}")
            print(f"  First image max: {np.max(images_data[0])}")
            print(f"  First image mean: {np.mean(images_data[0]):.2f}")
        print()
    
    # Access metadata group
    if 'metadata' in f:
        print("Metadata Group:")
        print(f"  Datasets: {list(f['metadata'].keys())}")
        
        # Print metadata for first few samples
        num_samples = f.attrs.get('total_samples', 0)
        if num_samples > 0:
            print(f"\n  First 5 samples:")
            for i in range(min(5, num_samples)):
                print(f"    Sample {i}:")
                print(f"      Year: {f['metadata/years'][i]}")
                storm_name = _decode_string(f['metadata/storm_names'][i])
                print(f"      Storm: {storm_name}")
                print(f"      Date: {f['metadata/dates'][i]}")
                print(f"      Time: {f['metadata/times'][i]}")
                print(f"      Wind Speed: {f['metadata/wind_speeds'][i]}")
                print(f"      Latitude: {f['metadata/latitudes'][i]}")
                print(f"      Longitude: {f['metadata/longitudes'][i]}")
                print(f"      Pressure: {f['metadata/pressures'][i]}")
        print()
    
    print("=" * 60)
    print(f"Total samples in database: {f.attrs.get('total_samples', 0)}")
    print("=" * 60)
