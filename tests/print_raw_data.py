import h5py
import numpy as np
import os
from collections import defaultdict

def _decode_string(value):
    """Helper function to decode h5py string values"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.bytes_):
        return value.decode('utf-8')
    return str(value)

def print_hurricane_data(db_path='data/hurricane_data.h5'):
    """Print hurricane_data.h5 in a readable format."""
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} does not exist!")
        return
    
    with h5py.File(db_path, 'r') as f:
        print("=" * 80)
        print("HURRICANE DATA H5 FILE SUMMARY")
        print("=" * 80)
        
        # File information
        file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"\nFile: {db_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Global attributes
        total_samples = f.attrs.get('total_samples', 0)
        print(f"Total samples: {total_samples:,}")
        
        if total_samples == 0:
            print("\nDatabase is empty.")
            return
        
        # Image information
        if 'images' in f and 'data' in f['images']:
            images = f['images/data']
            print(f"Image shape: {images.shape}")
            print(f"Image dtype: {images.dtype}")
            if images.shape[0] > 0:
                print(f"First image shape: {images[0].shape}")
                print(f"First image value range: [{np.min(images[0]):.2f}, {np.max(images[0]):.2f}]")
        
        # Metadata information
        print("\n" + "-" * 80)
        print("METADATA SUMMARY")
        print("-" * 80)
        
        if 'metadata' in f:
            meta = f['metadata']
            
            # Get all metadata arrays
            years = meta['years'][:] if 'years' in meta else []
            storm_names = meta['storm_names'][:] if 'storm_names' in meta else []
            dates = meta['dates'][:] if 'dates' in meta else []
            times = meta['times'][:] if 'times' in meta else []
            wind_speeds = meta['wind_speeds'][:] if 'wind_speeds' in meta else []
            latitudes = meta['latitudes'][:] if 'latitudes' in meta else []
            longitudes = meta['longitudes'][:] if 'longitudes' in meta else []
            pressures = meta['pressures'][:] if 'pressures' in meta else []
            
            # Statistics
            if len(wind_speeds) > 0:
                print(f"\nWind Speed Statistics:")
                print(f"  Min: {np.min(wind_speeds):.2f} knots")
                print(f"  Max: {np.max(wind_speeds):.2f} knots")
                print(f"  Mean: {np.mean(wind_speeds):.2f} knots")
                print(f"  Std: {np.std(wind_speeds):.2f} knots")
            
            if len(pressures) > 0:
                print(f"\nPressure Statistics:")
                print(f"  Min: {np.min(pressures):.2f} mb")
                print(f"  Max: {np.max(pressures):.2f} mb")
                print(f"  Mean: {np.mean(pressures):.2f} mb")
            
            if len(latitudes) > 0:
                print(f"\nLocation Statistics:")
                print(f"  Latitude range: [{np.min(latitudes):.2f}, {np.max(latitudes):.2f}]")
                print(f"  Longitude range: [{np.min(longitudes):.2f}, {np.max(longitudes):.2f}]")
            
            # Storm breakdown
            print("\n" + "-" * 80)
            print("STORM BREAKDOWN")
            print("-" * 80)
            
            storm_counts = defaultdict(int)
            for i in range(len(storm_names)):
                storm_name = _decode_string(storm_names[i])
                year = int(years[i]) if len(years) > i else 0
                storm_key = (year, storm_name)
                storm_counts[storm_key] += 1
            
            print(f"\nTotal unique storms: {len(storm_counts)}")
            print("\nStorms and timestep counts:")
            print("-" * 80)
            
            # Sort by year, then by storm name
            sorted_storms = sorted(storm_counts.items(), key=lambda x: (x[0][0], x[0][1]))
            
            for (year, storm_name), count in sorted_storms:
                print(f"  {storm_name:15s} ({year}): {count:5d} timesteps")
            
            # Year breakdown
            year_counts = defaultdict(int)
            for year in years:
                year_counts[int(year)] += 1
            
            print("\n" + "-" * 80)
            print("YEAR BREAKDOWN")
            print("-" * 80)
            for year in sorted(year_counts.keys()):
                print(f"  {year}: {year_counts[year]:,} timesteps")
        
        # Sample records
        print("\n" + "-" * 80)
        print("SAMPLE RECORDS (First 5)")
        print("-" * 80)
        
        num_samples = min(5, total_samples)
        for i in range(num_samples):
            print(f"\nRecord {i+1}:")
            if 'metadata' in f:
                meta = f['metadata']
                if 'storm_names' in meta and i < len(meta['storm_names']):
                    storm_name = _decode_string(meta['storm_names'][i])
                    print(f"  Storm: {storm_name}")
                if 'years' in meta and i < len(meta['years']):
                    print(f"  Year: {int(meta['years'][i])}")
                if 'dates' in meta and i < len(meta['dates']):
                    print(f"  Date: {int(meta['dates'][i])}")
                if 'times' in meta and i < len(meta['times']):
                    print(f"  Time: {int(meta['times'][i])}")
                if 'wind_speeds' in meta and i < len(meta['wind_speeds']):
                    print(f"  Wind Speed: {meta['wind_speeds'][i]:.2f} knots")
                if 'latitudes' in meta and i < len(meta['latitudes']):
                    print(f"  Latitude: {meta['latitudes'][i]:.2f}°")
                if 'longitudes' in meta and i < len(meta['longitudes']):
                    print(f"  Longitude: {meta['longitudes'][i]:.2f}°")
                if 'pressures' in meta and i < len(meta['pressures']):
                    print(f"  Pressure: {meta['pressures'][i]:.2f} mb")
                if 'images' in f and 'data' in f['images']:
                    img = f['images/data'][i]
                    print(f"  Image shape: {img.shape}")
                    print(f"  Image value range: [{np.min(img):.2f}, {np.max(img):.2f}]")
                    print(f"  Image mean: {np.mean(img):.2f}")
        
        print("\n" + "=" * 80)

if __name__ == '__main__':
    print_hurricane_data()

