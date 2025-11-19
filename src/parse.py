import torch
import math
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tarfile
from netCDF4 import Dataset
from pathlib import Path
from scipy.io import netcdf
import requests
import io
import h5py
from tqdm import tqdm
from datetime import datetime
import time
from bs4 import BeautifulSoup

# Storm names by year (North Atlantic)
stormsByYear = {
    1978: ['AMELIA', 'BESS', 'CORA', 'DEBRA', 'ELLA', 'FLOSSIE', 'GRETA', 'HOPE', 'IRMA', 'JULIET', 'KENDRA'],
    1979: ['ANA', 'BOB', 'CLAUDETTE', 'DAVID', 'ELENA', 'FREDERIC', 'GLORIA', 'HENRI'],
    1980: ['ALLEN', 'BONNIE', 'CHARLEY', 'DANIELLE', 'EARL', 'FRANCES', 'GEORGES', 'HERMINE', 'IVAN', 'JEANNE', 'KARL'],
    1981: ['ARLENE', 'BRET', 'CINDY', 'DENNIS', 'EMILY', 'FLOYD', 'GERT', 'HARVEY', 'IRENE', 'JOSE', 'KATRINA', 'MARIA'],
    1982: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO'],
    1983: ['ALICIA'],
    # 1983: ['ALICIA', 'BARRY', 'CHANTAL', 'DEAN'],
    1984: ['ARTHUR', 'BERTHA', 'CESAR', 'DIANA', 'EDOUARD', 'FRAN', 'GUSTAV', 'HORTENSE', 'ISIDORE', 'JOSEPHINE', 'KLAUS', 'LILI', 'MARCO'],
    1985: ['ANA', 'BOB', 'CLAUDETTE', 'DANNY', 'ELENA', 'FABIAN', 'GLORIA', 'HENRI', 'ISABEL', 'JUAN', 'KATE'],
    1986: ['ANDREW', 'BONNIE', 'CHARLEY', 'DANIELLE', 'EARL', 'FRANCES'],
    1987: ['ARLENE', 'BRET', 'CINDY', 'DENNIS', 'EMILY', 'FLOYD', 'GERT'],
    1988: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO', 'FLORENCE', 'GILBERT', 'HELENE', 'ISAAC', 'JOAN', 'KEITH', 'LESLIE', 'NADINE'],
    1989: ['ALLISON', 'BARRY', 'CHANTAL', 'DEAN', 'ERIN', 'FELIX', 'GABRIELLE', 'HUGO', 'IRIS', 'JERRY', 'KAREN'],
    1990: ['ARTHUR', 'BERTHA', 'CESAR', 'DIANA', 'EDOUARD', 'FRAN', 'GUSTAV', 'HORTENSE', 'ISIDORE', 'JOSEPHINE', 'KLAUS', 'LILI', 'MARCO', 'NANA'],
    1991: ['ANA', 'BOB', 'CLAUDETTE', 'DANNY', 'ERIKA', 'FABIAN', 'GRACE', 'HENRI'],
    1992: ['ANDREW', 'BONNIE', 'CHARLEY', 'DANIELLE', 'EARL', 'FRANCES'],
    1993: ['ARLENE', 'BRET', 'CINDY', 'DENNIS', 'EMILY', 'FLOYD', 'GERT', 'HARVEY'],
    1994: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO', 'FLORENCE', 'GORDON'],
    1995: ['ALLISON', 'BARRY', 'CHANTAL', 'DEAN', 'ERIN', 'FELIX', 'GABRIELLE', 'HUMBERTO', 'IRIS', 'JERRY', 'KAREN', 'LUIS', 'MARILYN', 'NOEL', 'OPAL', 'PABLO', 'ROXANNE', 'SEBASTIEN', 'TANYA'],
    1996: ['ARTHUR', 'BERTHA', 'CESAR', 'DOLLY', 'EDOUARD', 'FRAN', 'GUSTAV', 'HORTENSE', 'ISIDORE', 'JOSEPHINE', 'KYLE', 'LILI', 'MARCO'],
    1997: ['ANA', 'BILL', 'CLAUDETTE', 'DANNY', 'ERIKA', 'FABIAN', 'GRACE', 'HENRI'],
    1998: ['ALEX', 'BONNIE', 'CHARLEY', 'DANIELLE', 'EARL', 'FRANCES', 'GEORGES', 'HERMINE', 'IVAN', 'JEANNE', 'KARL', 'LISA', 'MITCH', 'NICOLE'],
    1999: ['ARLENE', 'BRET', 'CINDY', 'DENNIS', 'EMILY', 'FLOYD', 'GERT', 'HARVEY', 'IRENE', 'JOSE', 'KATRINA', 'LENNY'],
    2000: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO', 'FLORENCE', 'GORDON', 'HELENE', 'ISAAC', 'JOYCE', 'KEITH', 'LESLIE', 'MICHAEL', 'NADINE'],
    2001: ['ALLISON', 'BARRY', 'CHANTAL', 'DEAN', 'ERIN', 'FELIX', 'GABRIELLE', 'HUMBERTO', 'IRIS', 'JERRY', 'KAREN', 'LORENZO', 'MICHELLE', 'NOEL', 'OLGA'],
    2002: ['ARTHUR', 'BERTHA', 'CRISTOBAL', 'DOLLY', 'EDOUARD', 'FAY', 'GUSTAV', 'HANNA', 'ISIDORE', 'JOSEPHINE', 'KYLE', 'LILI'],
    2003: ['ANA', 'BILL', 'CLAUDETTE', 'DANNY', 'ERIKA', 'FABIAN', 'GRACE', 'HENRI', 'ISABEL', 'JUAN', 'KATE', 'LARRY', 'MINDY', 'NICHOLAS', 'ODETTE', 'PETER'],
    2004: ['ALEX', 'BONNIE', 'CHARLEY', 'DANIELLE', 'EARL', 'FRANCES', 'GASTON', 'HERMINE', 'IVAN', 'JEANNE', 'KARL', 'LISA', 'MATTHEW', 'NICOLE', 'OTTO'],
    2005: ['ARLENE', 'BRET', 'CINDY', 'DENNIS', 'EMILY', 'FRANKLIN', 'GERT', 'HARVEY', 'IRENE', 'JOSE', 'KATRINA', 'LEE', 'MARIA', 'NATE', 'OPHELIA', 'PHILIPPE', 'RITA', 'STAN', 'TAMMY', 'VINCE', 'WILMA', 'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'EPSILON', 'ZETA'],
    2006: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO', 'FLORENCE', 'GORDON', 'HELENE', 'ISAAC'],
    2007: ['ANDREA', 'BARRY', 'CHANTAL', 'DEAN', 'ERIN', 'FELIX', 'GABRIELLE', 'HUMBERTO', 'INGRID', 'JERRY', 'KAREN', 'LORENZO', 'MELISSA', 'NOEL', 'OLGA'],
    2008: ['ARTHUR', 'BERTHA', 'CRISTOBAL', 'DOLLY', 'EDOUARD', 'FAY', 'GUSTAV', 'HANNA', 'IKE', 'JOSEPHINE', 'KYLE', 'LAURA', 'MARCO', 'NANA', 'OMAR', 'PALOMA'],
    2009: ['ANA', 'BILL', 'CLAUDETTE', 'DANNY', 'ERIKA', 'FRED', 'GRACE', 'HENRI', 'IDA'],
    2010: ['ALEX', 'BONNIE', 'COLIN', 'DANIELLE', 'EARL', 'FIONA', 'GASTON', 'HERMINE', 'IGOR', 'JULIA', 'KARL', 'LISA', 'MATTHEW', 'NICOLE', 'OTTO', 'PAULA', 'RICHARD', 'SHARY', 'TOMAS'],
    2011: ['ARLENE', 'BRET', 'CINDY', 'DON', 'EMILY', 'FRANKLIN', 'GERT', 'HARVEY', 'IRENE', 'JOSE', 'KATIA', 'LEE', 'MARIA', 'NATE', 'OPHELIA', 'PHILIPPE', 'RINA', 'SEAN'],
    2012: ['ALBERTO', 'BERYL', 'CHRIS', 'DEBBY', 'ERNESTO', 'FLORENCE', 'GORDON', 'HELENE', 'ISAAC', 'JOYCE', 'KIRK', 'LESLIE', 'MICHAEL', 'NADINE', 'OSCAR', 'PATTY', 'RAFAEL', 'SANDY', 'TONY'],
    2013: ['ANDREA', 'BARRY', 'CHANTAL', 'DORIAN', 'ERIN', 'FERNAND', 'GABRIELLE', 'HUMBERTO', 'INGRID', 'JERRY', 'KAREN', 'LORENZO', 'MELISSA'],
    2014: ['ARTHUR', 'BERTHA', 'CRISTOBAL', 'DOLLY', 'EDOUARD', 'FAY', 'GONZALO', 'HANNA', 'JOSEPHINE', 'KYLE'],
    2015: ['ANA', 'BILL', 'CLAUDETTE', 'DANNY', 'ERIKA', 'FRED', 'GRACE', 'HENRI', 'IDA', 'JOAQUIN', 'KATE']
}

# Configuration
BASE_URL = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06'
DB_PATH = 'hurricane_data.h5'
LOG_FILE = 'download_log.txt'

# Year range to process
YEAR_MIN = 1983
YEAR_MAX = 1983  # Change this to process more years


def log_message(message, also_print=True):
    """Log message to file and optionally print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')
    
    if also_print:
        print(log_entry)


def create_database(db_path, image_shape=(301, 301)):
    """Initialize HDF5 database with proper structure"""
    log_message(f"Creating database at {db_path}")
    
    with h5py.File(db_path, 'w') as f:
        # Create images group with extensible dataset
        img_group = f.create_group('images')
        img_group.create_dataset(
            'data',
            shape=(0, image_shape[0], image_shape[1]),
            maxshape=(None, image_shape[0], image_shape[1]),
            dtype='float32',
            chunks=(100, image_shape[0], image_shape[1]),
            compression='gzip',
            compression_opts=4
        )
        
        # Create metadata group
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('wind_speeds', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        meta_group.create_dataset('years', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
        meta_group.create_dataset('storm_names', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        meta_group.create_dataset('dates', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
        meta_group.create_dataset('times', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
        meta_group.create_dataset('latitudes', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        meta_group.create_dataset('longitudes', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        meta_group.create_dataset('pressures', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        
        # Store global attributes
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['total_samples'] = 0
        f.attrs['image_shape'] = image_shape
        f.attrs['description'] = 'HURSAT-B1 North Atlantic Hurricane Satellite Data'
    
    log_message("Database created successfully")


def find_hursat_urls(year):
    """
    Scrape NOAA directory listing for a given year and return URLs for storms in stormsByYear.
    
    Args:
        year: Year to scrape (e.g., 1985)
    
    Returns:
        List of tuples: (full_url, storm_name) for storms in stormsByYear[year]
    """
    if year not in stormsByYear:
        log_message(f"Year {year} not in stormsByYear dictionary")
        return []
    
    # Construct directory URL
    dir_url = f"{BASE_URL}/{year}/"
    
    try:
        log_message(f"Scraping directory: {dir_url}", also_print=False)
        response = requests.get(dir_url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links ending in .tar.gz
        all_links = soup.find_all('a', href=True)
        tar_gz_files = [link['href'] for link in all_links if link['href'].endswith('.tar.gz')]
        
        # Filter files that contain any of the target storm names
        matching_urls = []
        for filename in tar_gz_files:
            # Extract storm name from filename
            # Format: HURSAT_b1_v06_1985234N13258_STORMNAME_c20170721.tar.gz
            parts = filename.split('_')
            if len(parts) >= 5:
                # Storm name is typically the 5th part (index 4)
                storm_name = parts[4].upper()
                
                if storm_name in stormsByYear[year]:
                    full_url = dir_url + filename
                    matching_urls.append((full_url, storm_name))
        
        log_message(f"Found {len(matching_urls)} matching files for year {year}", also_print=False)
        return matching_urls
        
    except requests.exceptions.RequestException as e:
        log_message(f"Error scraping directory {dir_url}: {str(e)}")
        return []
    except Exception as e:
        log_message(f"Error parsing directory listing for {year}: {str(e)}")
        return []


def download_and_parse_storm(url, year, storm_name):
    """
    Download a HURSAT tar.gz file and extract all time steps
    Returns list of (image, metadata) tuples
    """
    try:
        log_message(f"Downloading: {storm_name} ({year})", also_print=False)
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        records = []
        file_in_memory = io.BytesIO(response.content)
        
        with tarfile.open(fileobj=file_in_memory, mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.nc'):
                    try:
                        nc_file_buffer = tar.extractfile(member)
                        nc_file_bytes = nc_file_buffer.read()
                        
                        with Dataset('in_memory.nc', mode='r', memory=nc_file_bytes) as ds:
                            try:
                                # Extract satellite imagery (IRWIN = infrared window channel)
                                bt = np.array(ds.variables['IRWIN'][0][:])
                                
                                # Skip if image is invalid (all zeros or NaNs)
                                if np.all(bt == 0) or np.all(np.isnan(bt)):
                                    continue
                                
                                # Check if all valid (non-NaN) values are within valid range [150, 340]
                                # Skip if any valid value is outside this range
                                valid_mask = ~np.isnan(bt)
                                if not np.any(valid_mask):
                                    # No valid values, skip this member
                                    continue
                                
                                # Get only valid (non-NaN) values
                                valid_values = bt[valid_mask]
                                
                                # Check if any valid value is outside the range [150, 340]
                                if np.any(valid_values < 150) or np.any(valid_values > 340):
                                    continue
                                
                                # Extract metadata
                                wind_speed = float(ds.variables['WindSpd'][0])
                                
                                metadata = {
                                    'year': year,
                                    'storm_name': storm_name,
                                    'date': int(ds.variables['NomDate'][0]),
                                    'time': int(ds.variables['NomTime'][0]),
                                    'wind_speed': wind_speed,
                                    'latitude': float(ds.variables['CentLat'][0]),
                                    'longitude': float(ds.variables['CentLon'][0]),
                                    'pressure': float(ds.variables['CentPrs'][0])
                                }
                                
                                # print(ds.variables['NomDate'][0], ds.variables['NomTime'][0], bt, wind_speed)
                                
                                records.append((bt, metadata))
                                
                            except Exception as e:
                                log_message(f"  Error processing NetCDF file: {str(e)}", also_print=False)
                                continue
                    
                    except Exception as e:
                        log_message(f"  Error processing NetCDF file {member.name}: {str(e)}", also_print=False)
                        continue
        
        log_message(f"  {storm_name} ({year}): {len(records)} images extracted", also_print=False)
        return records
    
    except requests.exceptions.RequestException as e:
        log_message(f"  Failed to download {storm_name} ({year}): {str(e)}")
        return []
    except Exception as e:
        log_message(f"  Error processing {storm_name} ({year}): {str(e)}")
        return []


def append_to_database(db_path, records):
    """Append new records to HDF5 database, replacing duplicates if they exist"""
    if not records:
        return
    
    with h5py.File(db_path, 'a') as f:
        # Get current size and existing metadata
        current_size = f['images/data'].shape[0]
        
        # Read existing metadata to check for duplicates
        existing_years = f['metadata/years'][:]
        existing_storm_names = f['metadata/storm_names'][:]
        existing_dates = f['metadata/dates'][:]
        existing_times = f['metadata/times'][:]
        
        # Helper function to decode string
        def _decode_string(value):
            if isinstance(value, bytes):
                return value.decode('utf-8')
            elif isinstance(value, np.bytes_):
                return value.decode('utf-8')
            return str(value)
        
        # Create a mapping of existing record keys to indices for fast lookup
        existing_key_to_idx = {}
        for i in range(current_size):
            storm_name = _decode_string(existing_storm_names[i])
            key = (int(existing_years[i]), storm_name, int(existing_dates[i]), int(existing_times[i]))
            existing_key_to_idx[key] = i
        
        # Track how many new records we'll actually add
        new_records_count = 0
        
        # Process each record - replace duplicates or mark for appending
        for image, metadata in records:
            # Create key for this record
            key = (metadata['year'], metadata['storm_name'], metadata['date'], metadata['time'])
            
            if key in existing_key_to_idx:
                # Replace the existing record at the found index
                idx = existing_key_to_idx[key]
                f['images/data'][idx] = image.astype('float32')
                f['metadata/wind_speeds'][idx] = metadata['wind_speed']
                f['metadata/years'][idx] = metadata['year']
                f['metadata/storm_names'][idx] = metadata['storm_name']
                f['metadata/dates'][idx] = metadata['date']
                f['metadata/times'][idx] = metadata['time']
                f['metadata/latitudes'][idx] = metadata['latitude']
                f['metadata/longitudes'][idx] = metadata['longitude']
                f['metadata/pressures'][idx] = metadata['pressure']
            else:
                # New record, will append
                new_records_count += 1
        
        # If we have new records to append, resize and add them
        if new_records_count > 0:
            new_size = current_size + new_records_count
            
            # Resize datasets
            f['images/data'].resize(new_size, axis=0)
            f['metadata/wind_speeds'].resize(new_size, axis=0)
            f['metadata/years'].resize(new_size, axis=0)
            f['metadata/storm_names'].resize(new_size, axis=0)
            f['metadata/dates'].resize(new_size, axis=0)
            f['metadata/times'].resize(new_size, axis=0)
            f['metadata/latitudes'].resize(new_size, axis=0)
            f['metadata/longitudes'].resize(new_size, axis=0)
            f['metadata/pressures'].resize(new_size, axis=0)
            
            # Append new records
            append_idx = current_size
            for image, metadata in records:
                key = (metadata['year'], metadata['storm_name'], metadata['date'], metadata['time'])
                if key not in existing_key_to_idx:
                    f['images/data'][append_idx] = image.astype('float32')
                    f['metadata/wind_speeds'][append_idx] = metadata['wind_speed']
                    f['metadata/years'][append_idx] = metadata['year']
                    f['metadata/storm_names'][append_idx] = metadata['storm_name']
                    f['metadata/dates'][append_idx] = metadata['date']
                    f['metadata/times'][append_idx] = metadata['time']
                    f['metadata/latitudes'][append_idx] = metadata['latitude']
                    f['metadata/longitudes'][append_idx] = metadata['longitude']
                    f['metadata/pressures'][append_idx] = metadata['pressure']
                    append_idx += 1
            
            # Update total count
            f.attrs['total_samples'] = new_size
        else:
            # All records were duplicates, no size change needed
            # But we still need to update the count in case it was wrong
            f.attrs['total_samples'] = current_size


def process_all_storms():
    """Main pipeline: iterate through all storms and build database"""
    # Create database if it doesn't exist
    if not os.path.exists(DB_PATH):
        create_database(DB_PATH)
    
    log_message(f"Starting data collection for years {YEAR_MIN}-{YEAR_MAX}")
    
    total_storms = 0
    total_images = 0
    failed_downloads = []
    
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        if year not in stormsByYear:
            continue
        
        log_message(f"\n=== Processing Year {year} ===")
        
        # Get all URLs for this year
        year_urls = find_hursat_urls(year)
        
        if not year_urls:
            log_message(f"No URLs found for year {year}")
            continue
        
        log_message(f"Found {len(year_urls)} storms for year {year}")
        year_images = 0
        
        # Process each URL
        for url, storm_name in tqdm(year_urls, desc=f"Year {year}"):
            # Download and parse
            records = download_and_parse_storm(url, year, storm_name)
            
            if records:
                # Append to database
                append_to_database(DB_PATH, records)
                year_images += len(records)
                total_images += len(records)
                log_message(f"  {storm_name} ({year}): {len(records)} images", also_print=False)
            else:
                failed_downloads.append((year, storm_name, url))
            
            total_storms += 1
            
            # Small delay to be nice to the server
            time.sleep(0.25)
        
        log_message(f"Year {year} complete: {year_images} images added")
    
    # Final statistics
    log_message(f"\n{'='*50}")
    log_message(f"Data collection complete!")
    log_message(f"Total storms processed: {total_storms}")
    log_message(f"Total images collected: {total_images}")
    log_message(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        log_message("\nFailed downloads:")
        for year, name, url in failed_downloads:
            log_message(f"  - {year} {name}")
    
    # Print database statistics
    with h5py.File(DB_PATH, 'r') as f:
        n_samples = f.attrs['total_samples']
        image_shape = f.attrs['image_shape']
        log_message(f"\nDatabase statistics:")
        log_message(f"  Total samples: {n_samples}")
        log_message(f"  Image shape: {image_shape}")
        log_message(f"  Database size: {os.path.getsize(DB_PATH) / 1024**3:.2f} GB")


def delete_database():
    """Delete the hurricane_data.h5 database file"""
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            log_message(f"Successfully deleted {DB_PATH}")
        except Exception as e:
            log_message(f"Error deleting {DB_PATH}: {str(e)}")
    else:
        log_message(f"Database {DB_PATH} does not exist")


def test_single_url():
    """Test with a single known URL"""
    log_message("Testing with single URL")
    
    # Test URLs from your original code
    test_url = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/1978/HURSAT_b1_v06_1978151N15260_ALETTA_c20170721.tar.gz'
    
    # Create database
    if not os.path.exists(DB_PATH):
        create_database(DB_PATH)
    
    # Download and parse
    records = download_and_parse_storm(test_url, 1978, 'ALETTA')
    
    if records:
        log_message(f"Successfully extracted {len(records)} images")
        append_to_database(DB_PATH, records)
        log_message("Data saved to database")
        
        # Verify
        with h5py.File(DB_PATH, 'r') as f:
            print(f"\nDatabase contents:")
            print(f"  Total samples: {f.attrs['total_samples']}")
            print(f"  Image shape: {f['images/data'].shape}")
            print(f"  Wind speeds: {f['metadata/wind_speeds'][:]}")
    else:
        log_message("No records extracted")


if __name__ == '__main__':
    # Choose mode:
    # 1. Test with single URL first
    # 2. Then process all storms
    # 3. Delete database
    
    print("Choose mode:")
    print("1. Test with single URL")
    print("2. Process all storms (years {}-{})".format(YEAR_MIN, YEAR_MAX))
    print("3. Delete hurricane_data.h5")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        test_single_url()
    elif choice == '2':
        process_all_storms()
    elif choice == '3':
        delete_database()
    else:
        print("Invalid choice")