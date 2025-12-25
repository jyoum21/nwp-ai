# etl.py

import os
import io
import time
import h5py
import requests
import tarfile
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from netCDF4 import Dataset
from tqdm import tqdm

# Storm names by year (North Atlantic)
STORMS_BY_YEAR = {
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

# config
BASE_URL = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06'
DB_PATH = 'data/hurricane_data.h5'
LOG_FILE = 'download_log.txt'

# HurricaneDataBuilder: scrapes NOAA data and builds database
class HurricaneDataBuilder:

    # init
    def __init__(self, db_path, base_url, image_shape=(301, 301)):
        self.db_path = db_path
        self.base_url = base_url
        self.image_shape = image_shape
        self.log_file = 'download_log.txt'

    # log messages
    def log(self, message, print_msg=True):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{timestamp}] {message}"
        with open(self.log_file, 'a') as f:
            f.write(entry + '\n')
        if print_msg:
            print(entry)

    # create h5 file
    def _initialize_db(self):
        
        # overwrite file if exists
        if os.path.exists(self.db_path):
            self.log(f"Overwriting existing database at {self.db_path}")
            os.remove(self.db_path)
        else:
            self.log(f"Creating database at {self.db_path}")
        
        with h5py.File(self.db_path, 'w') as f:
            # initialize images
            img_grp = f.create_group('images')
            img_grp.create_dataset('data', shape=(0, *self.image_shape), 
                                 maxshape=(None, *self.image_shape),
                                 dtype='float32', compression='gzip', chunks=(10, *self.image_shape))
            
            # initialize metadata
            meta_grp = f.create_group('metadata')
            for key, dtype in [('wind_speeds', 'float32'), ('years', 'int32'), 
                               ('dates', 'int32'), ('times', 'int32'),
                               ('latitudes', 'float32'), ('longitudes', 'float32'), 
                               ('pressures', 'float32')]:
                meta_grp.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)
            
            # initialize storm names
            meta_grp.create_dataset('storm_names', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
            
            # attribute for total number of samples
            f.attrs['total_samples'] = 0

    # scrape online HURSAT data
    def _find_urls(self, year):
        
        # only process the years that we want
        if year not in STORMS_BY_YEAR: return []
        
        # get directory url for the year
        dir_url = f"{self.base_url}/{year}/"
        try:
            # fetch webpage and initialize BeautifulSoup
            resp = requests.get(dir_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # find all tar.gz links
            links = [l['href'] for l in soup.find_all('a', href=True) if l['href'].endswith('.tar.gz')]
            matching = []
            
            # check if link is a storm name we're looking for
            for filename in links:
                parts = filename.split('_')
                if len(parts) >= 5:
                    name = parts[4].upper()
                    if name in STORMS_BY_YEAR[year]:
                        matching.append((dir_url + filename, name))
            return matching # list of tuples (url, storm name)

        except Exception as e:
            self.log(f"Error scraping {year}: {e}")
            return []

    # download and extract valid images from a single storm URL
    def _process_storm(self, url, year, storm_name):
        try:
            # download the tar.gz file
            resp = requests.get(url, timeout=60)
            file_obj = io.BytesIO(resp.content)
            records = []
            analyzed_count = 0 # number of timesteps analyzed

            with tarfile.open(fileobj=file_obj, mode='r:gz') as tar:
                for member in tar.getmembers(): # each member is a file in the tar.gz file
                    # check if file is a netCDF file
                    if not member.name.endswith('.nc'): continue
                    
                    f_bytes = tar.extractfile(member).read()
                    with Dataset('mem', mode='r', memory=f_bytes) as ds:
                        analyzed_count += 1  # count every timestep analyzed
                        
                        # ensure that data is valid (150K - 340K temps only)
                        bt = np.array(ds.variables['IRWIN'][0][:])
                        if np.isnan(bt).any() or (bt < 150).any() or (bt > 340).any():
                            continue
                        
                        meta = {
                            'year': year, 'storm_name': storm_name,
                            'date': int(ds.variables['NomDate'][0]),
                            'time': int(ds.variables['NomTime'][0]),
                            'wind_speed': float(ds.variables['WindSpd'][0]),
                            'lat': float(ds.variables['CentLat'][0]),
                            'lon': float(ds.variables['CentLon'][0]),
                            'pressure': float(ds.variables['CentPrs'][0])
                        }
                        records.append((bt, meta)) # add image and metadata to the records list
            return records, analyzed_count

        except Exception as e:
            self.log(f"Failed {storm_name}: {e}", print_msg=False)
            return [], 0

    # save a batch of records to the h5 file
    def _save_batch(self, records):
        if not records: return
        
        # append the records to the h5 file
        with h5py.File(self.db_path, 'a') as f:
            n_current = f.attrs['total_samples']
            n_new = len(records)
            new_total = n_current + n_new

            # resize datasets
            f['images/data'].resize(new_total, axis=0) 
            for k in ['wind_speeds', 'years', 'storm_names', 'dates', 'times', 'latitudes', 'longitudes', 'pressures']:
                f[f'metadata/{k}'].resize(new_total, axis=0)

            # write data
            for i, (img, meta) in enumerate(records):
                idx = n_current + i
                f['images/data'][idx] = img
                f['metadata/wind_speeds'][idx] = meta['wind_speed']
                f['metadata/years'][idx] = meta['year']
                f['metadata/storm_names'][idx] = meta['storm_name']
                f['metadata/dates'][idx] = meta['date']
                f['metadata/times'][idx] = meta['time']
                f['metadata/latitudes'][idx] = meta['lat']
                f['metadata/longitudes'][idx] = meta['lon']
                f['metadata/pressures'][idx] = meta['pressure']
            
            # update total samples
            f.attrs['total_samples'] = new_total

    # main execution method
    def run(self, start_year, end_year):

        # initialize the database
        self._initialize_db()
        
        # note storms added and analyzed timesteps
        storms_added = {}  # {(year, storm_name): {'added': count, 'analyzed': count}}
        for year in range(start_year, end_year + 1):
            self.log(f"Processing {year}...")
            urls = self._find_urls(year) # get the urls for the year
            for url, name in tqdm(urls, desc=f"Year {year}"):
                data, analyzed_count = self._process_storm(url, year, name) # process each storm in the year
                storm_key = (year, name)

                # storm has valid timesteps
                if data:
                    self._save_batch(data)
                    storms_added[storm_key] = {
                        'added': len(data), # saved timesteps
                        'analyzed': analyzed_count # analyzed timesteps
                    }
                    self.log(f"Saved {len(data)} images for {name}", print_msg=False)

                # storm was analyzed but no valid timestep
                elif analyzed_count > 0:
                    storms_added[storm_key] = {
                        'added': 0,
                        'analyzed': analyzed_count
                    }
                time.sleep(0.1) # be nice to noaa's servers
        
        self._print_summary(storms_added)
    
    # print summary
    def _print_summary(self, storms_added):
        print("\n" + "=" * 60)
        print("ETL processing summary")
        print("=" * 60)
        
        if not storms_added:
            print("No storms were processed")
            return
        
        # filter to only storms that had valid timesteps
        storms_with_data = {k: v for k, v in storms_added.items() if v['added'] > 0}
        
        if storms_with_data:
            print(f"\nTotal storms added to data/hurricane_data.h5: {len(storms_with_data)}")
            total_added = sum(v['added'] for v in storms_with_data.values())
            total_analyzed = sum(v['analyzed'] for v in storms_with_data.values())
            print(f"Total timesteps added: {total_added}")
            print(f"Total timesteps analyzed: {total_analyzed}")
            print("\nStorms added to data/hurricane_data.h5:")
            print("-" * 60)
            
            # sort by year, then by storm name
            sorted_storms = sorted(storms_with_data.items(), key=lambda x: (x[0][0], x[0][1]))
            
            for (year, storm_name), counts in sorted_storms:
                print(f"  {storm_name} ({year}): {counts['added']} timesteps added (out of {counts['analyzed']} analyzed)")
            
            print("=" * 60)
        else:
            print("\nNo storms were added to the database (all timesteps failed validation).")
            print("=" * 60)

if __name__ == "__main__":
    builder = HurricaneDataBuilder(
        db_path='data/hurricane_data.h5',
        base_url='https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06'
    )

    # years to process
    YEAR_MIN = 1983
    YEAR_MAX = 1983
    builder.run(YEAR_MIN, YEAR_MAX)
