import torch
import math
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tarfile
from netCDF4 import Dataset
from pathlib import Path
from scipy.io import netcdf
import requests
import io

url1 = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/1978/HURSAT_b1_v06_1978151N15260_ALETTA_c20170721.tar.gz'
url2 = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2005/HURSAT_b1_v06_2005261N21290_RITA_c20170721.tar.gz'

"""  OLD METHOD  """
# filePath = 'data/raw/aletta.tar.gz'
# urllib.request.urlretrieve(url1, filePath)

# print('done')

# f = tarfile.open('data/raw/aletta.tar.gz')
# # f.extractall('./data/extracted/1978/')
# f.close()


# dir = Path.cwd()
# nc_path = 'data/extracted/1978/1978151N15260.ALETTA.1978.05.30.1500.34.GOE-2.033.hursat-b1.v06.nc'
# nc_dataset = Dataset(dir / nc_path)

# # for i in nc_dataset.variables.keys():
# #     print(i, nc_dataset.variables[i].shape)
# lats = nc_dataset.variables['lat']
# lons = nc_dataset.variables['lon']
# bt = nc_dataset.variables['IRWIN'][0]

# print(lats[:10])
# print(lons[:10])
# print(len(lats))
# print(len(lons))
# print(bt[:10])


# lons = [5, 6, 7]
# lats = [2, 3, 5]
# bt = [[4, 6, 7], [2, 9, 10], [1, 8, 11]]



"""  NEW METHOD  """
year = (url1.split('/')[-2], url1.split('/')[-1].split('_')[4])[0]
name = (url1.split('/')[-2], url1.split('/')[-1].split('_')[4])[1]

response = requests.get(url1)
response.raise_for_status()

file_in_memory = io.BytesIO(response.content)
with tarfile.open(fileobj=file_in_memory, mode='r:gz') as tar:
    nc_file = None
    for member in tar.getmembers():
        if member.name.endswith('.nc'):
            nc_file = member
            break
    
    if nc_file:
        nc_file_buffer = tar.extractfile(nc_file)
        nc_file_bytes = nc_file_buffer.read()

        with Dataset('in_memory.nc', mode='r', memory=nc_file_bytes) as ds:
            print("opened dataset")


            for i in ds.variables.keys():
                print(i, ds.variables[i][:])
            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:]
            bt = ds.variables['IRWIN'][0][:]

            plt.pcolormesh(lons, lats, bt, cmap='ocean')
            plt.colorbar()
            plt.title(name + ' ' + year + ' at time ' + str(ds.variables['NomDate'][0]) + ' ' + str(ds.variables['NomTime'][0]) + ' ' + str(ds.variables['WindSpd'][0]) + 'kts' )
            plt.show()





""" DISPLAY """
# plt.pcolormesh(lons, lats, bt, cmap='ocean')
# plt.colorbar()
# plt.show()







# tar_gz_path = 'data\HURSAT_b1_v06_2015270N27291_JOAQUIN_c20170721.tar.gz'
# extract_path = '\data\HURSAT_2015_JOAQUIN'

# os.makedirs(extract_path, exist_ok=True)

# with tarfile.open(tar_gz_path, "r:gz") as tar:
#     print('CONNECT')
#     tar.extractall(path=extract_path)

# print('done')

# ds = xr.open_dataset(file_path)