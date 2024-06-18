import xarray as xr
import zarr
import numcodecs

import os
import glob
import numpy as np
import torch

from neural_lam.constants import ERA5UKConstants

# Directory where the raw ERA5 data is stored
RAW_ERA5_PATH = ERA5UKConstants.RAW_ERA5_PATH

uk_bbox = {
    "lat_max": 63,
    "lat_min": 47,
    "lon_max": 4,
    "lon_min": -10,
}

def uk_max_subset(data):
    """
    [-20, 14, 38, 72]
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(72, 38), longitude=slice(360-20, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(72, 38), longitude=slice(0, 14))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def uk_big_subset(data):
    """
    data: xarray dataset
    Get a larger area around the UK, somewhat arbitrarily chosen to fit onto GPU memory
        
    [-13.0, 7.0, 45.0, 65.0]
    Dodgy hardcoding of lat/lon values due to wrap around
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(65, 45), longitude=slice(360-13, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(65, 45), longitude=slice(0, 7))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def uk_small_subset(data):
    """
    contains all the settlements (habited places) in the uk 
    [-8.00, 1.75, 49.75, 61.00]
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(61, 49.75), longitude=slice(360-8, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(61, 49.75), longitude=slice(0, 1.75))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def uk_subset(data):
    """
    data: xarray dataset
    Get UK local subset of data
    
    Dodgy hardcoding of lat/lon values due to wrap around
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(63, 47), longitude=slice(360-10, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(63, 47), longitude=slice(0, 4))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def proc_file(data, proccessed_dataset_path, subset=None, coarsen_fn=None):
    """
    Process a single .nc file into .npy files
    """
    if subset:
        data = subset(data)
    
    for i, time in enumerate(data['time'].values):
        time = data['time'].values[i]
        sample = data.sel(time=time)
        array = sample.to_array().values # (N_vars, N_levels, N_lat, N_lon)
        array = array.transpose(2,3,0,1) # (N_lat, N_lon, N_vars, N_levels)
        if coarsen_fn:
            array = coarsen_fn(array)
            
        N_lon, N_lat = array.shape[0], array.shape[1]
        array = array.reshape(N_lon*N_lat, -1) # (N_lon*N_lat, N_vars*N_levels)
                
        time_py = time.astype('M8[ms]').tolist() # numpy.datetime64 -> datetime.datetime
        date_str = time_py.strftime('%Y%m%d%H%M%S') # datetime.datetime -> str
        
        np.save(f'{proccessed_dataset_path}/{date_str}.npy', array)
        print("Proccessed file: ", date_str)

def save_dataset_samples(dataset, subset=None, coarsen_fn=None):
    """
    Convert ERA5 netcdf (.nc) files to numpy array files.
    Optionally take a subset of the ERA5 data.
    
    Each .nc file contains data for one month.
    We extract each time step from the data and save as a numpy array.
    
    Creates for ERA5 dataset:
        - samples/{split}/{date}.npy
    """
    # Training Files
    nc_files = []
    for year in ["2021", "2022"]:
        nc_files = nc_files + glob.glob(f'{RAW_ERA5_PATH}/{year}*.nc')
    nc_files.sort()
    proccessed_dataset_path = f"data/{dataset}/samples/train"
    # Train Files
    os.makedirs(proccessed_dataset_path, exist_ok=True)
    for j, filepath in enumerate(nc_files):
        data = xr.open_dataset(filepath)
        proc_file(data, proccessed_dataset_path, subset=subset, coarsen_fn=coarsen_fn)
    
    # Validation Files
    for month in ERA5UKConstants.VAL_MONTHS:
        filepath = f'{RAW_ERA5_PATH}/2023_{month}.nc'
        proccessed_dataset_path = f"data/{dataset}/samples/val/{month}"
        os.makedirs(proccessed_dataset_path, exist_ok=True)
        data = xr.open_dataset(filepath)
        proc_file(data, proccessed_dataset_path, subset=subset, coarsen_fn=coarsen_fn)    
        
    # Test Files
    for month in ERA5UKConstants.TEST_MONTHS:
        filepath = f'{RAW_ERA5_PATH}/2023_{month}.nc'
        proccessed_dataset_path = f"data/{dataset}/samples/val/{month}"
        os.makedirs(proccessed_dataset_path, exist_ok=True)
        data = xr.open_dataset(filepath)
        proc_file(data, proccessed_dataset_path, subset=subset, coarsen_fn=coarsen_fn)    

def create_xy(dataset, subset=None, coarsen_fn=None):
    """
    Creates for ERA5 dataset
        - nwp_xy.npy
        - border_mask.npy
    """
    nc_files = glob.glob(f'{RAW_ERA5_PATH}/*.nc')
    proccessed_dataset_path = f"data/{dataset}/static"
    os.makedirs(proccessed_dataset_path, exist_ok=True)
    
    # Use first data file to obtain lat/lon grid
    data = xr.open_dataset(nc_files[0])
    if subset:
        data = subset(data)

    longitudes = data.longitude.values
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    latitudes = data.latitude.values

    t_lat = torch.from_numpy(latitudes)
    t_lon = torch.from_numpy(longitudes)

    lat_grid, lon_grid = torch.meshgrid(t_lat, t_lon, indexing="ij") # both (lat, lon) / (y, x)
    grid_array = torch.stack(
        (lon_grid, lat_grid),
    ).numpy() # (2, y, x)

    if coarsen_fn:
        grid_array = coarsen_fn(grid_array)
        
    np.save(os.path.join(proccessed_dataset_path, "nwp_xy.npy"), grid_array)
    
    # Create border mask (is all zero for ERA5 dataset since there is no border)
    border_mask = np.zeros(grid_array.shape[1:], dtype=bool)
    np.save(os.path.join(proccessed_dataset_path, "border_mask.npy"), border_mask)
    
# TODO: refactor this shit
def _create_xy(data, dataset, subset=None):
    """
    Creates for ERA5 dataset
        - nwp_xy.npy
        - border_mask.npy
    """
    proccessed_dataset_path = f"data/{dataset}/static"
    os.makedirs(proccessed_dataset_path, exist_ok=True)
    
    if subset:
        data = subset(data)

    longitudes = data.longitude.values
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    latitudes = data.latitude.values

    t_lat = torch.from_numpy(latitudes)
    t_lon = torch.from_numpy(longitudes)

    lat_grid, lon_grid = torch.meshgrid(t_lat, t_lon, indexing="ij") # both (lat, lon) / (y, x)
    grid_array = torch.stack(
        (lon_grid, lat_grid),
    ).numpy() # (2, y, x)
    
    np.save(os.path.join(proccessed_dataset_path, "nwp_xy.npy"), grid_array)
    
    # Create border mask (is all zero for ERA5 dataset since there is no border)
    border_mask = np.zeros(grid_array.shape[1:], dtype=bool)
    np.save(os.path.join(proccessed_dataset_path, "border_mask.npy"), border_mask)
    
def _save_dataset_samples(dataset, subset=None):
    """
    Convert ERA5 netcdf (.nc) files to numpy array files.
    Optionally take a subset of the ERA5 data.
    
    Each .nc file contains data for one month.
    We extract each time step from the data and save as a numpy array.
    
    Creates for ERA5 dataset:
        - samples/{split}/{date}.npy
    """
    # Training Files
    nc_files = []
    for year in ["2021", "2022"]:
        nc_files = nc_files + glob.glob(f'{RAW_ERA5_PATH}/{year}*.nc')
    proccessed_dataset_path = f"data/{dataset}/samples/train"
    os.makedirs(proccessed_dataset_path, exist_ok=True)
    for j, filepath in enumerate(nc_files):
        data = open_file(filepath, coarsen=22)
        proc_file(data, proccessed_dataset_path, subset=subset)
        
    # Validation Files
    for month in ERA5UKConstants.VAL_MONTHS:
        filepath = f'{RAW_ERA5_PATH}/2023_{month}.nc'
        proccessed_dataset_path = f"data/{dataset}/samples/val/{month}"
        os.makedirs(proccessed_dataset_path, exist_ok=True)
        data = open_file(filepath, coarsen=22)
        proc_file(data, proccessed_dataset_path, subset=subset)   
        
def open_file(filepath, coarsen=None):
    data = xr.open_dataset(filepath, chunks={"time": 4})
    if coarsen:
        data = data.coarsen(longitude=coarsen, latitude=coarsen, boundary="pad").mean()
    return data

def make_global_dataset(coarsen=None):
    name = "era5_global"
    _save_dataset_samples(name)
    
    nc_files = glob.glob(f'{RAW_ERA5_PATH}/2022*.nc')
    data = open_file(nc_files[0], coarsen=coarsen)
    _create_xy(data, name)

if __name__ == "__main__":
    # make_global_dataset()
    
    # name = "era5_uk"
    # subset = uk_subset
    # save_dataset_samples(name, subset=subset)
    # create_xy(name, subset=subset)
    
    # name = "era5_uk_big_2_years"
    # subset = uk_big_subset
    # save_dataset_samples(name, subset=subset)
    # create_xy(name, subset=subset)
    
    name = "era5_uk_small"
    subset = uk_small_subset
    save_dataset_samples(name, subset=subset)
    create_xy(name, subset=subset)
    
    name = "era5_uk"
    subset = uk_subset
    save_dataset_samples(name, subset=subset)
    create_xy(name, subset=subset)
    
    name = "era5_uk_big"
    subset = uk_big_subset
    save_dataset_samples(name, subset=subset)
    create_xy(name, subset=subset)
    
    name = "era5_uk_max"
    subset = uk_max_subset
    save_dataset_samples(name, subset=subset)
    create_xy(name, subset=subset)