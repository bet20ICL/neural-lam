import xarray as xr
import zarr
import numcodecs

import os
import glob
import numpy as np
import torch

# Directory where the raw ERA5 data is stored
RAW_ERA5_PATH = "/vol/bitbucket/bet20/dataset/era5/global_full"

uk_bbox = {
    "lat_max": 63,
    "lat_min": 47,
    "lon_max": 4,
    "lon_min": -10,
}

def uk_subset(data):
    """
    data: xarray dataset
    Get UK local subset of data
    
    Dodgy hardcoding of lat/lon values due to wrap around
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(63, 47), longitude=slice(350, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(63, 47), longitude=slice(0, 4))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def save_dataset_samples(dataset, split="train", subset=None):
    """
    Convert ERA5 netcdf (.nc) files to numpy array files.
    Optionally take a subset of the ERA5 data.
    
    Each .nc file contains data for one month.
    We extract each time step from the data and save as a numpy array.
    
    Creates for ERA5 dataset:
        - samples/{split}/{date}.npy
    """
    nc_files = glob.glob(f'{RAW_ERA5_PATH}/*.nc')
    proccessed_dataset_path = f"data/{dataset}/samples/{split}"
    os.makedirs(proccessed_dataset_path, exist_ok=True)

    for j, filepath in enumerate(nc_files):
        data = xr.open_dataset(filepath)
        if subset:
            data = subset(data)
        
        for i, time in enumerate(data['time'].values):
            time = data['time'].values[i]
            sample = data.sel(time=time)
            array = sample.to_array().values # (N_vars, N_levels, N_lat, N_lon)
            array = array.transpose(3,2,0,1) # (N_lon, N_lat, N_vars, N_levels)
            N_lon, N_lat = array.shape[0], array.shape[1]
            array = array.reshape(N_lon*N_lat, -1) # (N_lon*N_lat, N_vars*N_levels)
                   
            time_py = time.astype('M8[ms]').tolist() # numpy.datetime64 -> datetime.datetime
            date_str = time_py.strftime('%Y%m%d%H%M%S') # datetime.datetime -> str
            
            np.save(f'{proccessed_dataset_path}/{date_str}.npy', array)
            print("Proccessed file: ", date_str)

def create_xy(dataset, subset=None):
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

    t_lon = torch.from_numpy(longitudes)
    t_lat = torch.from_numpy(latitudes)

    lon_lat_grid = torch.stack(
        torch.meshgrid(t_lon, t_lat, indexing="ij"), dim=0
    ) # (2, lon, lat) or (2, x, y)

    grid_array = lon_lat_grid.numpy()
    np.save(os.path.join(proccessed_dataset_path, "nwp_xy.npy"), grid_array)
    
    # Create border mask (is all zero for ERA5 dataset since there is no border)
    border_mask = np.zeros(grid_array.shape[1:], dtype=bool)
    np.save(os.path.join(proccessed_dataset_path, "border_mask.npy"), border_mask)

if __name__ == "__main__":
    save_dataset_samples("era5_uk", subset=uk_subset)
    # create_xy("era5_uk_full", subset=uk_subset)
