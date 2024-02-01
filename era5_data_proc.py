import xarray as xr
import zarr
import numcodecs
import constants

import glob
import numpy as np

def save_dataset_samples():
    """
    converts nc files to npy
    just run this once
    """
    nc_files = glob.glob(f'{constants.DATASET_PATH}/global/*.nc')
    proccessed_dataset_path = "data/era5_uk_reduced/samples/train"

    for j, filepath in enumerate(nc_files):
        data = xr.open_dataset(filepath)
        
        for i, time in enumerate(data['time'].values):
            time = data['time'].values[i]
            sample = data.sel(time=time)
            array = sample.to_array().values # (n_vars, n_levels, n_lat, n_lon)
            time_py = time.astype('M8[ms]').tolist() # numpy.datetime64 -> datetime.datetime
            date_str = time_py.strftime('%Y%m%d%H%M%S') # datetime.datetime -> str
            
            np.save(f'{proccessed_dataset_path}/{date_str}.npy', array)
            print("Proccessed file: ", date_str)

def create_era5_stats():
    name = "2022_02"
    filepath = f'{constants.DATASET_PATH}/global/{name}.nc'
    data = xr.open_dataset(filepath)
    era5_global_mean = data.mean(dim=("time", "latitude", "longitude")).values
    np.save(f'{constants.DATASET_PATH}/global/{name}_mean.npy', era5_global_mean)
    print("Saved global mean at ", f'{constants.DATASET_PATH}/global/{name}_mean.npy')

def create_era5_grid_features(args):
    pass

# create_era5_stats()
save_dataset_samples()