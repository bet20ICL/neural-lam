"""Download ERA5 Data using Copernicus Data Store API

API request in the format:

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'geopotential',
        'pressure_level': [
            '150', '250',
        ],
        'year': [
            '1964', '1970',
        ],
        'month': [
            '01', '07',
        ],
        'day': [
            '01', '07',
        ],
        'time': [
            '00:00', '06:00',
        ],
    },
    'download.nc')
"""

import cdsapi
import xarray as xr
import zarr
import numcodecs
import sys
from pathlib import Path

# Directory where the raw ERA5 data is stored
RAW_ERA5_PATH = "/vol/bitbucket/bet20/dataset/era5"

c = cdsapi.Client()

region = 'global_full'
year = [2022, 2023]
month = [i for i in range(1, 13)]

year_str = [str(i) for i in year]
month_str = [str(i).zfill(2) for i in month]
day_str = [str(i).zfill(2) for i in range(1, 32)]
time_str = ['00:00', '06:00', '12:00', '18:00']

dir_path = f'{RAW_ERA5_PATH}/{region}'
path = Path(dir_path)
path.mkdir(parents=True, exist_ok=True)

def download_atmospheric_vars(y_str):
    """Download atmospheric variables for the specified year, months and times

    Args:
        y_str (_type_): _description_
    """
    for m in month_str:
        file_path = f'{dir_path}/{y_str}_{m}.nc'
        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        'geopotential', 'specific_humidity','temperature',
                        'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                    ],
                    'pressure_level': [
                        '50', '150', '250', '400', '500', '600', '850', '1000'
                    ],
                    'year': y_str,
                    'month': m,
                    'day': day_str,
                    'time': time_str,
                    # 'area': area,
                    'format': 'netcdf',
                },
                file_path
            )
        except Exception as e:
            print("=========== Error occured: ===========")
            print(e)

def download_surface_vars():
    """Download surface variables
    """
    file_path = f'{dir_path}/static_variables.nc'
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'land_sea_mask',
            ],
            'year': ['2022', '2023'],
            'month': [
                '03', '10',
            ],
            'day': [
                '11', '28',
            ],
            'time': ['09:00', '21:00'],
        },
        file_path
    )

for y in year_str:
    download_atmospheric_vars(y)

# download_surface_vars()