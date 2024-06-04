# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import xarray as xr

# First party
from era5_data_proc import uk_subset, uk_small_subset, uk_big_subset, open_file
from neural_lam.constants import ERA5UKConstants

def era5_static_features(grid_xy, dataset, coarsen=None):
    """Get static features for the grid nodes (surface geopotential and land sea mask)

    Args:
        grid_xy (_type_): _description_

    Returns:
        array: _description_
    """
    
    static_dataset_path = os.path.join(ERA5UKConstants.RAW_ERA5_PATH, "static_variables.nc")
    
    if dataset == "era5_global":
        static_data = open_file(static_dataset_path, coarsen=22)
    else:
        static_data = xr.open_dataset(static_dataset_path)
        if dataset == "era5_uk":
            subset = uk_subset
        elif dataset == "era5_uk_small":
            subset = uk_small_subset
        elif dataset in ["era5_uk_big", "era5_uk_big_coarse"]:
            subset = uk_big_subset
        static_data = subset(static_data)
        
    static_data = static_data.sel(time=static_data['time'].values[0]).to_array().values # (N_var, N_y, N_x)
    if coarsen:
        static_data = static_data[:, ::coarsen, ::coarsen]
    static_data = static_data.transpose(1, 2, 0).reshape(grid_xy.shape[0], -1) # (N_y * N_x, N_var)
    return static_data

def create_era5_grid_features(args, static_dir_path):
    """
    Create static features for the grid nodes
    """
    # -- Static grid node features --
    grid_xy = torch.tensor(
        np.load(os.path.join(static_dir_path, "nwp_xy.npy"))
    )  # (2, N_y, N_x)
    
    grid_xy = grid_xy.reshape(2, -1).T # (N_y * N_x, 2)
    grid_xy = np.radians(grid_xy)
    grid_lons = grid_xy[:, 0]
    grid_lats = grid_xy[:, 1]
    grid_features = torch.stack(
        (
            np.cos(grid_lats), 
            np.sin(grid_lons), 
            np.cos(grid_lons)
        ), 
        dim=1
    )
    torch.save(grid_features, os.path.join(static_dir_path, "grid_features_simple.pt"))
    
    static_data = era5_static_features(grid_xy, args.dataset, args.coarsen)
    grid_features = torch.cat(
        (grid_features, torch.tensor(static_data)), dim=1
    ) # (N_grid, N_var)
    grid_features = grid_features.to(torch.float)
    torch.save(grid_features, os.path.join(static_dir_path, "grid_features.pt"))

def main(args=None):
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--coarsen",
        type=int,
        default=None,
        help="Coarsen factor for the dataset (default: None - do not coarsen)",
    )
    if args:
        args = parser.parse_args(args)
    else:    
        args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")
    
    if "era5" in args.dataset:
        create_era5_grid_features(args, static_dir_path)
        return

    # -- Static grid node features --
    grid_xy = torch.tensor(
        np.load(os.path.join(static_dir_path, "nwp_xy.npy"))
    )  # (2, N_x, N_y)
    grid_xy = grid_xy.flatten(1, 2).T  # (N_grid, 2)
    pos_max = torch.max(torch.abs(grid_xy))
    grid_xy = grid_xy / pos_max  # Divide by maximum coordinate

    geopotential = torch.tensor(
        np.load(os.path.join(static_dir_path, "surface_geopotential.npy"))
    )  # (N_x, N_y)
    geopotential = geopotential.flatten(0, 1).unsqueeze(1)  # (N_grid,1)
    gp_min = torch.min(geopotential)
    gp_max = torch.max(geopotential)
    # Rescale geopotential to [0,1]
    geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_grid, 1)

    grid_border_mask = torch.tensor(
        np.load(os.path.join(static_dir_path, "border_mask.npy")),
        dtype=torch.int64,
    )  # (N_x, N_y)
    grid_border_mask = (
        grid_border_mask.flatten(0, 1).to(torch.float).unsqueeze(1)
    )  # (N_grid, 1)

    # Concatenate grid features
    grid_features = torch.cat(
        (grid_xy, geopotential, grid_border_mask), dim=1
    )  # (N_grid, 4)

    torch.save(grid_features, os.path.join(static_dir_path, "grid_features.pt"))


if __name__ == "__main__":
    main()
