import xarray as xr
import zarr
import numcodecs

import os
import glob
import numpy as np
import torch

from neural_lam.constants import ERA5UKConstants
import era5_data_proc
import create_grid_features
import create_mesh
from graph_utils import load_graph

def subsample(factor=3):
    """
    Coarsen a dataset by subsampling
    
    Expects array to either
    1) grid array with shape (2, y, x) 
    2) data array with shape (y, x, levels, vars)
    """
    def _subsample(array, factor=factor):
        if len(array.shape) == 3:
            return array[:, ::factor, ::factor]
        
        if len(array.shape) == 4:
            return array[::factor, ::factor]
        raise ValueError("Array shape not supported")
    
    return _subsample

def mean_coarsen(array):
    """
    Coarsen a dataset by taking the mean of each block
    """
    raise NotImplementedError("mean_coarsen not implemented")

def multi_res_edges(fine_graph, full_coarse_graph, coarse_graph):
    # edges from coarse mesh to fine mesh
    coarse2fine_edge_index = [[], []]

    for i, (u, v) in enumerate(full_coarse_graph.m2m_edge_index.T):
        u_uid = full_coarse_graph.mesh_uids[u]
        v_uid = full_coarse_graph.mesh_uids[v]
        if (
            u_uid in coarse_graph.mesh_uids and
            v_uid in fine_graph.mesh_uids
        ):
            coarse2fine_edge_index[0].append(coarse_graph.mesh_uids.index(u_uid))
            coarse2fine_edge_index[1].append(fine_graph.mesh_uids.index(v_uid))

    coarse2fine_edge_index = torch.tensor(coarse2fine_edge_index, dtype=torch.int64)
    coarse2fine_edge_set = {tuple(e) for e in coarse2fine_edge_index.T}
    return coarse2fine_edge_index, coarse2fine_edge_set
    
def create_border_mask():
    full_dataset = "era5_uk_big"
    local_dataset = "era5_uk_small"
    full_grid = np.load(f"./data/{full_dataset}/static/nwp_xy.npy")
    local_grid = np.load(f"./data/{local_dataset}/static/nwp_xy.npy")
    
    print("Full dataset shape:", full_grid.shape)
    print("Local dataset shape:", local_grid.shape)
    
    full_grid_2d = full_grid.reshape(2, -1).T
    local_grid_2d = local_grid.reshape(2, -1).T
    mask = np.isin(full_grid_2d, local_grid_2d).all(axis=1)
    mask = 1.0 - mask.reshape(full_grid.shape[1:]).astype(int)
    # np.save(f"./data/{full_dataset}/static/border_mask.npy", mask)
    np.save(f"./data/{full_dataset}/static/border_mask.npy", mask)
    
def uk_hierarchy():
    # Download the datsets
    small_dataset = "era5_uk_small"
    big_dataset = "era5_uk_big"
    coarse_big_dataset = f"{big_dataset}_coarse" 
    
    # subset = era5_data_proc.uk_small_subset
    # era5_data_proc.save_dataset_samples(small_dataset, subset=subset)
    # era5_data_proc.create_xy(small_dataset, subset=subset)
    # remember to also
    # create_grid_features
    # create_parameter_weights
    
    # subset = era5_data_proc.uk_big_subset
    # # era5_data_proc.save_dataset_samples(big_dataset, subset=subset) # don't need to save the full samples
    # era5_data_proc.create_xy(big_dataset, subset=subset) # just need this grid for graph generation
    # remember to also
    # create_grid_features
    # create_parameter_weights
    
    # subset = era5_data_proc.uk_big_subset
    # era5_data_proc.save_dataset_samples(coarse_big_dataset, subset=subset, coarsen_fn=subsample())
    # era5_data_proc.create_xy(coarse_big_dataset, subset=subset, coarsen_fn=subsample())
    # remember to also
    create_grid_features.main(
        ["--dataset", coarse_big_dataset, "--coarsen", "3"]
    )
    # create_parameter_weights
    
    # big_graph_name = "uk_big_ico"
    small_graph_name = "uk_small_ico"
    coarse_big_graph_name = "uk_big_coarse_ico"
    
    # # Generate the graphs
    # args = ["--dataset", small_dataset, "--graph", small_graph_name]
    # create_mesh.main(args)
    
    # args = ["--dataset", big_dataset, "--graph", big_graph_name]
    # create_mesh.main(args)
    
    args = ["--dataset", coarse_big_dataset, "--graph", coarse_big_graph_name, "--max_order", "5"]
    create_mesh.main(args)
    
    # # Create edges from uk_big_coarse_ico to uk_small_ico
    fine_graph = load_graph(small_dataset, "uk_small_ico")
    full_coarse_graph = load_graph(big_dataset, "uk_big_ico")
    coarse_graph = load_graph(coarse_big_dataset, "uk_big_coarse_ico")
    coarse2fine_edge_index, fine2coarse_edge_set = multi_res_edges(fine_graph, full_coarse_graph, coarse_graph)
    
    fine_graph_path = f"graphs/{small_graph_name}"
    torch.save(
        coarse2fine_edge_index, 
        os.path.join(fine_graph_path, "coarse2fine_edge_index.pt")
    )

def test_subsample():
    x = torch.randn(81, 81, 6, 8)
    y = subsample(x, 3)
    print(y.shape)

if __name__ == "__main__":
    uk_hierarchy()
    # create_border_mask()