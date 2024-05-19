# FYP Notes

## ERA5 Dataset

`era5_download_data.py`
- download era5 data from CDSAPI in zarr format

`era5_data_proc.py`
- process downloaded data

## Workflow

python era5_download_data.py

python era5_data_proc.npy

python create_mesh.py --dataset era5_uk --graph uk_graphcast

python create_grid_features.py --dataset era5_uk

python create_parameter_weights.py --dataset era5_uk --batch_size 2

python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --epochs 5 --batch_size 4


# Wednesday

# Sprint 1: Get ERA5 training

Graph generation verified

border_mask.npy verified

grid features
- verify nwp_xy.npy coordinates are right format (x)
- verify grid_features are made correctly (x)

era5_dataset
- verify order of features in each time step file (x)
    - its in the same order as the x array file
- make new files (x)
- complete the feature map (x)
- figure out validation split (x)


graph
- verify graph generation correct (x)

# Thursday

# Sprint 1:
- verify validation split (x)
- figure out mean and std normalisation (x)
- limit training to 1 step for now, eval on multiple steps (x)
- val subset 
- start training

# Friday

# Sprint 0
- read Keisler (x)

# Saturday

- read Stormer ()
- read climax ()

- read weatherbench2 ()
- read inductive biases ()
- read graph u-nets ()
- read graph pooling ()
- read multiscale gnns ()

# Sprint 1
- heatmap (x)
- sanity check against weatherbench 2 ()

# Sprint 2: Fill in TODOs about the dataset
- forcing features (x)
- parameter weights
    - pressure weights
        - figure this out (x)
        - implement (x)
    - latitude weights
        - read Stormer (x) 
        - read GraphCast (x)
        - check modulus ()
        - implement ()

# Sunday

# Sprint 1
- finish previous work 
- setup data processing script on cirrus
    - auto save .nc files in to .np files + train split ()

- investigate slow validation loader ()

# Sprint 3: baseline models
- gcn ()
- gat ()

# Sprint 4
- read aman's code ()
- space attention ()

# Sprint 5
- train larger model ()