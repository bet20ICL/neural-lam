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
- finish previous work (x)
- setup data processing script on cirrus
    - auto save .nc files in to .np files + train split (x)

- investigate slow validation loader (x)
    - AR steps take a long time

- investigate g2m and m2g indices (x)
    - its fucked up for no reason

- understand existing gnn completely ()
    - interaction net (x)
    - graph-lam (x)

# Sprint 2: baseline models
- gcn (x)

# Monday

# Communication
    - SEND
        - present gcn vs graphcast results (x)
        - find out how to calculate flops 

    - RECV
        - cirrus issues
        - sync tuesday/wednesday about space attention implementation

# Sprint 0
- analyse gcn vs graphcast (x)
- point cloud (x)
- grid features (x)
    - doc, 114955
- grid_features, forcing features (x)
    - doc, 114956
- set up sulis (kind of?)
- gat (x)
- read weatherbench2 (x) 
- sanity check against weatherbench 2 (x)

# Sprint 1
- understand graph transformers (_)

# Sprint 2
- understand diff pool (_)

# Sprint 3
- read aman's code (x)

# Sprint 4
- Stormer paper (_)
- ClimaX paper (_)
- ClimaX code (_)
- space attention (_)

# Sprint 4
- train larger model (_)

# Sprint 5
- set up training on cirrus (x)

- grid features, forcing (115039)

- grid features, no forcing (115042)


# Meeting
- Profiling
    - trainable parameters:
        - returned by PyTorch Lightning
    - inference flops:
        - using fvcore
        - graph_lam
            - unsupported operations:
                Unsupported operator aten::silu encountered 19 time(s)
                Unsupported operator aten::index_select encountered 12 time(s)
                Unsupported operator aten::expand_as encountered 6 time(s)
                Unsupported operator aten::scatter_add_ encountered 6 time(s)
                Unsupported operator aten::add encountered 13 time(s)
                Unsupported operator aten::mul encountered 3 time(s)

        
        - gcn_lam
            - gradient issues? 

# Thursday 

# Method Verification

## Alignment of Data Features
- `nwp_xy.npy`
- `samples/2022*.npy`
- 48 features are aligned in the order listed in `constants.ipynb` (x)

## Grid Feature Creation
- `grid_features.npy`
- grid features are being created as stated in graphcast (x)
    - BUG FOUND! 
    - using cos(lat), sin(lon), cos(lat) instead cos(lon)

## Forcing Features
- `era5_dataset.py`
- forcing features created as stated (x) 
    - BUG FOUND!
    - using i instead of idx!!

## Parameter Weights
- `create_parameter_weights.py`
- parameter weights created as stated (x)
    - seems reasonable...

## Mesh Graph
- `graphcast_mesh.py`
- `graphcast_utils.py`
- check locations of mesh nodes (x)
- check mesh connectivity
    - adjacent nodes connected (x)
        - its hard to see if higher order edges are present
        - if they are, they arent uniformly distributed across the mesh...
- check g2m connectivity (x)
    - grid node should connect to 
- check m2g connectivity (x)
    - each grid node is in the centre of a triangle, connect to the corners of the triangle


# Sprint 0
- finish verification above (_) 

# Sprint 1
- Graph Transformers (x)
    - come up with plan to implement (_)
- Graph U-Nets (_)

# Sprint 2
- try rectangular graph on era5-uk (_)

# Sprint 3
- space attention (_)


# TO ASK
- is loading each timestep from memory (~300 MBs) efficient? maybe memory bottleneck?

    - get runtime results between gat, gcn, graph-lam and compare
    - what else can i do in that case?

- hierarchical 


# Friday 

# Sprint 0
- Log times in WandB (x)

# Saturday
- Graph U-Nets (x)
- DiffPool (x)

## Data normalisation
- `era5_dataset.py`, `create_parameter_weights.py`
- global mean and var (_)
- step differences (x)

# Sunday
- Space Attention ()
- global mean and var (x)
- whats the error of predicting the mean ()

# Sprint 2
- try rectangular graph on era5-uk ()

# Sprint 3
- space attention ()