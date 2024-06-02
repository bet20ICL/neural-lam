#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=ec249

# texlive
export PATH=/work/ec249/ec249/bet20/texlive//bin/x86_64-linux:$PATH

# WandB
export WANDB_API_KEY=a969b84c10ffe6ee78383e7dbf3454ee494726d5

# # Required for running jobs on GPU node
# export WANDB_CONFIG_DIR="/work/ec249/ec249/bet20/.config/wandb"

# # Matplotlib
# export MPLCONFIGDIR="/work/ec249/ec249/bet20/.config/matplotlib"

# Load the required modules 
source /work/ec249/ec249/bet20/myenv/bin/activate
which python
# nvcc --version

# GPU job does not have access to internet,
# must turn WandB offline for script to run
# wandb online
# srun python train_model.py --model graph_lam --graph multiscale --epochs 10
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --simple_grid 1 --no_forcing 1 --simple_param_weights 1 --epochs 100
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --epochs 150 --batch_size 4

# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --simple_grid 1 --no_forcing 1 --simple_param_weights 1
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --simple_grid 0 --no_forcing 1 --simple_param_weights 1
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --simple_grid 1 --no_forcing 0 --simple_param_weights 1
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --simple_grid 1 --no_forcing 1 --simple_param_weights 0
# srun python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 16 --simple_grid 0 --no_forcing 0 --simple_param_weights 0
# srun python train_model.py --model gcn_lam --dataset era5_uk --graph uk_graphcast --batch_size 16 --simple_grid 0 --no_forcing 0 --simple_param_weights 0
# srun python train_model.py --model gat_lam --dataset era5_uk --graph uk_graphcast --batch_size 8 --simple_grid 0 --no_forcing 0 --simple_param_weights 0

# 5705571
# python train_model.py --model graph_lam --dataset era5_uk --graph uk_ico --batch_size 8

# 5705588
# python train_model.py --model graph_lam --dataset era5_uk_small --graph uk_small_ico --batch_size 8

# 5705590
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5705962
# python train_model.py --model graph_lam --dataset era5_uk --graph uk_ico --batch_size 8

# 5705963
# python train_model.py --model graph_lam --dataset era5_uk_small --graph uk_small_ico --batch_size 8

# 5705964
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8