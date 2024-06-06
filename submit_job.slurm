#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
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

# 5706045
# python train_model.py --model gat_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5713102
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5713103
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1

# 5713104
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1 --train_loss_mask 1

# 5713108
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5713585
# test new graph loading doesnt break
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5713876
# test time resolution data doesnt break
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 1

# 5714254
# 2 step
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 2

# 5714255
# 4 step
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 4

# 5714112 
# 2 GPUs

# 5714117
# 4 GPUs

# 5714263
# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5714305
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 2 --epochs 2 --overfit 1

# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 2 --epochs 2 --overfit 1

# 5714388
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5714389
# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# multi gpu
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8
 
# 5716066
# python train_model.py --model graph_lam --dataset era5_uk_big_2_years --graph uk_big_ico --batch_size 8


# eval multimodel
# 5716190
python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --eval val --load ./saved_models/graph_lam-4x64-06_06_03-7248/min_val_loss.ckpt
