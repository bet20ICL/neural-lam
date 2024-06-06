#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
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
python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8
 
# 5716033