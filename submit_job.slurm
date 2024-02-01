#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=ec249

# texlive
export PATH=/work/ec249/ec249/bet20/texlive//bin/x86_64-linux:$PATH

# Load the required modules 
source /work/ec249/ec249/bet20/nlam/bin/activate

# srun python train_model.py --model gat
srun python train_model.py --model graph_lam --dataset era5_uk_reduced --graph graphcast --epochs 5 --batch_size 1
# srun python era5_data_proc.py