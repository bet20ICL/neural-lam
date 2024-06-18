#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --account=ec249

# WandB
export WANDB_API_KEY=a969b84c10ffe6ee78383e7dbf3454ee494726d5 
source /work/ec249/ec249/bet20/myenv/bin/activate
which python

# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8
python train_multi_gpu_mre.py