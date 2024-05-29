#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=bet20 # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_log/slurm20-%j.out

export PATH=/homes/bet20/miniconda3/bin:$PATH
export PATH=/usr/bin/latex:$PATH

source activate /vol/bitbucket/bet20/nlam
source /vol/cuda/11.7.1/setup.sh
# source /vol/cuda/12.0.0/setup.sh

which python

# wandb online
# python train_model.py --model gat_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --epochs 5
# python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 2
# python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4 --epochs 150
# python train_model.py --model graph_lam --dataset meps_example --graph multiscale --epochs 5 --batch_size 2
# python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 4
# python train_model.py --model stats_model --dataset era5_uk --graph uk_graphcast --batch_size 4 --no_forcing 1 --epochs 1

# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_graphcast --batch_size 8


# python train_model.py --model graph_lam --dataset era5_uk --graph uk_graphcast --batch_size 8
python train_model.py --model graph_lam --dataset era5_global --graph global_graphcast --batch_size 8