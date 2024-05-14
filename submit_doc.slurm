#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=bet20 # required to send email notifcations - please replace <your_username> with your college login name or email address

export PATH=/homes/bet20/miniconda3/bin:$PATH
export PATH=/usr/bin/latex:$PATH

source activate /vol/bitbucket/bet20/nlam
source /vol/cuda/11.7.1/setup.sh
# source /vol/cuda/12.0.0/setup.sh

# wandb online
# python train_model.py --graph multiscale --model graph_lam --batch_size 2
# python check_wandb.py
nvidia-smi
python train_model.py --model graph_lam --graph multiscale --epochs 5 --batch_size 2
# python test.py