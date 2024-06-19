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
export WANDB__SERVICE_WAIT=300

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

# ==================================================
# Logging 

# Logging every step 1 - 12
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1
# 5741963
# ./wandb/offline-run-20240619_020644-a5epmugh

# python train_model.py --model graph_lam --dataset era5_uk_big_2_years --graph uk_big_ico --batch_size 8 --val_loss_mask 1
# 5741964
# ./wandb/offline-run-20240619_020644-jksub7ka

# ==================================================
# Baselines
# python train_model.py --model graph_lam --dataset era5_uk_small --graph uk_small_ico --batch_size 8
# 5741965
# ./wandb/offline-run-20240619_020702-9sbgz5sj
# saved_models/graph_lam-4x64-06_19_02-9238


# python train_model.py --model graph_lam --dataset era5_uk --graph uk_ico --batch_size 8 --val_loss_mask 1
# 5746322 

# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1
# 5741966
# ./wandb/offline-run-20240619_020703-06yyak4y
# 5745313
#
# 5745478
#

# This is training on 2 years of data due to a bug
# python train_model.py --model graph_lam --dataset era5_uk_max --graph uk_max_ico --batch_size 8 --val_loss_mask 1
# 5741967
# ./wandb/offline-run-20240619_021935-h992xeqw

# python train_model.py --model graph_lam --dataset era5_uk_max --graph uk_max_ico --batch_size 8 --val_loss_mask 1
# 5741969
# ./wandb/offline-run-20240619_022914-bwqvyvjw

# ==================================================
# Simple GNN Baselines

# python train_model.py --model gcn_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1
# 5745085
# 

# python train_model.py --model gat_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1
# 5741971
# ./wandb/offline-run-20240619_023354-jy5fasuv


# ==================================================
# Coarse Graph Baselines

# python train_model.py --model graph_lam --dataset era5_uk_big_coarse --graph uk_big_coarse_ico --batch_size 8 --val_loss_mask 1
# 5741975
# ./wandb/offline-run-20240619_033833-8jv5amvs

# python train_model.py --model graph_lam --dataset era5_uk_max_coarse --graph uk_max_coarse_ico --batch_size 8 --val_loss_mask 1
# 5741978
# ./wandb/offline-run-20240619_041516-ewk89ws6

# ==================================================
# Space Attention Ablation

# Attention Layer Before Mesh Proc
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --attention_first 1 --layer_norm 1 --mesh_residual 1
# 5741984
# ./wandb/offline-run-20240619_041518-lvhrx4jz
# 5745437
# 

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --attention_first 1 --layer_norm 1
# 5741985
# ./wandb/offline-run-20240619_041933-gufvyrg3
# 5745439
# 

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --attention_first 1
# 5741986
# ./wandb/offline-run-20240619_041933-qirxp2pa

# Attention Layer After Mesh Proc
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --mesh_residual 1
# 5741987
# ./wandb/offline-run-20240619_043014-1twluq8u
# 5745443

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1
# 5741988
# ./wandb/offline-run-20240619_043014-2eoqn4jw

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8
# 5741989
# ./wandb/offline-run-20240619_043128-0a83ggj5

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --mesh_residual 1
# 5741990
# ./wandb/offline-run-20240619_043314-a6bj47uj
# 5745444

# ==================================================
# Space Attention V2

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --attention_first 1 --layer_norm 1 --mesh_residual 1
# 5746089

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --layer_norm 1 --mesh_residual 1
# 5746090

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --mesh_residual 1
# 5746098

# ==================================================
# Space Attention V3

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --attention_first 1 --layer_norm 1 --mesh_residual 1
# 5746099

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --layer_norm 1 --mesh_residual 1
# 5746101

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --mesh_residual 1
# 5746110