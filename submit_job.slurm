#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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
# ./wandb/offline-run-20240605_013508-go0bmewj
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5713585
# ./wandb/offline-run-20240605_120353-xq0j5u29
# test new graph loading doesnt break
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5713878
# test time resolution data doesnt break
# run name missing
# ./wandb/offline-run-20240605_141206-rrlbvxku
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 1

# 5714254
# run name missing
# ./wandb/offline-run-20240605_200317-twt1dqiq
# BUGGED - step length not implemented correctly
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 2

# 5714255
# 4 step
# ./wandb/offline-run-20240605_200128-kr3r537b
# BUGGED - step length not implemented correctly
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 4

# 5714112 
# 2 GPUs

# 5714117
# 4 GPUs

# 5714263 - failed
# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5714305
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 2 --epochs 2 --overfit 1

# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 2 --epochs 2 --overfit 1

# 5714388
# ./wandb/offline-run-20240606_030547-whycevkt
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5714389
# ./wandb/offline-run-20240606_030544-gg6100hv
# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# multi gpu
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8
 
# 5716066
# ./wandb/offline-run-20240606_180710-wm9fg0qa
# python train_model.py --model graph_lam --dataset era5_uk_big_2_years --graph uk_big_ico --batch_size 8

# ====================================================================================================
# recording run names in output after this point

# eval multimodel
# 5716190
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --eval val --load ./saved_models/graph_lam-4x64-06_06_03-7248/min_val_loss.ckpt

# 5716248
# 5716268
# longer rollouts
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --ar_steps 12

# 5716269
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 2 --epochs 2 --overfit 1

# 5716279
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --ar_steps 8

# BIG DATASET!
# 5716301
# graph_lam-4x64-06_07_02-4091
# ./wandb/offline-run-20240607_020103-v2sg0gol
# python train_model.py --model graph_lam --dataset era5_uk_max --graph uk_max_ico --batch_size 8

# 
# uk_big_small with nearest neighbour coarse 2 fine edges

# 5716311 
# graph_lam-4x64-06_07_03-8434
# ./wandb/offline-run-20240607_032059-w4yi6ek4
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5716308 
# graph_lam-4x64-06_07_03-0837
# ./wandb/offline-run-20240607_032004-jsphwhvf
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8

# 5716318 
# ./wandb/offline-run-20240607_033537-ghutnlwk
# python train_multi_model.py --model graph_lam --dataset era5_uk_max_small --batch_size 8

# 5716319 
# ./wandb/offline-run-20240607_033812-4uiaele1
# python train_multi_model.py --model graph_lam --dataset era5_uk_max_small_v2 --batch_size 8

# 5717247
# test dataset refactor didnt break anything
# ./wandb/offline-run-20240607_233313-15gl53gn
# graph_lam-4x64-06_07_23-8758
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 1

# 5717254
# test dataset refactor allows more training data for larger step length
# ./wandb/offline-run-20240608_015115-cn21mjx7
# graph_lam-4x64-06_08_01-1108
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 2

# 5717255
# test dataset refactor allows more training data for larger step length
# ./wandb/offline-run-20240608_020323-q8pr3ftl
# graph_lam-4x64-06_08_02-1493
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 4

# 5719297
# graph_lam-4x64-06_09_20-8435
# ./wandb/offline-run-20240609_204525-kunytwlq
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5719299
# graph_lam-4x64-06_09_20-7634
# ./wandb/offline-run-20240609_204834-rr5q4bnp
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5719300
# multi_time_model-4x64-06_09_20-1826
# ./wandb/offline-run-20240609_205759-29hhe3jz
# python train_model.py --model multi_time_model --time_resolution_levels 3 --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5719301
# graph_lam-4x64-06_09_20-4750
# ./wandb/offline-run-20240609_205859-e18iuu7m
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8

# 5719302
# graph_lam-4x64-06_09_21-7214
# ./wandb/offline-run-20240609_210158-0wr7isjf 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --mesh_residual 1

# 5719303
# graph_lam-4x64-06_09_21-5611
# ./wandb/offline-run-20240609_210158-cqoyello
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --attention_first 1

# 5719305
# graph_lam-4x64-06_09_21-8544
# ./wandb/offline-run-20240609_210529-b705dyvm
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v2 --batch_size 8 --mesh_residual 1 --attention_first 1

# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --eval test --val_loss_mask 1 --load ./saved_models/graph_lam-4x64-06_07_23-8758/min_val_loss.ckpt

# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --eval test --val_loss_mask 0 --load ./saved_models/graph_lam-4x64-06_07_23-8758/min_val_loss.ckpt

# 5719642
# graph_lam-4x64-06_10_01-9467
# ./wandb/offline-run-20240610_014854-vb4tknu7
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5719644
# ./wandb/offline-run-20240610_015714-pmhi0zl2
# graph_lam-4x64-06_10_01-6036
# python train_multi_dataset.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5718073
# ./wandb/offline-run-20240608_200314-j3hmhi2u
# graph_lam-4x64-06_08_20-8007
# python train_model.py --model graph_lam --dataset era5_uk_big_coarse --graph uk_big_coarse_ico --batch_size 8

# ====================================================
# Min val loss used after this point - bad!
# MULTI MODELS EXEMPT!

# 5719645 
# ./wandb/offline-run-20240610_022656-189fmoyz
# graph_lam-4x64-06_10_02-0509
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5719646
# ./wandb/offline-run-20240610_022856-1g5wx1zk
# graph_lam-4x64-06_10_02-6025
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --mesh_residual 1

# 5719647
# ./wandb/offline-run-20240610_022857-kvqeou88
# graph_lam-4x64-06_10_02-0384
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --attention_first 1

# 5719648 
# ./wandb/offline-run-20240610_022857-wpgrdxu4
# graph_lam-4x64-06_10_02-4064
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --mesh_residual 1 --attention_first 1

# 5719650
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# Test loss rollout 
# 5723573
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5723574 
# ./wandb/offline-run-20240610_210349-ns5p8ag5
# graph_lam-4x64-06_10_21-9503
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --ar_steps 2

# 5723575 
# ./wandb/offline-run-20240610_210402-esk7osdk
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --ar_steps 4

# Time resolution baselines
# 5723576
# graph_lam-4x64-06_10_21-3569
# ./wandb/offline-run-20240610_210625-na2wwhtx
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 1

# 5723577
# failed
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 2

# 5723578
# failed
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 4

# Test Space resolution with layer norm
# 5723599
# ./wandb/offline-run-20240610_210719-rk5sqmup
# graph_lam-4x64-06_10_21-6010
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 5723636
# ./wandb/offline-run-20240610_213742-1m39szm3
# graph_lam-4x64-06_10_21-3968
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1

# 5723641
# ./wandb/offline-run-20240610_214003-awzpj563
# graph_lam-4x64-06_10_21-8449
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --mesh_residual 1

# 5723642
# wandb sync ./wandb/offline-run-20240610_214011-w3dgcdje
# graph_lam-4x64-06_10_21-7700
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --attention_first 1

# 5723643
# wandb sync ./wandb/offline-run-20240610_214041-yjbcublz
# graph_lam-4x64-06_10_21-5209
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --mesh_residual 1 --attention_first 1

# Time resolution models
# 5723604
# ./wandb/offline-run-20240610_210848-jhhfryu8
# multi_time_model-4x64-06_10_21-3412
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5723644
# multi_time_model-4x64-06_10_21-5222
# ./wandb/offline-run-20240610_214041-ipju7s1r
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --layer_norm 1

# 5723645
# multi_time_model-4x64-06_10_21-2692
# ./wandb/offline-run-20240610_214051-to4r93hh
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --layer_norm 1 --mesh_residual 1

# 5723646
# multi_time_model-4x64-06_10_21-7416
# ./wandb/offline-run-20240610_214101-kfw3q5rm
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --layer_norm 1 --attention_first 1

# 5723647
# multi_time_model-4x64-06_10_21-2429
# ./wandb/offline-run-20240610_214112-9pnxm6zo
# python train_model.py --model multi_time_model --time_resolution_levels 2 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --layer_norm 1 --mesh_residual 1 --attention_first 1

# 5723609
# multi_time_model-4x64-06_10_21-8581
# ./wandb/offline-run-20240610_210848-djbp8q4d
# python train_model.py --model multi_time_model --time_resolution_levels 3 --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5723648
# multi_time_model-4x64-06_10_21-5936 
# ./wandb/offline-run-20240610_214140-xu7kynqn
# python train_model.py --model multi_time_model --time_resolution_levels 3 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --layer_norm 1

# 5723654
# graph_lam-4x64-06_10_22-4577
# ./wandb/offline-run-20240610_221400-ugsa40s3
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --epochs 2

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --eval val --load ./saved_models/graph_lam-4x64-06_06_03-7248/min_val_loss.ckpt

# uk max 
# python train_model.py --model graph_lam --dataset era5_uk_max --graph uk_max_ico --batch_size 8 --eval test --val_loss_mask 1 --load ./saved_models/graph_lam-4x64-06_07_02-4091/min_val_loss.ckpt

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1

# 5726451
# ./wandb/offline-run-20240611_181219-z89vpilc
# graph_lam-4x64-06_11_18-4552
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 

# 5726539   
# ./wandb/offline-run-20240611_184511-cke3pfax
# graph_lam-4x64-06_11_18-8972
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --layer_norm 1 --mesh_residual 1

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --layer_norm 1 --mesh_residual 1

# 5726674
# ./wandb/offline-run-20240612_005502-5cztwnax
# graph_lam-4x64-06_12_00-8439
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small_v3 --batch_size 8 --layer_norm 1 --mesh_residual 1

# 5726693
# graph_lam-4x64-06_12_01-8098
# ./wandb/offline-run-20240612_014911-h6ovb8tu
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8

# 5727424
# graph_lam-4x64-06_12_13-7912
# ./wandb/offline-run-20240612_130401-0rnu0gyw
# python train_model.py --model graph_lam --dataset era5_uk_big_2_years --graph uk_big_ico --batch_size 8 --two_years 1

# 5728317
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --layer_norm 1 --mesh_residual 1

# 5728318
# python train_multi_model.py --model graph_lam --dataset era5_uk_max_small --batch_size 8 --layer_norm 1 --mesh_residual 1

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8

# 
# python train_multi_model.py --model graph_lam --dataset era5_uk_max_small --batch_size 8

# 5728495 / 5728497
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --val_loss_mask 1

# 5728513
# python train_model.py --model graph_lam --dataset era5_uk_max --graph uk_max_ico --batch_size 8 --val_loss_mask 1

# 5728728
# 
# python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --step_length 4

# python train_multi_model.py --model attention_lam_v2 --dataset era5_uk_big_small --batch_size 8

# python train_model.py --model graph_lam --dataset era5_uk_big_2_years --graph uk_big_ico --batch_size 8 --two_years 1

python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 