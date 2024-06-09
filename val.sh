python train_model.py --model graph_lam --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --eval val --val_loss_mask 1 --load ./saved_models/graph_lam-4x64-06_05_14-1856/min_val_loss.ckpt

python train_multi_model.py --model graph_lam --dataset era5_uk_big_small --batch_size 8 --eval val --load ./saved_models/graph_lam-4x64-06_06_03-7248/min_val_loss.ckpt

python train_model.py --model multi_time_model --time_resolution_levels 3 --dataset era5_uk_big --graph uk_big_ico --batch_size 8 --epochs 2