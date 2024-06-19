# Standard library
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl

import torch
from lightning_fabric.utilities import seed
import wandb

# First-party
from neural_lam import constants, utils
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.attention_lam import AttentionLAM
from neural_lam.models.multi_ar_model import MultiARModel
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.models.gcn_lam import GCNLAM
from neural_lam.models.gat_lam import GATLAM
from neural_lam.models.stats_model import StatsModel

from neural_lam.weather_dataset import WeatherDataset
from neural_lam.era5_dataset import ERA5UKDataset, ERA5MultiResolutionDataset
from neural_lam.constants import MEPSConstants, ERA5UKConstants
from train_model import get_args

import os
# Required for running jobs on GPU node
os.environ["WANDB_CONFIG_DIR"] = "/work/ec249/ec249/bet20/.config/wandb"
wandb_mode = os.environ.get('WANDB_MODE')
print(f"WANDB_MODE: {wandb_mode}")

MODELS = {
    "graph_lam": GraphLAM,
    "gcn_lam": GCNLAM,
    "gat_lam": GATLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
    "stats_model": StatsModel,
}
    
def main():
    """
    Main function for training and evaluating models
    """
    args = get_args()
    
    # Get an (actual) random run id as a unique identifier
    # This needs to run before seed is set
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)
    
    if args.dataset == "era5_uk_big_small":
        args.dataset_names = ["era5_uk_big_coarse", "era5_uk_small"]
        args.graphs = ["uk_big_coarse_ico", "uk_small_ico"]
        args.coarse2fine_edges = [None, "big_coarse2fine"]
        
    elif args.dataset == "era5_uk_big_small_v2":
        args.dataset_names = ["era5_uk_big_coarse", "era5_uk_small"]
        args.graphs = ["uk_big_coarse_ico", "uk_small_ico"]
        args.coarse2fine_edges = [None, "big_coarse2fine_v2"]
        
    elif args.dataset == "era5_uk_big_small_v3":
        args.dataset_names = ["era5_uk_big_coarse", "era5_uk_small"]
        args.graphs = ["uk_big_coarse_ico", "uk_small_ico"]
        args.coarse2fine_edges = [None, "big_coarse2fine_v3"]
        
    elif args.dataset == "era5_uk_max_small":
        args.dataset_names = ["era5_uk_max_coarse", "era5_uk_small"]
        args.graphs = ["uk_max_coarse_ico", "uk_small_ico"]
        args.coarse2fine_edges = [None, "max_coarse2fine"]
        
    elif args.dataset == "era5_uk_max_small_v2":
        args.dataset_names = ["era5_uk_max_coarse", "era5_uk_small"]
        args.graphs = ["uk_max_coarse_ico", "uk_small_ico"]
        args.coarse2fine_edges = [None, "max_coarse2fine_v2"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_set = ERA5MultiResolutionDataset(
        args.dataset_names,
        pred_length=args.ar_steps,
        split="train",
        standardize=bool(args.standardize),
        args=args,
    )
    val_set = ERA5MultiResolutionDataset(
        args.dataset_names,
        pred_length=12,
        split="val",
        standardize=bool(args.standardize),
        args=args,
    )
    args.constants = ERA5UKConstants
    
    if args.load:
        model = MultiARModel.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        # Load model parameters Use new args for model
        model = MultiARModel(args)
        
    # summary = ModelSummary(model, max_depth=3)
    # print(summary)
    print("===== Model initialized =====")

    # Load data
    train_loader = torch.utils.data.DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
        print("===== CUDA ENABLED =====")
        print("Using deterministic algorithms:", torch.are_deterministic_algorithms_enabled())
    else:
        device_name = "cpu"

    # Make run name
    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )
    # Print out the first layer of models at all layers
    model_summary_callback = pl.callbacks.ModelSummary(max_depth=3)
    logger = pl.loggers.WandbLogger(
        project=constants.WANDB_PROJECT,
        name=run_name,
        config=args,
        offline=True,
    )
    print("===== Logger initialized =====")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[model_summary_callback, checkpoint_callback],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        overfit_batches=args.overfit,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger, args.constants)  # Do after wandb.init
    print("===== Run Name =====")
    print(wandb.run.name)
    print("===== Trainer initialized =====")
    
    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            # TODO: currently using val set for testing
            test_set = val_set
            
            eval_loader = torch.utils.data.DataLoader(
                test_set,
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        start_time = time.time()
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        end_time = time.time()
        elapsed_time_mins = (end_time - start_time) / 60
        wandb.log({"time_elapsed": elapsed_time_mins})
        
        if "era5" in args.dataset:
            # TODO: currently using val set for testing
            test_set = val_set
        else:
            test_set = WeatherDataset(
                args.dataset,
                pred_length=max_pred_length,
                split="test",
                subsample_step=args.step_length,
                subset=bool(args.subset_ds),
            )
        
        eval_loader = torch.utils.data.DataLoader(
            test_set,
            args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader)


if __name__ == "__main__":
    main()
