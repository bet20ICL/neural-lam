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
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.models.gcn_lam import GCNLAM
from neural_lam.models.gat_lam import GATLAM

from neural_lam.weather_dataset import WeatherDataset
from neural_lam.era5_dataset import ERA5UKDataset
from neural_lam.constants import MEPSConstants, ERA5UKConstants

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
}

def get_args():
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )

    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        type=int,
        default=0,
        help="Train only on control member of ensemble data "
        "(default: 0 (False))",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metrics.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--standardize",
        type=int,
        default=1,
        help="Standardize dataset "
        "(default: 1 (true))",
    )
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="Overfit to single batch for debugging "
        "(default: 0 (false))",
    )
    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    # Ablation 
    parser.add_argument(
        "--simple_grid",
        type=int,
        default=0,
        help="Use simple grid features (default: 0 (false))",
    )
    parser.add_argument(
        "--simple_param_weights",
        type=int,
        default=0,
        help="Use simple parameter weights (default: 0 (false))", 
    )
    parser.add_argument(
        "--no_forcing",
        type=int,
        default=0,
        help="Do not add forcing features to dataset (default: 0 (false))",
    )
    args = parser.parse_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"
    
    return args
    
def main():
    """
    Main function for training and evaluating models
    """
    args = get_args()

    # Set seed
    seed.seed_everything(args.seed)
    
    if args.dataset == "era5_uk":
        train_set = ERA5UKDataset(
            args.dataset,
            pred_length=args.ar_steps,
            split="train",
            standardize=bool(args.standardize),
            args=args,
        )
        val_set = ERA5UKDataset(
            args.dataset,
            pred_length=28,
            split="val",
            standardize=bool(args.standardize),
            args=args,
        )
        args.constants = ERA5UKConstants
    elif args.dataset == "meps_example":
        train_set = WeatherDataset(
            args.dataset,
            pred_length=args.ar_steps,
            split="train",
            subsample_step=args.step_length,
            standardize=bool(args.standardize),
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        )
        
        max_pred_length = (65 // args.step_length) - 2  # 19
        val_set = WeatherDataset(
            args.dataset,
            pred_length=max_pred_length,
            split="val",
            subsample_step=args.step_length,
            standardize=bool(args.standardize),
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        )
        args.constants = MEPSConstants
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)
    print("===== Model initialized =====")

    # Make run name
    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)
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
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        overfit_batches=args.overfit,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger, args.constants)  # Do after wandb.init
    print("===== Trainer initialized =====")
    
    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    args.dataset,
                    pred_length=max_pred_length,
                    split="test",
                    subsample_step=args.step_length,
                    subset=bool(args.subset_ds),
                ),
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


if __name__ == "__main__":
    main()
