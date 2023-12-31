import random
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities import seed
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import wandb

from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.models.gcn_model import GCNModel

from neural_lam.weather_dataset import WeatherDataset
from neural_lam import constants, utils

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
    "gcn": GCNModel,
}

def get_args():
    parser = ArgumentParser(description='Train or evaluate NeurWP models for LAM')

    # General options
    parser.add_argument('--dataset', type=str, default="meps_example",
        help='Dataset, corresponding to name in data directory (default: meps_example)')
    parser.add_argument('--model', type=str, default="graph_lam",
        help='Model architecture to train/evaluate (default: graph_lam)')
    parser.add_argument('--subset_ds', type=int, default=0,
        help='Use only a small subset of the dataset, for debugging (default: 0=false)')
    parser.add_argument('--seed', type=int, default=42,
        help='random seed (default: 42)')
    parser.add_argument('--n_workers', type=int, default=4,
        help='Number of workers in data loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
        help='upper epoch limit (default: 200)')
    parser.add_argument('--batch_size', type=int, default=4,
        help='batch size (default: 4)')
    parser.add_argument('--load', type=str,
        help='Path to load model parameters from (default: None)')
    parser.add_argument('--restore_opt', type=int, default=0,
        help='If optimizer state shoudl be restored with model (default: 0 (false))')
    parser.add_argument('--precision', type=str, default=32,
        help='Numerical precision to use for model (32/16/bf16) (default: 32)')

    # Model architecture
    parser.add_argument('--graph', type=str, default="multiscale",
        help='Graph to load and use in graph-based model (default: multiscale)')
    parser.add_argument('--hidden_dim', type=int, default=64,
        help='Dimensionality of all hidden representations (default: 64)')
    parser.add_argument('--hidden_layers', type=int, default=1,
        help='Number of hidden layers in all MLPs (default: 1)')
    parser.add_argument('--processor_layers', type=int, default=4,
        help='Number of GNN layers in processor GNN (default: 4)')
    parser.add_argument('--mesh_aggr', type=str, default="sum",
        help='Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)')

    # Training options
    parser.add_argument('--ar_steps', type=int, default=1,
        help='Number of steps to unroll prediction for in loss (1-19) (default: 1)')
    parser.add_argument('--control_only', type=int, default=0,
        help='Train only on control member of ensemble data (default: 0 (False))')
    parser.add_argument('--loss', type=str, default="mse",
        help='Loss function to use (default: mse)')
    parser.add_argument('--step_length', type=int, default=3,
        help='Step length in hours to consider single time step 1-3 (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate (default: 0.001)')
    parser.add_argument('--val_interval', type=int, default=1,
        help='Number of epochs training between each validation run (default: 1)')

    # Evaluation options
    parser.add_argument('--eval', type=str,
        help='Eval model on given data split (val/test) (default: None (train model))')
    parser.add_argument('--n_example_pred', type=int, default=1,
        help='Number of example predictions to plot during evaluation (default: 1)')

    # Parse
    args = parser.parse_args()
    
    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"

    return args

def main():
    args = get_args()
    
    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    train_loader = torch.utils.data.DataLoader(
            WeatherDataset(
                args.dataset, 
                pred_length=args.ar_steps, 
                split="train",
                subsample_step=args.step_length, 
                subset=bool(args.subset_ds),
                control_only=args.control_only
            ),
            args.batch_size, shuffle=True, num_workers=args.n_workers)
    # max_pred_length = (65 // args.step_length) - 2 # 19
    
    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)
    
    batch = next(iter(train_loader))
    output = model.training_step(batch)
    
    init_states, target_states, static_features, forcing_windowed = batch
    # print("Batch shape: ", batch.shape)
    print("Init states shape: ", init_states.shape)
    print("Target states shape: ", target_states.shape)
    print("Static features shape: ", static_features.shape)
    print("Forcing windowed shape: ", forcing_windowed.shape)
    print("Output shape: ", output.shape)
    
if __name__ == "__main__":
    main()