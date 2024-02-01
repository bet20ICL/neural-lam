import random
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities import seed
from argparse import ArgumentParser
import time
import os

import torch_geometric as pyg

from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.models.gcn_model import GCNModel
from neural_lam.models.gat_model import GATModel

from neural_lam.weather_dataset import WeatherDataset
from neural_lam.era5_uk_dataset import ERA5UKDataset
from neural_lam import constants, utils

train_loader = pyg.loader.DataLoader(
                ERA5UKDataset(
                    args.dataset,
                    pred_length=args.ar_steps,
                    split="train",
                    subsample_step=args.step_length,
                    subset=bool(args.subset_ds),
                    control_only=args.control_only,
                    standardize=False,
                ),
                args.batch_size, shuffle=True, num_workers=args.n_workers)