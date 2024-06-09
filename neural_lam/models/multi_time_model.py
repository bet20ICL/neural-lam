# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

# First-party
from neural_lam import constants, metrics, utils, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.attention_lam import AttentionLAM

class MultiTimeModel(ARModel):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters()
        self.resolutions = args.resolutions 
        self.n_levels = len(args.resolutions)
        
        self.models = nn.ModuleList()
    
        # First model is lowest resolution
        # Does not need to attend to any other model
        # args.dataset = args.dataset_names[0]
        # args.graph = args.graphs[0]
        args.coarse2fine = args.coarse2fine_edges[0]
        args.is_first_model = True
        args.no_decoder = True
        self.models.append(AttentionLAM(args))
        
        args.is_first_model = False
        for i in range(1, self.n_levels - 1):
            # args.dataset = args.dataset_names[i]
            # args.graph = args.graphs[i]
            args.coarse2fine = args.coarse2fine_edges[i]
            self.models.append(
                AttentionLAM(args)
            )
        args.coarse2fine = args.coarse2fine_edges[-1]
        args.no_decoder = False
        self.models.append(
            AttentionLAM(args)
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        
        This is simply the complement of the border mask (1 - self.border_mask).
        """
        # return self.interior_mask[:, 0].to(torch.bool)
        # TODO: add mask for each level
        return None 

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)
            
    def unroll_2_levels(self, init_states, forcing_features, true_states):
        raise NotImplementedError("No prediction step implemented")
    
    def unroll_3_levels(self, init_states, forcing_features, true_states):
        # total number of steps in rollout
        # nice numbers are 8, 12 
        pred_steps = forcing_features[-1].shape[1]
        
        n_levels = len(self.resolutions)
        prev_prev_state = [
            init_states[i][:, 0] for i in range(n_levels)
        ]
        prev_state = [
            init_states[i][:, 1] for i in range(n_levels)
        ]
        latent_state_per_resolution = [
            [] for _ in range(n_levels - 1) # don't store the latent state of the last level
        ]
        prediction_list = []
        for i in range(pred_steps):
            if i % 4 == 0:
                mesh_state = self.models[0].predict_step(
                    prev_prev_state[0],
                    prev_state[0],
                    forcing_features[0][:, i // 4],
                )
                latent_state_per_resolution[0].append(mesh_state)
                prev_prev_state[0] = prev_state[0]

            if i % 2 == 0:
                mesh_state = self.models[1].predict_step(
                    prev_prev_state[1],
                    prev_state[1],
                    forcing_features[1][:, i // 2],
                    latent_state_per_resolution[0][-1],
                )
                latent_state_per_resolution[1].append(mesh_state)
                prev_prev_state[1] = prev_state[1]
            
            grid_state, _, _ = self.models[2].predict_step(
                prev_prev_state[2],
                prev_state[2],
                forcing_features[2][:, i],
                latent_state_per_resolution[1][-1],
            )
            
            prev_prev_state[2] = prev_state[2]
            prev_state[0] = grid_state
            prev_state[1] = grid_state
            prediction_list.append(grid_state)
            
        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        
        # TODO: implement this    
        if self.output_std:
            pass
        pred_std = self.per_var_std
        
        return prediction, pred_std

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: list of tensors with shape (B, pred_steps, num_grid_nodes, d_f)
        
        [
            (B, pred_steps // 4, num_grid_nodes, d_f) for level 0,
            (B, pred_steps // 2, num_grid_nodes, d_f) for level 1,
            (B, pred_steps, num_grid_nodes, d_f) for level 2,
        ]
        """
        # TODO: refactor so levels aren't hardcoded 
        if len(self.resolutions) == 2:
            return self.unroll_2_levels(init_states, forcing_features, true_states)
        
        if len(self.resolutions) == 3:
            return self.unroll_3_levels(init_states, forcing_features, true_states)
        
        raise NotImplementedError("Only 2 and 3 levels are supported")
            
            
    # TODO: these methods need to be overridden to unwrap target
    # when loading from multitime dataset the target is wrapped in a list for some fucking reason
    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing_features,
        ) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        # Target states is a list
        # The last tensor is the highest resolution
        return prediction, target_states[-1], pred_std
    
    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, forcing = batch
        batch = (prediction, target[-1], forcing)
        return super().test_step(batch, batch_idx)