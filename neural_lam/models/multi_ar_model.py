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
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.attention_lam import AttentionLAM
from neural_lam.models.attention_lam_v2 import AttentionLAMv2

class MultiARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr
        self.args = args
        self.constants = args.constants
        self.n_levels = len(args.dataset_names)
        self.output_std = bool(args.output_std)
        
        self.models = nn.ModuleList()
    
        # First model is lowest resolution
        # Does not need to attend to any other model
        args.dataset = args.dataset_names[0]
        args.graph = args.graphs[0]
        args.coarse2fine = args.coarse2fine_edges[0]
        args.is_first_model = True
        self.models.append(AttentionLAM(args))
        
        args.is_first_model = False
        
        for i in range(1, self.n_levels):
            args.dataset = args.dataset_names[i]
            args.graph = args.graphs[i]
            args.coarse2fine = args.coarse2fine_edges[i]
            if args.model == "attention_lam_v2":
                m = AttentionLAMv2(args)
            else:
                m = AttentionLAM(args)
            self.models.append(m)

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.step_length = args.step_length * 6 # Number of hours per pred. step
        self.val_metrics = [
            {"mse": [],}
            for _ in range(self.n_levels)
        ]
        self.test_metrics = [
            {"mse": [], "mae": [],}
            for _ in range(self.n_levels)
        ]
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = [
            [] for _ in range(self.n_levels)
        ]
        

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


    def predict_step_3_levels(
        self, 
        prev_state, 
        prev_prev_state,
        forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        
        Output: X_t+1, std(X_t+1) or None
        """
        
        # Hard coded for 2 levels at the moment
        coarse_prev_state, fine_prev_state = prev_state
        coarse_prev_prev_state, fine_prev_prev_state, = prev_prev_state
        coarse_forcing, fine_forcing = forcing
        
        coarse_grid, coarse_std, coarse_mesh = self.models[0].predict_step(
            coarse_prev_state,
            coarse_prev_prev_state, 
            coarse_forcing
        )
        
        fine_grid, fine_std, _ = self.models[1].predict_step(
            fine_prev_state, 
            fine_prev_prev_state,
            fine_forcing,
            coarse_mesh, # attend to the coarse mesh
        )
        
        return [coarse_grid, fine_grid], [coarse_std, fine_std]
        # raise NotImplementedError("No prediction step implemented")
        
    def predict_step_2_levels(
        self, 
        prev_state, 
        prev_prev_state,
        forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        
        Output: X_t+1, std(X_t+1) or None
        """
        
        # Hard coded for 2 levels at the moment
        coarse_prev_state, fine_prev_state = prev_state
        coarse_prev_prev_state, fine_prev_prev_state, = prev_prev_state
        coarse_forcing, fine_forcing = forcing
        
        coarse_grid, coarse_std, coarse_mesh = self.models[0].predict_step(
            coarse_prev_state,
            coarse_prev_prev_state, 
            coarse_forcing
        )
        
        fine_grid, fine_std, _ = self.models[1].predict_step(
            fine_prev_state, 
            fine_prev_prev_state,
            fine_forcing,
            coarse_mesh, # attend to the coarse mesh
        )
        
        return [coarse_grid, fine_grid], [coarse_std, fine_std]
        # raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        n_levels = len(init_states)
        prev_prev_state = [init_states[i][:, 0] for i in range(n_levels)]
        prev_state = [init_states[i][:, 1] for i in range(n_levels)]
        pred_steps = forcing_features[0].shape[1]
        
        prediction_per_level = [[] for _ in range(n_levels)]
        # pred_std_per_level = [[] for _ in range(n_levels)]
        pred_std_per_level = [None for _ in range(n_levels)]

        for i in range(pred_steps):
            forcing = [forcing_features[j][:, i] for j in range(n_levels)]

            if n_levels == 2:
                pred_state, pred_std = self.predict_step_2_levels(
                    prev_state, prev_prev_state, forcing
                )
            else:
                pass
            # pred_state: (B, num_grid_nodes, d_f) * n_levels
            # pred_std: (B, num_grid_nodes, d_f) * n_levels or None * n_levels

            for j in range(n_levels):
                prediction_per_level[j].append(pred_state[j])
                # pred_std_per_level[j].append(pred_std[j])
                pred_std_per_level[j] = pred_std[j]
                 
            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state
            
        prediction = [
            torch.stack(
                prediction_per_level[i], dim=1
            ) # (B, pred_steps, num_grid_nodes, d_f)
            for i in range(n_levels)
        ]
        
        # TODO: fix this logic when self.output_std is implemented
        # pred_std = [
        #     torch.stack(
        #         pred_std_per_level[i], dim=1
        #     )
        #     for i in range(n_levels)
        # ]
        return prediction, pred_std

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

        return prediction, target_states, pred_std
    
    def forward(self, batch):
        """
        Forward pass for profiling with fvcore
        Not used otherwise
        """
        return self.common_step(batch)[0]
    
    # def backward(self, loss, optimizer, optimizer_idx):
    #     super().backward(loss, optimizer, optimizer_idx)
    #     unused_params = []
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             unused_params.append(name)
    #     if unused_params:
    #         print(f"Unused parameters: {unused_params}")

    def training_step(self, batch):
        """
        Train on single batch
        
        Input: batch (X (weather state), Y (correct next state))
        Output: loss (L(f(X), Y) some function of model ouptut and correct output)
        
        n_levels: multi resolution levels
        batch:
            init_states:
                list of length n_levels, each containing:
                init_state: (B, 2, num_grid_nodes, d_features)
            target_states:
                list of length n_levels, each containing:
                target_state: (B, pred_steps, num_grid_nodes, d_features)
            forcing_features: 
                list of length n_levels, each containing:
                forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
        """
        prediction, target, pred_std = self.common_step(batch)
        n_levels = len(batch[0])
        
        # Compute loss for each level in hierarchy
        batch_loss_per_level = []
        for i in range(n_levels):
            level_loss = torch.mean(
                self.loss(
                    prediction[i], target[i], pred_std[i], mask=self.interior_mask_bool
                )
            ) # mean over unrolled times and batch
            batch_loss_per_level.append(level_loss)
        
        batch_loss = torch.mean(torch.stack(batch_loss_per_level))
        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss
    
    # def on_after_backward(self):
    #     # Check for unused parameters after the backward pass
    #     total, used = 0, 0
    #     for name, param in self.named_parameters():
    #         total += 1
    #         if param.grad is None:
    #             print(f"Parameter {name} was not used in the backward pass.")
    #         else:
    #             used += 1
    #     print(f"Total parameters: {total}, used: {used}")
 

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std = self.common_step(batch)
        n_levels = len(prediction)
        time_step_loss_per_level = []
        mean_loss_per_level = []
        for i in range(n_levels):
            time_step_loss = torch.mean(
                self.loss(
                    prediction[i], target[i], pred_std[i], mask=self.interior_mask_bool
                ),
                dim=0,
            ) # (time_steps-1)
            mean_loss = torch.mean(time_step_loss)
            time_step_loss_per_level.append(time_step_loss)
            mean_loss_per_level.append(mean_loss)

        val_log_dict = {}    
        for i in range(n_levels):
            # Log loss per level, per time step forward and mean
            for step in self.constants.VAL_STEP_LOG_ERRORS:
                val_log_dict[f"level-{i}_val_loss_unroll{step}"] = time_step_loss_per_level[i][step - 1]
            val_log_dict[f"level-{i}_val_mean_loss"] = mean_loss_per_level[i]
            
        val_log_dict[f"val_mean_loss"] = mean_loss_per_level[-1]
            
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )
        for i in range(n_levels):
            # Store MSEs
            entry_mses = metrics.mse(
                prediction[i],
                target[i],
                pred_std[i],
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.val_metrics[i]["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        for i in range(self.n_levels):
            # Create error maps for all test metrics
            self.aggregate_and_plot_metrics(
                self.val_metrics[i], 
                prefix=f"level-{i}_val",
                level=i,
            )

            # Clear lists with validation metrics values
            for metric_list in self.val_metrics[i].values():
                metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, pred_std = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        n_levels = len(prediction)
        time_step_loss_per_level = []
        mean_loss_per_level = []
        for i in range(n_levels):
            time_step_loss = torch.mean(
                self.loss(
                    prediction[i], target[i], pred_std[i], mask=self.interior_mask_bool
                ),
                dim=0,
            ) # (time_steps-1)
            mean_loss = torch.mean(time_step_loss)
            time_step_loss_per_level.append(time_step_loss)
            mean_loss_per_level.append(mean_loss)
            
        test_log_dict = {}
        for i in range(n_levels):
            # Log loss per level, per time step forward and mean
            for step in self.constants.VAL_STEP_LOG_ERRORS:
                test_log_dict[f"level-{i}_test_loss_unroll{step}"] = time_step_loss_per_level[i][step - 1]
            test_log_dict[f"level-{i}_test_mean_loss"] = mean_loss_per_level[i]
            
        test_log_dict[f"test_mean_loss"] = mean_loss_per_level[-1]
        self.log_dict(
            test_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
        for i in range(n_levels):
            for metric_name in ("mse", "mae"):
                metric_func = metrics.get_metric(metric_name)
                batch_metric_vals = metric_func(
                    prediction[i],
                    target[i],
                    pred_std[i],
                    mask=self.interior_mask_bool,
                    sum_vars=False,
                )  # (B, pred_steps, d_f)
                self.test_metrics[i][metric_name].append(batch_metric_vals)

            if self.output_std:
                # Store output std. per variable, spatially averaged
                mean_pred_std = torch.mean(
                    pred_std[i][..., self.interior_mask_bool, :], dim=-2
                )  # (B, pred_steps, d_f)
                self.test_metrics[i]["output_std"].append(mean_pred_std)

            # Save per-sample spatial loss for specific times
            spatial_loss = self.loss(
                prediction[i], target[i], pred_std[i], average_grid=False
            )  # (B, pred_steps, num_grid_nodes)
            log_spatial_losses = spatial_loss[:, self.constants.VAL_STEP_LOG_ERRORS - 1]
            self.spatial_loss_maps[i].append(log_spatial_losses)
            # (B, N_log, num_grid_nodes)

        # TODO: add plotting for ERA5
        # # Plot example predictions (on rank 0 only)
        # if (
        #     self.trainer.is_global_zero
        #     and self.plotted_examples < self.n_example_pred
        # ):
        #     # Need to plot more example predictions
        #     n_additional_examples = min(
        #         prediction.shape[0], self.n_example_pred - self.plotted_examples
        #     )

        #     self.plot_examples(
        #         batch, n_additional_examples, prediction=prediction, args=self.args
        #     )

    def plot_examples(self, batch, n_examples, level, prediction=None, args=None):
        """
        (Used in test_step only)
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target = self.common_step(batch)

        target = batch[1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.data_std[level] + self.data_mean[level]
        target_rescaled = target * self.data_std[level] + self.data_mean[level]

        # Iterate over the examples
        for pred_slice, target_slice in zip(
            prediction_rescaled[:n_examples], target_rescaled[:n_examples]
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # TODO: write code for plotting ERA5 predictions as well
            if args and "era5" not in args.dataset:
                # Iterate over prediction horizon time steps
                for t_i, (pred_t, target_t) in enumerate(
                    zip(pred_slice, target_slice), start=1
                ):
                    # Create one figure per variable at this time step
                    var_figs = [
                        vis.plot_prediction(
                            pred_t[:, var_i],
                            target_t[:, var_i],
                            self.interior_mask[:, 0],
                            title=f"{var_name} ({var_unit}), "
                            f"t={t_i} ({self.step_length*t_i} h)",
                            vrange=var_vrange,
                        )
                        for var_i, (var_name, var_unit, var_vrange) in enumerate(
                            zip(
                                self.constants.PARAM_NAMES_SHORT,
                                self.constants.PARAM_UNITS,
                                var_vranges,
                            )
                        )
                    ]

                    example_i = self.plotted_examples
                    wandb.log(
                        {
                            f"{var_name}_example_{example_i}": wandb.Image(fig)
                            for var_name, fig in zip(
                                self.constants.PARAM_NAMES_SHORT, var_figs
                            )
                        }
                    )
                    plt.close(
                        "all"
                    )  # Close all figs for this time step, saves memory

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_pred_{self.plotted_examples}.pt"
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_target_{self.plotted_examples}.pt"
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        (Used in val and test step)
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        full_log_name = f"{prefix}_{metric_name}"
        log_dict = {}

        metric_fig = vis.plot_error_map(
            metric_tensor, self.constants, step_length=self.step_length
        )
        log_dict[full_log_name] = wandb.Image(metric_fig)
        
        # Plot a summary error map 
        # The same figure as above but with a subset of variables for clarity
        summary_metric_fig = vis.plot_error_map(
            metric_tensor, self.constants, step_length=self.step_length, summary=True
        )
        log_dict[f"{full_log_name}_summary"] = wandb.Image(summary_metric_fig)
        
        # Plot rollout error curves
        summary_metric_curves = vis.plot_error_curves(
            metric_tensor, self.constants, step_length=self.step_length, summary=True
        )
        for name, fig in summary_metric_curves:
            log_dict[f"{full_log_name}_{name}_rollout"] = wandb.Image(fig)

        if "test" in prefix:
            # Save pdf
            if metric_fig:
                metric_fig.savefig(
                    os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
                )
            # Save errors also as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        if full_log_name in self.constants.METRICS_WATCH:
            for var_i, timesteps in self.constants.VAR_LEADS_METRICS_WATCH.items():
                var = self.constants.PARAM_NAMES_SHORT[var_i]
                log_dict.update(
                    {
                        f"{full_log_name}_{var}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]  # 1-indexed in constants
                        for step in timesteps
                    }
                )

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix, level):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.models[level].data_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        for i in range(self.n_levels):
            # Create error maps for all test metrics
            self.aggregate_and_plot_metrics(
                self.test_metrics[i], 
                prefix=f"level-{i}_test",
                level=i
            )
                
            # TODO: figure out plotting for ERA5
            # Plot spatial loss maps
            spatial_loss_tensor = self.all_gather_cat(
                torch.cat(self.spatial_loss_maps[i], dim=0)
            )  # (N_test, N_log, num_grid_nodes)
            if self.trainer.is_global_zero:
                mean_spatial_loss = torch.mean(
                    spatial_loss_tensor, dim=0
                )  # (N_log, num_grid_nodes)

                # TODO: figure out plotting for ERA5
                if "era5" not in self.args.dataset:
                    loss_map_figs = [
                        vis.plot_spatial_error(
                            loss_map,
                            self.interior_mask[:, 0],
                            title=f"Test loss, t={t_i} ({self.step_length*t_i} h)",
                        )
                        for t_i, loss_map in zip(
                            self.constants.VAL_STEP_LOG_ERRORS, mean_spatial_loss
                        )
                    ]

                    # log all to same wandb key, sequentially
                    for fig in loss_map_figs:
                        wandb.log({"test_loss": wandb.Image(fig)})

                    # also make without title and save as pdf
                    pdf_loss_map_figs = [
                        vis.plot_spatial_error(loss_map, self.interior_mask[:, 0])
                        for loss_map in mean_spatial_loss
                    ]
                    pdf_loss_maps_dir = os.path.join(wandb.run.dir, "spatial_loss_maps")
                    os.makedirs(pdf_loss_maps_dir, exist_ok=True)
                    for t_i, fig in zip(
                        self.constants.VAL_STEP_LOG_ERRORS, pdf_loss_map_figs
                    ):
                        fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
                
                # save mean spatial loss as .pt files also
                torch.save(
                    mean_spatial_loss.cpu(),
                    os.path.join(wandb.run.dir, f"level_{i}_mean_spatial_loss.pt"),
                )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
