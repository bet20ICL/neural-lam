import torch

# First-party
from neural_lam import utils
# from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_lam import GraphLAM


class StatsModel(GraphLAM):
    """
    This model always predicts the mean of the dataset.
    
    This model is used to test the performance of the model when the prediction is always the mean of the dataset.
    Will only work in eval mode (will break if used in training mode).
    """

    def __init__(self, args):
        super().__init__(args)
        self.stats = utils.load_dataset_stats(
            args.dataset, device="cuda:0"
        )
        
    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
       
        prediction = self.stats["data_mean"]
        prediction = prediction.repeat(prev_state.shape[0], prev_state.shape[1], 1)
        # print(prediction.shape, prev_state.shape
        
        return prediction, None
        
    # def common_step(self, batch):
    #     (
    #         init_states,
    #         target_states,
    #         forcing_features,
    #     ) = batch
        
        
        # # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # prediction = self.stats["data_mean"]
        # prediction = prediction.repeat(*target_states.shape[:-1], 1)
        # print(prediction.shape)
        # # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        # pred_std = self.per_var_std
        
        # return prediction, target_states, pred_std