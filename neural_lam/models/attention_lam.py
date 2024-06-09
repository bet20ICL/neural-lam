# Third-party
import torch
import torch_geometric as pyg
from torch_geometric.nn import TransformerConv

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.graph_lam import GraphLAM

class AttentionLAM(GraphLAM):
    """
    Full graph-based LAM model that can be used with different
    (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
    Oskarsson et al. (2023).
    """

    def __init__(self, args):
        super().__init__(args)

        assert (
            not self.hierarchical
        ), "GraphLAM does not use a hierarchical mesh graph"
        
        self.is_first_model = args.is_first_model
        self.mesh_residual = getattr(args, "mesh_residual", None)
        self.attention_first = getattr(args, "attention_first", None)
        # mesh_dim = self.mesh_static_features.shape[1]
        if not self.is_first_model:
            self.attention_layer = TransformerConv(
                in_channels=args.hidden_dim,
                out_channels=args.hidden_dim,
            )
    
    def cross_attention(self, mesh_rep, coarse_mesh_rep):
        mesh_rep_batch = torch.reshape(mesh_rep, (-1, mesh_rep.shape[-1])) # (B*N_mesh, d_h)
        coarse_mesh_rep_batch = torch.reshape(coarse_mesh_rep, (-1, coarse_mesh_rep.shape[-1])) # (B*N_mesh, d_h)
        N_fine_mesh, N_coarse_mesh = mesh_rep.shape[1], coarse_mesh_rep.shape[1]
        idx_offset = torch.tensor([[N_coarse_mesh], [N_fine_mesh]], device=mesh_rep.device) # (2, 1)
        batch_size = mesh_rep.shape[0]
        edge_index_batch = torch.cat([
            self.coarse2fine_edge_index + idx_offset * i
            for i in range(batch_size)
        ], dim=1) # (2, B*M_mesh)
                
        # TransformerConv uses 'scatter' which does not have a deterministic implementation
        torch.use_deterministic_algorithms(False)
        mesh_rep_batch = self.attention_layer(
            (coarse_mesh_rep_batch, mesh_rep_batch),
            edge_index_batch,
        )
        torch.use_deterministic_algorithms(True)

        mesh_rep = mesh_rep_batch.reshape((batch_size, N_fine_mesh, mesh_rep_batch.shape[-1]))
        return mesh_rep
    
    def predict_step(
        self, 
        prev_state, 
        prev_prev_state,
        forcing, 
        coarse_mesh_rep=None,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        batch_size = prev_state.shape[0]

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        if not self.no_decoder:
            m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        
        if self.attention_first:
            # Attention Step
            if not self.is_first_model:
                # coarse_mesh_rep: (B, num_coarse_mesh_nodes, d_h)
                mesh_att = self.cross_attention(mesh_rep, coarse_mesh_rep)
                # and possibly more message passing steps here
                if self.mesh_residual:
                    mesh_rep = mesh_rep + mesh_att
                else:
                    mesh_rep = mesh_att

            # Run processor step
            mesh_rep = self.process_step(mesh_rep)
        else:
            # Run processor step
            mesh_rep = self.process_step(mesh_rep)
            
            # Attention Step
            if not self.is_first_model:
                # coarse_mesh_rep: (B, num_coarse_mesh_nodes, d_h)
                mesh_att = self.cross_attention(mesh_rep, coarse_mesh_rep)
                # and possibly more message passing steps here
                if self.mesh_residual:
                    mesh_rep = mesh_rep + mesh_att
                else:
                    mesh_rep = mesh_att
            
            
        
        if self.no_decoder:
            return mesh_rep
        
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)
        
        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, d_f)
            # Note: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = self.per_var_std # (d_f,) or (num_grid_nodes, d_f)

        # Rescale with one-step difference statistics
        rescaled_delta_mean = (
            pred_delta_mean * self.step_diff_std + self.step_diff_mean
        )

        # Residual connection for full state
        return prev_state + rescaled_delta_mean, pred_std, mesh_rep
