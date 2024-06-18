# Third-party
import torch
import torch_geometric as pyg
from torch_geometric.nn import TransformerConv
import pytorch_lightning as pl

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.base_graph_model import BaseGraphModel

class InterAttentionNet(pl.LightningModule):
    """
    Implementation of a generic Interaction Network preceded by attention
    """
    def __init__(
        self,
        m2m_edge_index,
        c2f_edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
        layer_norm=True,
        mesh_resid=True,
    ):
        super().__init__()
        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim
        self.coarse2fine_edge_index = c2f_edge_index
        self.layer_norm = layer_norm
        self.mesh_resid = mesh_resid
        self.attention_layer = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )
        self.proc_layer = InteractionNet(
            m2m_edge_index,
            input_dim=input_dim,
            update_edges=update_edges,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            edge_chunk_sizes=edge_chunk_sizes,
            aggr_chunk_sizes=aggr_chunk_sizes,
            aggr=aggr,
        )
        if self.layer_norm:
            self.att_norm = torch.nn.LayerNorm(hidden_dim)
        
    def forward(self, mesh_rep, coarse_mesh_rep, edge_rep):
        """
        mesh_rep: (B, N_mesh, d_h)
        coarse_mesh_rep: (B, N_coarse_mesh, d_h)
        edge_rep: (M_mesh, d_h)
        """
        mesh_rep_batch = torch.reshape(mesh_rep, (-1, mesh_rep.shape[-1]))
        coarse_mesh_rep_batch = torch.reshape(coarse_mesh_rep, (-1, coarse_mesh_rep.shape[-1]))
        N_fine_mesh, N_coarse_mesh = mesh_rep.shape[1], coarse_mesh_rep.shape[1]
        idx_offset = torch.tensor([[N_coarse_mesh], [N_fine_mesh]], device=mesh_rep.device)
        batch_size = mesh_rep.shape[0]
        print(idx_offset.device, self.coarse2fine_edge_index.device)
        edge_index_batch = torch.cat([
            self.coarse2fine_edge_index + idx_offset * i
            for i in range(batch_size)
        ], dim=1)
        
        torch.use_deterministic_algorithms(False)
        mesh_rep_batch, weights = self.attention_layer(
            (coarse_mesh_rep_batch, mesh_rep_batch),
            edge_index_batch,
            return_attention_weights=True,
        )
        torch.use_deterministic_algorithms(True)
        mesh_rep_batch = mesh_rep_batch.reshape((batch_size, N_fine_mesh, mesh_rep_batch.shape[-1]))
        if self.mesh_resid:
            mesh_rep_batch = mesh_rep + mesh_rep_batch
        if self.layer_norm:
            mesh_rep_batch = self.att_norm(mesh_rep_batch)
        mesh_rep, edge_rep = self.proc_layer(mesh_rep_batch, mesh_rep_batch, edge_rep)
        return mesh_rep, edge_rep
        

class AttentionLAMv2(BaseGraphModel):
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

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        print("c2f device:", self.coarse2fine_edge_index.device)
        # GNNs
        # processor
        processor_nets = [
            InterAttentionNet(
                self.m2m_edge_index,
                self.coarse2fine_edge_index,
                args.hidden_dim,
                hidden_layers=args.hidden_layers,
                aggr=args.mesh_aggr,
            )
            for _ in range(args.processor_layers)
        ]
        self.processor = pyg.nn.Sequential(
            "mesh_rep, coarse_mesh_rep, edge_rep",
            [
                (net, "mesh_rep, coarse_mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )
        
    def _update_proccessor_graph(self):
        """
        Update the graph used by the processor. 
        Override this method in subclasses.
        
        If the processor does not require the graph, this method can be empty
        """
        # Update each InteractionNet in processor with the new edge index
        for net in self.processor:
            net.set_edge_index(self.m2m_edge_index)

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep, coarse_mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, M_mesh, d_h)

        mesh_rep, _ = self.processor(
            mesh_rep, coarse_mesh_rep, m2m_emb_expanded
        )  # (B, N_mesh, d_h)
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
        
        mesh_rep = self.process_step(mesh_rep, coarse_mesh_rep)
            
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
