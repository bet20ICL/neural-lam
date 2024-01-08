import torch
import torch_geometric as pyg
import einops

from torch.nn import ReLU
from torch_geometric.nn import GAT

from neural_lam import utils
from neural_lam.models.base_graph_model import BaseGraphModel


class GATModel(BaseGraphModel):
    """
    GAT model for weather forecasting.

    Don't use any edge features for now.

    Based on GraphCast, but using GAT instead of InteractionNet for GNN layers.
    """

    def __init__(self, args):
        super().__init__(args)

        assert not self.hierarchical, "GAT Model does not use a hierarchical mesh graph"

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

        # GNNs
        # processor
        self.processor = GAT(
            in_channels=args.hidden_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.processor_layers,
            out_channels=args.hidden_dim,
            v2=True,
            act=ReLU(inplace=True),
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """

        # GAT uses operations with no deterministic implementation in CUDA
        # To train GAT on GPU, turn off deterministic algorithms
        determ = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        # Process each mesh in batch separately
        # This is required since GAT does not support batched processing
        mesh_rep = torch.stack(
            [
                self.processor(graph, edge_index=self.m2m_edge_index)
                for graph in mesh_rep
            ],
            dim=0,
        )
        torch.use_deterministic_algorithms(determ)
        # mesh_rep = self.processor(mesh_rep, self.m2m_edge_index) # (B, N_mesh, d_h)
        return mesh_rep
