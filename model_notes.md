
These are graph features
self.*:
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,

# self.N_grid: number of grid nodes
# self.N_mesh: number of mesh nodes
self.N_grid, grid_static_dim = self.grid_static_features.shape # 63784 = 268x238
self.N_mesh, N_mesh_ignore = self.get_num_mesh()


# number of edges in subgraphs
# number of features in subgraphs
self.g2m_edges, g2m_dim = self.g2m_features.shape
self.m2g_edges, m2g_dim = self.m2g_features.shape



# EMBEDDING LAYERS:
        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim]*(args.hidden_layers + 1)

        # each returns torch mlp module
        self.grid_embedder = utils.make_mlp([grid_dim] +
                self.mlp_blueprint_end)
        self.g2m_embedder = utils.make_mlp([g2m_dim] +
                self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] +
                self.mlp_blueprint_end)

        # Graph Lam Specific
                # Define sub-models
                # Feature embedders for mesh
                self.mesh_embedder = utils.make_mlp([mesh_dim] +
                        self.mlp_blueprint_end)
                self.m2m_embedder = utils.make_mlp([m2m_dim] +
                        self.mlp_blueprint_end)        
# GNNs
# encoder
        self.g2m_gnn = InteractionNet(self.g2m_edge_index,
                args.hidden_dim, hidden_layers=args.hidden_layers, update_edges=False)
        self.encoding_grid_mlp = utils.make_mlp([args.hidden_dim]
                + self.mlp_blueprint_end)


# Graph Lam Specific
# GNNs
# processor
    processor_nets = [
            InteractionNet(
                    self.m2m_edge_index,
                    args.hidden_dim, 
                    hidden_layers=args.hidden_layers, 
                    aggr=args.mesh_aggr
            )
            for _ in range(args.processor_layers)
    ]
    self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep", [
                    (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                    for net in processor_nets
            ]
    )

# decoder
        self.m2g_gnn = InteractionNet(
                self.m2g_edge_index,
                args.hidden_dim,
                hidden_layers=args.hidden_layers,
                update_edges=False)

# Output mapping (hidden_dim -> output_dim)
self.output_map = utils.make_mlp([args.hidden_dim]*(args.hidden_layers + 1) +\
        [self.grid_state_dim], layer_norm=False) # No layer norm on this one
