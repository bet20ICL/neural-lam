import os
import numpy as np
import torch

class GraphData:
    def __init__(
        self,
        grid_xy, 
        mesh_pos, 
        m2m_edge_index, m2m_edge_set, 
        g2m_edge_index, g2m_edge_set, 
        m2g_edge_index, m2g_edge_set,
        mesh_node_levels,
        mesh_uids
    ):
        self.grid_xy = grid_xy
        self.mesh_pos = mesh_pos
        self.m2m_edge_index = m2m_edge_index
        self.g2m_edge_index = g2m_edge_index
        self.m2g_edge_index = m2g_edge_index
        self.m2m_edge_set = m2m_edge_set
        self.g2m_edge_set = g2m_edge_set
        self.m2g_edge_set = m2g_edge_set
        self.mesh_node_levels = mesh_node_levels
        self.mesh_uids = mesh_uids

def load_graph(dataset, graph_name):
    dir = f"./data/{dataset}/static/nwp_xy.npy"
    graph_dir_path = os.path.join("graphs", graph_name)
    
    # Load grid
    grid_xy = np.load(dir)
    print(grid_xy.shape)
    grid_xy = grid_xy.reshape(2, -1)

    # Load mesh nodes
    mesh_pos = torch.load(os.path.join(graph_dir_path, "mesh_pos.pt")).T # (2, n_mesh)
    mesh_pos = mesh_pos[[0, 1]]
    m2m_edge_index = torch.load(os.path.join(graph_dir_path, "m2m_edge_index.pt"))[0]
    edge_set = {tuple(sorted(e)) for e in m2m_edge_index.T}

    g2m_edge_index = torch.load(os.path.join(graph_dir_path, "g2m_edge_index.pt"))
    g2m_edge_set = sorted(list({tuple(e) for e in g2m_edge_index.T}))

    m2g_edge_index = torch.load(os.path.join(graph_dir_path, "m2g_edge_index.pt"))
    m2g_edge_set = sorted(list({tuple(e) for e in m2g_edge_index.T}))

    mesh_node_levels = torch.load(os.path.join(graph_dir_path, "mesh_node_levels.pt"))
    mesh_uids = torch.load(os.path.join(graph_dir_path, "mesh_uids.pt"))
    
    return GraphData(grid_xy, mesh_pos, m2m_edge_index, edge_set, g2m_edge_index, g2m_edge_set, m2g_edge_index, m2g_edge_set, mesh_node_levels, mesh_uids)

def degrees(edge_index):
    degrees = [0] * (max(edge_index[0]) + 1)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[0, i]] += 1
        
    return max(degrees), min(degrees)

def grid_extent(grid_xy):
    x, y = grid_xy
    return [x.min(), x.max(), y.min(), y.max()]

def verify_graph(graph):
    g2m_edge_index = graph.g2m_edge_index
    m2g_edge_index = graph.m2g_edge_index
    grid_xy = graph.grid_xy
    
    print("Verify g2m connectivity")
    print("Grid Nodes min, max: ", g2m_edge_index[0].min(), g2m_edge_index[0].max())
    print("Mesh nodes min, max: ", g2m_edge_index[1].min(), g2m_edge_index[1].max())
    print("Grid Nodes unique:", g2m_edge_index[0].unique().shape[0])
    print("Mesh nodes unique:", g2m_edge_index[1].unique().shape[0])
    
    print("Verify m2g connectivity")
    print("Grid Nodes min, max: ", m2g_edge_index[0].min(), m2g_edge_index[0].max())
    print("Mesh nodes min, max: ", m2g_edge_index[1].min(), m2g_edge_index[1].max())
    print("Grid Nodes unique:", m2g_edge_index[0].unique().shape[0])
    print("Mesh nodes unique:", m2g_edge_index[1].unique().shape[0])
    
    grid_box = grid_extent(grid_xy)
    print("Grid Bounding box:")
    print(grid_box)