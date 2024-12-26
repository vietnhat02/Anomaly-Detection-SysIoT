import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

def load_sage(path):
    nodes = np.load(path + "nodes.npy", allow_pickle=True)
    num_nodes = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}
    
    adj = np.load(path + "adj.npy", allow_pickle=True)
    label = np.load(path + "label_bi.npy", allow_pickle=True).flatten()
    edge_feat = np.load(path + "edge_feat_scaled.npy", allow_pickle=True)

    if edge_feat.dtype == object:
        edge_feat = np.stack(edge_feat).astype(np.float32)
    else:
        edge_feat = edge_feat.astype(np.float32)

    if len(adj) != len(edge_feat):
        raise ValueError("adj và edge_feat phải có cùng số lượng cạnh.")

    for i, line in enumerate(adj):
        node1, node2 = line
        if node1 not in node_map or node2 not in node_map:
            raise ValueError(f"Nút {node1} hoặc {node2} không có trong node_map.")

    adj_lists = defaultdict(set)
    for i, line in enumerate(adj):
        node1 = node_map[line[0]]
        node2 = node_map[line[1]]
        adj_lists[node1].add(i)
        adj_lists[node2].add(i)

    node_feat = np.ones((num_nodes, 64), dtype=np.float32)
    node_features = nn.Embedding(num_nodes, node_feat.shape[1])
    node_features.weight = nn.Parameter(torch.FloatTensor(node_feat), requires_grad=False)

    edge_features = nn.Embedding(len(edge_feat), edge_feat.shape[1])
    edge_features.weight = nn.Parameter(torch.FloatTensor(edge_feat), requires_grad=False)

    return num_nodes, node_map, adj, label, edge_feat, edge_features, adj_lists, node_features
