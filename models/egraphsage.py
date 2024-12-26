import torch
import torch.nn as nn
import torch.nn.functional as F

class EGraphSage(nn.Module):
    def __init__(self, enc, edge_features, node_map, adj, residual=False, decoder_hidden_dims=[128, 16]):
        super(EGraphSage, self).__init__()
        self.enc = enc
        self.edge_features = edge_features
        self.node_map = node_map
        self.adj = adj
        self.residual = residual

        decoder_input_dim = 2 * enc.embed_dim + edge_features.embedding_dim if residual else 2 * enc.embed_dim

        decoder_layers = []
        input_dim = decoder_input_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, decoder_input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, edges):
        lines = self.adj[edges]
        node1, node2 = lines[:,0], lines[:,1]
        
        nodes_id_1 = [self.node_map[node] for node in node1]
        nodes_id_2 = [self.node_map[node] for node in node2]

        embeds1 = self.enc(nodes_id_1).t().to(self.decoder[0].weight.device)
        embeds2 = self.enc(nodes_id_2).t().to(self.decoder[0].weight.device)

        if self.residual:
            edge_feats = self.edge_features(torch.LongTensor(edges)).to(self.decoder[0].weight.device)
            edge_embeds = torch.cat([embeds1, embeds2, edge_feats], dim=1)
        else:
            edge_embeds = torch.cat([embeds1, embeds2], dim=1)

        reconstructed = self.decoder(edge_embeds)

        if self.residual:
            reconstructed = reconstructed + edge_embeds

        return reconstructed, edge_embeds