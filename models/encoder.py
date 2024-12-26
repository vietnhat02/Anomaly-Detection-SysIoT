import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, node_features, feature_dim, embed_dim, adj_lists, aggregator,
                 num_sample=None, base_model=None, gcn=False, cuda=False):
        super(Encoder, self).__init__()
        self.node_features = node_features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim + self.embed_dim if self.gcn else self.feat_dim)
        )
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if self.gcn:
            if self.cuda:
                self_feats = self.node_features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.node_features(torch.LongTensor(nodes))
            neigh_feats[torch.isnan(neigh_feats)] = 1e-2
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = self.weight.matmul(combined.t())
        combined = F.relu(combined)
        return combined
