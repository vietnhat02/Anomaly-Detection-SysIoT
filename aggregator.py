# models/aggregator.py
import torch
import torch.nn as nn
import random
from torch.autograd import Variable

class MeanAggregator(nn.Module):
    def __init__(self, edge_features, cuda=False, gcn=False):
        super(MeanAggregator, self).__init__()
        self.edge_features = edge_features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=None):
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [
                _set(_sample(list(to_neigh), num_sample)) if len(to_neigh) >= num_sample else to_neigh
                for to_neigh in to_neighs
            ]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh.union(set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]

        unique_edges_list = list(set.union(*samp_neighs))
        unique_edges = {n: i for i, n in enumerate(unique_edges_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_edges)))
        column_indices = [unique_edges[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.edge_features(torch.LongTensor(unique_edges_list).cuda())
        else:
            embed_matrix = self.edge_features(torch.LongTensor(unique_edges_list))
        
        to_feats = mask.mm(embed_matrix)
        return to_feats