import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, pretrained_emb,
            n_units=[1433, 8, 7], n_heads=8,
            dropout=0.1, attn_dropout=0.0, fine_tune=False):
        super(GAT, self).__init__()
        nhid * nheads

        self.dropout = dropout
        self.attention = GraphAttentionLayer()


    def forward(self, x, vertices, adj):
    #  x inf_features: features: two dummy features indicating whether the user is active and whether the user is the ego. shape: (m,n,2)
    #  vertices: node ids of the sampled ego-network, each id is a value from 0 to |V|-1. shape:(m,n)
    #  adj: the sampled sub-graphs of a user, which is represented as adjacency matrix. shape: (m, n, n)
 
        emb = self.embedding(vertices)
        x = torch.cat((x, emb), dim=2)
        x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)
