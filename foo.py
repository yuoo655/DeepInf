import numpy as np

from utils import  load_data
"""
 |V|: total number of nodes of the social network
 m: the number of instances, each instance is an ego-network of a user samlped from the social network.
 n: node number of the sub-network

 embeddings: pretrained node embeddings with shape (|V|, 64)
 vertex_features: vertext features with shape (|V|, 7)
 train_data: training dataset
 valid_data: validation dataset
 test_data: test dataset
 *_graphs: the sampled sub-graphs of a user, which is represented as adjacency matrix. shape: (m, n, n)
 *_inf_features: two dummy features indicating whether the user is active and whether the user is the ego. shape: (m,n,2)
 
 *_vertices: node ids of the sampled ego-network, each id is a value from 0 to |V|-1. shape:(m,n)
 *_labels: corresponding label of each instance. shape:(m,)
"""
embeddings, vertex_features, train_data, valid_data, test_data = load_data(64)

train_graphs, train_inf_features, train_labels, train_vertices= train_data
valid_graphs, valid_inf_features, valid_labels, valid_vertices= valid_data
test_graphs, test_inf_features, test_labels, test_vertices= test_data

print(vertex_features.shape)
print(embeddings.shape)
print(train_graphs.shape)
print(train_vertices.shape)
print(train_inf_features.shape)
print(train_labels.shape)

"""
acquire the corresponding vertex features and embeddings of an instance.
"""
vertex_features[train_vertices[0],:]
embeddings[train_vertices[0],:]