import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

class SimpleGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGNN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, adj_matrix):
        # Simple message passing
        x = self.conv1(x)
        x = F.relu(torch.matmul(adj_matrix, x))
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

def visualize_graph(adj_matrix):
    G = nx.from_numpy_array(adj_matrix.numpy())
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', 
            node_size=500, width=2, edge_color='gray')
    plt.savefig('graph.png')
    print("Graph visualization saved as 'graph.png'")
