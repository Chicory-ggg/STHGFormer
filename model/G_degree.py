import torch
import networkx as nx
import pandas as pd

# get matrix
G = pd.read_csv("A.csv", header=None)
G = nx.DiGraph(G)

# get num of nodes
num_nodes = G.number_of_nodes()
num_nodes = int(num_nodes)

# in_degree & out_degree
G_indegree = G.in_degree()
G_ID = []
for i in range(num_nodes):
    G_ID.append(G_indegree[i])
G_outdegree = G.out_degree()
G_OD = []
for i in range(num_nodes):
    G_OD.append(G_outdegree[i])

G_ID = torch.tensor(G_ID, dtype=torch.float)
G_OD = torch.tensor(G_OD, dtype=torch.float)

def num_node():
    return num_nodes
def two_degree():
    return G_ID, G_OD