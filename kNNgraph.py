# create kNN graph from data
import sklearn
import pandas as pd
import numpy as np
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph
import networkx as nx

# world data (2d and 3d)
world3d = pd.read_csv("./world_data/rawdata_world_3d.csv")
world2d = pd.read_csv("./world_data/rawdata_world_2d.csv")

# circle data
circle = pd.read_csv("./circle_data/rawdata_circle.csv")

# swiss roll data
swiss = pd.read_csv("./swiss_roll_data/rawdata_swiss_roll.csv")

world3dy = pd.DataFrame(data=world3d["y"], columns=["y"])
circle_y = pd.DataFrame(data=circle["y"], columns=["y"])
swiss_y = pd.DataFrame(data=swiss["y"], columns=["y"])

# remove label y
world3d.drop(["y"], axis=1, inplace=True)
circle.drop(["y"], axis=1, inplace=True)
swiss.drop(["y"], axis=1, inplace=True)

print(world3d.shape)
print(circle.shape)
print(swiss.shape)

# kNN graph for first

X = circle.iloc[0:60, :]
print(X.shape)

A = kneighbors_graph(X, 3, mode="connectivity", include_self=False)


# GRAPHS
g = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph())

nx.draw(g)
plt.savefig("graphtest3NN.png", dpi=300, transparent=True)

