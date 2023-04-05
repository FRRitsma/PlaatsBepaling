""" 
Generate the files required for formulating the problem
"""

# %% Imports:
import os
os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as Tree
from hilbert.hilbert import HilbertCoder
from itertools import combinations
import pickle


# Fix random seed for repeatability:
np.random.seed(0)

# HARDCODES:
NO_CLUSTERS = 4
NO_HOUSES = 400
NO_FACILITIES = 3
NO_STATES = 100


# Create cluster centers:
grid = np.arange(3)
grid = np.meshgrid(grid, grid)
cluster_centers = np.vstack([grid[0].ravel(), grid[1].ravel()]).T


# House locations:
house_locations = np.empty([0, 2])
for i in cluster_centers:
    house_locations = np.vstack(
        [
            house_locations,
            i + np.random.normal(scale=0.18, size=[int(NO_HOUSES / NO_CLUSTERS), 2]),
        ]
    )

np.save("house_locations.npy", house_locations, allow_pickle=True)

# Create hilbert encoder:
hc = HilbertCoder(house_locations)
with open("hilbertcoder.pkl", "wb") as fs:
    pickle.dump(hc, fs)

