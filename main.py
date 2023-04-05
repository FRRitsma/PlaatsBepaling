# %%
# Run problem formulation
import problem.problem
from hilbert.hilbert import HilbertCoder
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as Tree

# Load hilbert coder:
with open('hilbertcoder.pkl', 'rb') as fs:
    hilbertcoder = pickle.load(fs)

# Load houses:
house_locations = np.load('house_locations.npy')
# Random msr location:
msr_locations = np.array([[.4,.6],[.6,.7], [.8,.5]])
