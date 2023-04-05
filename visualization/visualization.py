import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as Tree
import numpy as np

def display_houses_facilities(facility_locations: np.ndarray, house_locations: np.ndarray):
    tree = Tree(facility_locations)
    _, assign = tree.query(house_locations)

    plt.scatter(house_locations[:,0], house_locations[:,1], c = assign, s = 1)
    plt.scatter(facility_locations[:,0], facility_locations[:,1])
    plt.show()
