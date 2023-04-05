from itertools import combinations
from hilbert.hilbert import HilbertCoder
import numpy as np
from sklearn.neighbors import KDTree as Tree


# For a given amount of combinations, how many samples are needed:
def n_samples_for_combination(k: int, samples: int):
    n = k + 1
    while True:
        if sum(1 for _ in combinations("_" * n, k)) > samples:
            return n - 1
        n += 1


def loss_function(
    house_locations: np.ndarray,
    facility_locations: np.ndarray,
) -> float:
    """
    Loss function sums up the distance of all houses to the nearest facility and returns it as loss.

    Args:
        house_locations (np.ndarray): Point cloud of house locations
        facility_locations (np.ndarray): Point cloud of facility locations

    Returns:
        float: Summed up nearest distance
    """
    if type(facility_locations) is list:
        return np.array([loss_function(house_locations, f) for f in facility_locations])
    tree = Tree(facility_locations)
    distances, _ = tree.query(house_locations)
    summed_distance = np.sum(distances)
    return summed_distance


# Separate function for explore:
def explore_row(
    row: np.ndarray,
    all_states: np.ndarray,
    idx: int,
) -> np.ndarray:
    
    row = row.copy()
    row = row.ravel()
    # Find upper limit:
    upper_limit = 1
    if idx > 0:
        upper_limit = row.ravel()[idx - 1]
    # Filter all states:
    state_set = all_states[all_states < upper_limit]
    # Choose random state:
    new_value = state_set[np.random.randint(len(state_set))]
    row[idx] = new_value
    return row


