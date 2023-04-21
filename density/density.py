# %%
"""
Generating a function/class for estimating the density of a given state.

Eventual goal is to prioritize states with high evaluation in low density areas
"""

# TODO: Improve axis selection

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class DensityTree:
    # Mandatory:
    min_leaf_size: int

    # Optional arguments:
    volume: float = field(default=1)
    min_by_axis: np.ndarray = field(default=None)
    max_by_axis: np.ndarray = field(default=None)

    # Derived, but with default:
    is_terminal: bool = field(init=False, default=True)

    # Derived in fit:
    axis: int = field(init=False, default=None)
    threshold: float = field(init=False, default=None)
    small_branch: DensityTree = field(init=False, default=None)
    big_branch: DensityTree = field(init=False, default=None)
    dif_by_axis: np.ndarray = field(init=False, default=None)
    output_value: np.float64 = field(init=False, default=None)

    def __post_init__(self):
        pass

    # Fitting the current branch
    def fit(self, data: np.ndarray) -> None:
        assert data.ndim == 2, f"Data must be 2-dimensional, got ndim = {data.ndim}"

        # If min_leaf_size is reached:
        n_samples = data.shape[0]
        if n_samples < self.min_leaf_size:
            self.output_value = np.ones(1) * (n_samples / self.volume)
            return

        # Set as non-terminal:
        self.is_terminal = False

        # Initialize self values:
        self.__fit_init(data)
        axis = self.get_axis(data)
        self.axis = axis

        # Get threshold:
        column = data[:, self.axis]
        threshold = np.median(column)
        self.threshold = threshold

        # Initialize big branch:
        big_mask = column > threshold
        big_volume = (
            (self.max_by_axis[axis] - threshold) / self.dif_by_axis[axis]
        ) * self.volume
        big_max_by_axis, big_min_by_axis = (
            self.max_by_axis.copy(),
            self.min_by_axis.copy(),
        )
        big_min_by_axis[axis] = threshold
        self.big_branch = DensityTree(
            self.min_leaf_size,
            volume=big_volume,
            min_by_axis=big_min_by_axis,
            max_by_axis=big_max_by_axis,
        )
        self.big_branch.fit(data[big_mask, :])

        # Initialize small branch:
        small_mask = column <= threshold
        small_volume = (
            (threshold - self.min_by_axis[axis]) / self.dif_by_axis[axis]
        ) * self.volume        
        small_max_by_axis, small_min_by_axis = (
            self.max_by_axis.copy(),
            self.min_by_axis.copy(),
        )
        small_max_by_axis[axis] = threshold
        self.small_branch = DensityTree(
            self.min_leaf_size,
            volume=small_volume,
            min_by_axis=small_min_by_axis,
            max_by_axis=small_max_by_axis,
        )
        self.small_branch.fit(data[small_mask, :])

        

    def __fit_init(self, data: np.ndarray) -> None:
        # Initialize min/max if not given:
        if self.min_by_axis is None:
            self.min_by_axis = np.min(data, axis=0)
        if self.max_by_axis is None:
            self.max_by_axis = np.max(data, axis=0)
        self.dif_by_axis = self.max_by_axis - self.min_by_axis

    @staticmethod
    def get_axis(data: np.ndarray) -> int:
        # TODO: make some better way of choosing axis:
        return np.random.choice(np.arange(data.shape[1]))

    # Predicting for new input
    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.is_terminal:
            return self.output_value*np.ones([data.shape[0]])
        
        # Recursive call: 
        output_array = np.empty(data.shape[0])
        big_mask = data[:, self.axis] > self.threshold
        small_mask = data[:, self.axis] <= self.threshold
        big_output = self.big_branch.predict(data[big_mask,:])
        small_output = self.small_branch.predict(data[small_mask,:])
        output_array[big_mask] = big_output
        output_array[small_mask] = small_output

        return output_array


#  Test and Develop:
np.random.seed(1)

X = np.random.normal(loc=np.ones([5000, 2]))
plt.scatter(X[:, 0], X[:, 1])

xx, yy = np.meshgrid(np.linspace(-2,4,100), np.linspace(-2,4,100))
xx, yy = xx.reshape(-1,1), yy.reshape(-1,1)
xy = np.hstack((xx, yy))


dt = DensityTree(10)
dt.fit(X)
y = dt.predict(xy)

plt.scatter(xy[:,0], xy[:,1], c=y)

# %%