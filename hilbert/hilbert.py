import numpy as np
from numpy import matlib
from sklearn.neighbors import KDTree as Tree

"""
This module implements the functions for creating the HilbertCoder. 
This can be used to encode a two dimensional point (i.e., location) to a one dimensional space.
"""


def hilbert_curve_iteration(base_array: np.ndarray) -> np.ndarray:
    max_val = np.max(base_array) + 1
    return np.hstack(
        [
            np.vstack(
                [
                    base_array + max_val,
                    base_array[::-1, ::-1].T,
                ]
            ),
            np.vstack(
                [
                    base_array + max_val * 2,
                    base_array.T + max_val * 3,
                ]
            ),
        ]
    )


def hilbert_iterative(iters: int):
    base_array = np.array([[1, 2], [0, 3]], dtype=np.uint32)
    for _ in range(iters):
        base_array = hilbert_curve_iteration(base_array)
    return base_array


class HilbertCoder:
    """
    For encoding a two dimensional position two a one dimensional line (domain: [0,1]) and back again.
    """
    def __init__(
        self,
        array: np.ndarray = None,
        # Domain of point to be encoded:
        xmin: float = None,
        xmax: float = None,
        ymin: float = None,
        ymax: float = None,
        # Granularity of encoding:
        granularity: int = 8,
    ) -> None:
        if array is not None:
            xmin, ymin = np.min(array, axis=0)
            xmax, ymax = np.max(array, axis=0)
        # TODO: Perform Assertion checks:
        else:
            assert xmin is not None, "Provide either array or xmin/xmax/ymin/ymax"

        # Create scale:
        minn = np.array([xmin, ymin])
        maxx = np.array([xmax, ymax])

        # Create hilbert array:
        hilbert_array = hilbert_iterative(granularity)
        bit = hilbert_array.shape[0]

        values = hilbert_array.reshape(-1, 1)
        coords = np.matlib.repmat(np.arange(bit), bit, 1)
        coords = np.hstack(
            (
                coords.reshape(-1, 1),
                np.rot90(coords).reshape(-1, 1),
            )
        )

        # Scale values:
        values = values / np.max(values)
        coords = (coords - coords.min(axis=0)) / (
            coords.max(axis=0) - coords.min(axis=0)
        )
        coords = coords * (maxx - minn) + minn

        # Create and self assing trees:
        self.decoding_tree = Tree(values.reshape(-1, 1))
        self.encoding_tree = Tree(coords)
        self.coords = coords
        self.values = values

    def encoding(self, points: np.ndarray) -> np.ndarray:
        _, ind = self.encoding_tree.query(points)
        return self.values[ind].reshape(-1)

    def decoding(self, encoding: np.ndarray) -> np.ndarray:
        if encoding.ndim == 1:
            _, ind = self.decoding_tree.query(encoding.reshape(-1, 1))
        if encoding.ndim == 2:
            return [self.decoding(e) for e in encoding]

        return self.coords[ind].reshape(len(encoding), -1)
