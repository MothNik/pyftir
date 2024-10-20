"""
This test suite implements all tests for the autoregressive model estimation via the
Burg method in the module :mod:`pyscopee.augment.extrapolate._numpy_base`.

"""

# === Imports ===

import numpy as np
import pytest

from pyscopee.augment.extrapolate._numpy_base import arburg_fast

def arburg_slow(
    xs: np.ndarray,
    x_lens: np.ndarray,
    order: int,
) -> np.ndarray:
    def get_x_matrix(x: np.ndarray, order: int) -> np.ndarray:
        x_matrix = np.empty(shape=(x.size - order, order + 1), dtype=np.float64)
        for iter_i in range(0, order + 1):
            x_matrix[::, order - iter_i] = x[iter_i : x.size - order + iter_i]

        return x_matrix

    a = np.array([1.0])
    for iter_ord in range(0, order):
        mat_j = np.flip(np.eye(iter_ord + 2), axis=1)
        r_matrix = np.zeros(shape=(iter_ord + 2, iter_ord + 2))
        for iter_i, num_elements in enumerate(x_lens):
            x = xs[iter_i, 0:num_elements]
            x_matrix = get_x_matrix(x, iter_ord + 1)
            r_matrix += mat_j @ x_matrix.T @ x_matrix @ mat_j + x_matrix.T @ x_matrix

        k_reflect = -(
            np.append(a, 0.0)
            @ mat_j
            @ r_matrix
            @ np.append(a, 0.0)
            / (np.append(a, 0.0) @ r_matrix @ np.append(a, 0.0))
        )

        a = np.append(a, 0.0) + k_reflect * np.flip(np.append(a, 0.0))
        print(iter_ord, k_reflect, a.tolist())

    return a