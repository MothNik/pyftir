"""
Mod :mod:`_utils.types`

This module provides type definitions used across the ``pyftir`` package.

"""

# === Imports ===

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

# === Type Definitions ===

# a real numeric value
RealNumeric = Union[int, float, np.integer, np.floating]

# a real numeric Arraylike
RealNumericArrayLike = Union[RealNumeric, ArrayLike]


# === Functions ===


def get_checked_real_numeric(value: Any) -> float:
    """
    Checks if a value is a real numeric and returns it as a float.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    checked_value : :class:`float`
        The checked value as a float.

    Raises
    ------
    TypeError
        If the value is not a real numeric.

    """

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"Expected a real numeric value, got {type(value)}.")

    return float(value)


def get_checked_real_numeric_1d_array_like(value: Any) -> np.ndarray:
    """
    Checks if a value is a 1D Array-like of real numeric values and returns it as a
    NumPy 1D Array.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    checked_value : :class:`numpy.ndarray` of shape (n, )
        The checked value.

    Raises
    ------
    TypeError
        If the value is not a 1D Array-like of real numeric values.
    ValueError
        If the value is not a 1D Array-like.

    """

    # first, the value is converted to a NumPy array
    value_array = np.atleast_1d(value)

    # then, the value is checked to be a 1D array
    if value_array.ndim != 1:
        raise ValueError(
            f"Expected a 1D Array-like, but got a {value_array.ndim}D Array."
        )

    # afterwards, the value is checked to be a 1D array of real numeric values
    if not np.isreal(value_array).all():
        raise TypeError(
            "Expected a 1D Array-like of real numeric values, but got a 1D Array-like "
            "with non-real numeric values."
        )

    return value_array
