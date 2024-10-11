"""
Mod :mod:`_utils.types`

This module provides type definitions used across the ``pyftir`` package.

"""

# === Imports ===

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

# === Type Definitions ===

# a real numeric value
RealNumeric = Union[int, float, np.integer, np.floating]
Integer = Union[int, np.integer]

# a real numeric Arraylike
RealNumericArrayLike = Union[RealNumeric, ArrayLike]
