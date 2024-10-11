"""
Module :mod:`augment.extrapolate`

This module provides functions for extrapolating signals beyond their original range.

Currently, the following methods are implemented:

- Burg's method for autoregressive model estimation

"""

# === Imports ===

from ._extrapolate import arburg, extrapolate_autoregressive  # noqa: F401
