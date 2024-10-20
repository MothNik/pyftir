"""
Module :mod:`augment.extrapolate._extrapolate`

This module contains the actual functions to extrapolate signals beyond their original
range that include input validation and implementation selection.

Currently, the following extrapolation methods are available:

- Burg's method for autoregressive model estimation

"""

# === Setup ===

__all__ = [
    "arburg",
    "extrapolate_autoregressive",
]

# === Imports ===

from typing import Optional, Tuple
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from ..._utils import (
    Integer,
    RealNumeric,
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)
from ._numba_base import numba_arburg_fast, numba_extrapolate_autoregressive
from ._numpy_base import arburg_fast as numpy_arburg_fast
from ._numpy_base import extrapolate_autoregressive as numpy_extrapolate_autoregressive

# === Functions ===

def arburg(
    x: ArrayLike,
    order: Integer = 1,
    tikhonov_lambda: Optional[RealNumeric] = None,
    jit: bool = True,
) -> np.ndarray:
    """
    Computes the AR coefficients for an autoregressive model using a fast implementation
    of Burg's method that relies on an implicit matrix formulation that even allows for
    Tikhonov regularisation.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The real input signal for which the AR coefficients are to be computed.
        Its data type is internally promoted to ``numpy.float64``.
        It has to hold at least ``2`` elements.
    order : :class:`int`, default=``1``
        The order of the autoregressive model.
        It has to be within the range ``[1, len(x) - 1]``.
    tikhonov_lambda : :class:`float` or :class:`int` or ``None``, default=``None``
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.
        ``None`` is equivalent to ``0.0``.
    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.

    Returns
    -------
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,)
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.

    Raises
    ------
    TypeError
        If ``x``, ``order``, or ``tikhonov_lambda`` are not of the expected type.
    ValueError
        If ``x`` is not a 1D Array-like or not of expected size.
    ValueError
        If ``order`` is not within the allowed range.

    References
    ----------
    The implementation is based on the pseudo-code provided in [1]_.

    .. [1] Vos K., A Fast Implementation of Burg's Method (2013)

    """

    # --- Input Validation ---

    x_internal = get_validated_real_numeric_1d_array_like(
        value=x,
        name="x",
        min_size=2,
        max_size=None,
        output_dtype=np.float64,
    )
    order = get_validated_integer(
        value=order,
        name="order",
        min_value=1,
        max_value=x_internal.size - 1,
    )

    tikhonov_lambda = get_validated_real_numeric(
        value=tikhonov_lambda if tikhonov_lambda is not None else 0.0,
        name="tikhonov_lambda",
    )

    # --- Computation ---

    # depending on the choice of the user, the Numba-accelerated or the NumPy-based
    # implementation is used
    arburg_func = numba_arburg_fast if jit else numpy_arburg_fast
    return arburg_func(  # type: ignore
        x=x_internal,  # type: ignore
        order=order,  # type: ignore
        tikhonov_lambda=tikhonov_lambda,  # type: ignore
    )


def extrapolate_autoregressive(
    x: ArrayLike,
    ar_coeffs: ArrayLike,
    pad_width: Tuple[Integer, Integer] = (0, 0),
    jit: bool = True,
    zero_lag_warn: bool = True,
) -> np.ndarray:
    """
    Extrapolates a signal beyond its original range using the coefficients of an
    autoregressive model.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The real input signal to be extrapolated.
        It is internally promoted to ``numpy.float64``.
        Its length has to be at least ``2``.
    ar_coeffs : Array-like of shape (order + 1,)
        The AR coefficients of the autoregressive model.
        There have to be at least ``2`` (AR(1) model) and at most ``len(x) - 1``
        coefficients.
        They are internally promoted to ``numpy.float64``.
        The zero-lag coefficient ``ar_coeffs[0]`` is expected to be present. In case
        this coefficient is not exactly equal to ``1.0``, all coefficients are
        normalised by this value and a warning is issued (see ``zero_lag_warn``).
    pad_width : (``int``, ``int``), default=``(0, 0)``
        The size of the extrapolation on the left and right side of the input signal,
        respectively.
        Negative values are silently clipped to ``0``, which means that no extrapolation
        is performed on the respective side.
    jit : :class:`bool`, default=``True``
        Whether to use the Numba-accelerated implementation (``True``) or the
        NumPy-based implementation (``False``).
        If Numba is not available, the function silently falls back to the NumPy-based
        implementation.
    zero_lag_warn : :class:`bool`, default=``True``
        Whether to issue a warning if the zero-lag coefficient of the AR model is not
        exactly equal to ``1.0`` (``True``) or not (``False``).

    Returns
    -------
    x_extrapolated : :class:`numpy.ndarray` of shape (n + pad_left + pad_right,)
        The extrapolated signal.

    Raises
    ------
    TypeError
        If ``x``, ``ar_coeffs``, or ``pad_width`` are not of the expected type.
    ValueError
        If ``x`` or ``ar_coeffs`` are not 1D Array-like.
    ValueError
        If ``x`` or ``ar_coeffs`` are not of expected size.

    """

    # --- Input Validation ---

    x_internal = get_validated_real_numeric_1d_array_like(
        value=x,
        name="x",
        min_size=2,
        max_size=None,
        output_dtype=np.float64,
    )
    ar_coeffs_internal = get_validated_real_numeric_1d_array_like(
        value=ar_coeffs,
        name="ar_coeffs",
        min_size=2,
        max_size=x_internal.size - 1,
        output_dtype=np.float64,
    )

    pad_width_internal = tuple(
        get_validated_integer(
            value=value,
            name=f"pad_width[{index}]",
            min_value=0,
            max_value=None,
            clip=True,
        )
        for index, value in enumerate(pad_width)
    )

    # --- Computation ---

    # if the padding is zero, the extrapolation is equivalent to the original signal
    if pad_width_internal == (0, 0):
        return x_internal

    # if the zero-lag coefficient is not exactly 1.0, a scaling is performed and a
    # warning is issued if requested
    if ar_coeffs_internal[0] != 1.0:
        ar_coeffs_internal = ar_coeffs_internal / ar_coeffs_internal[0]
        if zero_lag_warn:
            warn(
                f"The zero-lag coefficient of the AR model is not exactly 1.0, but "
                f"{ar_coeffs_internal[0]:.5e}.\n"
                f"All coefficients are normalised by this value.\n"
                f"This warning can be suppressed by setting 'zero_lag_warn=False'.",
                RuntimeWarning,
            )

    # the Numba-accelerated or the NumPy-based implementation is used depending on the
    # user's choice
    extrapolation_func = (
        numba_extrapolate_autoregressive if jit else numpy_extrapolate_autoregressive
    )
    return extrapolation_func(  # type: ignore
        x=x_internal,  # type: ignore
        ar_coeffs=ar_coeffs_internal,  # type: ignore
        pad_width_left=pad_width_internal[0],  # type: ignore
        pad_width_right=pad_width_internal[1],  # type: ignore
    )
