"""
Mod :mod:`test_data.black_body`

This module implements a Planck blackbody radiation function with the wavenumbers
as the independent variable.

"""

# === Imports ===

from typing import Any, Literal, Tuple, Union, overload

import numpy as np

from .._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_checked_real_numeric,
    get_checked_real_numeric_1d_array_like,
)

# === Constants ===

# the speed of light in m / s
_SPEED_OF_LIGHT = 299_792_45_8

# the Planck constant in J * s
_PLANCK_CONSTANT = 6.626_070_150e-34

# the Boltzmann constant in J / K
_BOLTZMANN_CONSTANT = 1.380_649e-23

# the peak wavenumber slope
_PEAK_WAVENUMBER_SLOPE = 2.821439372  # dimensionless


# === Auxiliary Functions ===


def _get_checked_kelvin_temperature(
    temperature: Any,
    temperature_unit: Literal["K", "k", "C", "c", "F", "f"],
) -> float:
    """
    Checks the temperature and converts it to Kelvin if necessary.

    """

    # the temperature and temperature unit are checked
    temperature = get_checked_real_numeric(value=temperature)
    temperature_unit_internal = temperature_unit.lower()
    if temperature_unit_internal not in {"k", "c", "f"}:
        raise ValueError(
            f"Expected temperature_unit to be one of 'K', 'k', 'C', 'c', 'F', 'f', but "
            f"got {temperature_unit}."
        )

    # if required, the temperature is converted to Kelvin
    if temperature_unit == "c":
        temperature = _temperature_celsius_to_kelvin(temperature=temperature)
    elif temperature_unit == "f":
        temperature = _temperature_fahrenheit_to_kelvin(temperature=temperature)

    # if the temperature is below absolute zero, an error is raised
    if temperature < 0.0:
        raise ValueError(
            f"The temperature of {temperature} K is below absolute zero (0 K)."
        )

    return temperature


# === Functions ===


def _temperature_celsius_to_kelvin(temperature: RealNumeric) -> float:
    """
    Converts a temperature from °C to K.

    """

    return float(temperature) + 273.15


def _temperature_fahrenheit_to_kelvin(temperature: RealNumeric) -> float:
    """
    Converts a temperature from °F to K.

    """

    celsius_temperature = (float(temperature) - 32.0) * (5.0 / 9.0)
    return _temperature_celsius_to_kelvin(temperature=celsius_temperature)


@overload
def black_body_spectrum(
    wavenumbers: RealNumeric,
    temperature: RealNumeric,
    temperature_unit: Literal["K", "k", "C", "c", "F", "f"] = "K",
) -> float: ...


@overload
def black_body_spectrum(
    wavenumbers: RealNumericArrayLike,
    temperature: RealNumeric,
    temperature_unit: Literal["K", "k", "C", "c", "F", "f"] = "K",
) -> np.ndarray: ...


def black_body_spectrum(
    wavenumbers: RealNumericArrayLike,
    temperature: RealNumeric,
    temperature_unit: Literal["K", "k", "C", "c", "F", "f"] = "K",
) -> Union[float, np.ndarray]:
    """
    Computes the Planck blackbody radiation spectrum for a given temperature and
    wavenumbers.

    Parameters
    ----------
    wavenumbers : class:`int` or :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The wavenumber(s) in 1 / cm.
    temperature : class:`int` or :class:`float`
        The temperature given in the specified ``temperature_unit``.
    temperature_unit : {``"K"``, ``"k"``, ``"C"``, ``"c"``, ``"F"``, ``"f"``}, default=``"K"``
        The unit of the ``temperature`` which can be:

        - ``"K"`` or ``"k"``: Kelvin
        - ``"C"`` or ``"c"``: degree Celsius
        - ``"F"`` or ``"f"``: degree Fahrenheit

    Returns
    -------
    black_body_spectrum : :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The blackbody radiation spectrum in W * cm / (m² * sr) evaluated at the
        specified ``wavenumbers``.
        It will be a scalar if the input ``wavenumbers`` was a scalar and a NumPy
        Array otherwise.

    Raises
    ------
    ValueError
        If the ``temperature_unit`` is not one of the allowed values.
    ValueError
        If the ``temperature`` is below absolute zero after conversion to Kelvin.

    References
    ----------
    The equation was taken from [1]_ and [2]_.

    .. [1] Wikipedia, Planck's Law - Different forms,
       URL: en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    .. [2] SpectralCalc.com, Calculation of Blackbody Radiance,
       URL: www.spectralcalc.com/blackbody/blackbody.html

    """  # noqa: E501

    # --- Input Validation ---

    # the wavenumbers are checked and converted to a 1D NumPy array
    wavenumbers = get_checked_real_numeric_1d_array_like(value=wavenumbers)

    # then, the temperature is converted to Kelvin
    temperature = _get_checked_kelvin_temperature(
        temperature=temperature,
        temperature_unit=temperature_unit,
    )

    # --- Computation ---

    black_body_spectrum = (
        (2.0 * 1e8 * _PLANCK_CONSTANT * _SPEED_OF_LIGHT * _SPEED_OF_LIGHT)
        * wavenumbers
        * wavenumbers
        * wavenumbers
    )

    black_body_spectrum /= (
        np.exp(
            (
                (_PLANCK_CONSTANT * _SPEED_OF_LIGHT * 100.0)
                / (_BOLTZMANN_CONSTANT * temperature)
            )
            * wavenumbers
        )
        - 1.0
    )

    # if the wavenumbers were a scalar, the result is converted to a scalar
    if np.isscalar(wavenumbers):
        return float(black_body_spectrum)

    # for Array-like wavenumbers, the result is returned as a NumPy Array
    return black_body_spectrum


def black_body_peak(
    temperature: RealNumeric,
    temperature_unit: Literal["K", "k", "C", "c", "F", "f"] = "K",
) -> Tuple[float, float]:
    """
    Computes the wavenumber at which the Planck blackbody radiation spectrum peaks
    for a given temperature as well as the peak intensity.

    Parameters
    ----------
    temperature : class:`int` or :class:`float`
        The temperature given in the specified ``temperature_unit``.
    temperature_unit : {``"K"``, ``"k"``, ``"C"``, ``"c"``, ``"F"``, ``"f"``}, default=``"K"``
        The unit of the ``temperature`` which can be:

        - ``"K"`` or ``"k"``: Kelvin
        - ``"C"`` or ``"c"``: degree Celsius
        - ``"F"`` or ``"f"``: degree Fahrenheit

    Returns
    -------
    black_body_peak_wavenumber : :class:`float`
        The wavenumber in 1 / cm at which the blackbody radiation spectrum peaks.
    black_body_peak_intensity : :class:`float`
        The peak intensity in W * cm / (m² * sr) of the blackbody radiation spectrum.

    Raises
    ------
    ValueError
        If the ``temperature_unit`` is not one of the allowed values.
    ValueError
        If the ``temperature`` is below absolute zero after conversion to Kelvin.

    References
    ----------
    The equation was taken from [1]_ and [2]_.

    .. [1] Wikipedia, Planck's Law - Different forms,
       URL: en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    .. [2] SpectralCalc.com, Calculation of Blackbody Radiance,
       URL: www.spectralcalc.com/blackbody/blackbody.html

    """  # noqa: E501

    # --- Input Validation ---

    # the temperature is converted to Kelvin
    temperature = _get_checked_kelvin_temperature(
        temperature=temperature,
        temperature_unit=temperature_unit,
    )

    # --- Computation ---

    black_body_peak_wavenumber = (
        _PEAK_WAVENUMBER_SLOPE
        * _BOLTZMANN_CONSTANT
        * temperature
        / (100.0 * _PLANCK_CONSTANT * _SPEED_OF_LIGHT)
    )

    black_body_peak_intensity = black_body_spectrum(
        wavenumbers=black_body_peak_wavenumber,
        temperature=temperature,
        temperature_unit="K",
    )

    return black_body_peak_wavenumber, black_body_peak_intensity
