"""
Module :mod:`spectra_simulate._validated_conversion`

This module offers functions to convert and validate physical quantities like

- temperature
- pressure
- concentrations
- pathlengths

to their target units.

"""

# === Imports ===

from typing import Any, Literal, Optional

from .._utils import RealNumeric, get_checked_real_numeric

# === Constants ===

# the pressure conversion factors
# mbar to bar
_MBAR_TO_BAR = 1e-3
# hPa to bar
_HPA_TO_BAR = 1e-3
# kPa to bar
_KPA_TO_BAR = 1e-2
# atm to bar
_ATM_TO_BAR = 1.01325

# the concentration conversion factors
# ppmv to dimensionless
_PPMV_TO_DIMENSIONLESS = 1e-6
# percent to dimensionless
_PERCENT_TO_DIMENSIONLESS = 1e-2

# the pathlength conversion factors
# meter to centimeter
_METERS_TO_CENTIMETERS = 1e2
# decimeter to centimeter
_DECIMETERS_TO_CENTIMETERS = 1e1
# mm to cm
_MILLIMETERS_TO_CENTIMETERS = 1e-1
# µm to cm
_MICROMETERS_TO_CENTIMETERS = 1e-4

# === Types ===

# the temperature unit
TemperatureUnit = Literal["K", "k", "C", "c", "F", "f"]
# the pressure unit
PressureUnit = Literal["bar", "mbar", "hPa", "kPa", "atm"]
# the concentration unit for gases
GasConcentrationUnit = Literal["ppmv", "percent", "%", "dimensionless"]
# the pathlength unit
PathlengthUnit = Literal["m", "dm", "cm", "mm", "µm"]

# ===  Temperature ===


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


def get_checked_temperature_kelvin(
    temperature: Any,
    temperature_unit: TemperatureUnit,
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
    if temperature_unit_internal == "c":
        temperature = _temperature_celsius_to_kelvin(temperature=temperature)
    elif temperature_unit_internal == "f":
        temperature = _temperature_fahrenheit_to_kelvin(temperature=temperature)

    # if the temperature is below absolute zero, an error is raised
    if temperature < 0.0:
        raise ValueError(
            f"The temperature of {temperature} K is below absolute zero (0 K)."
        )

    return temperature


# === Pressure ===


def get_checked_pressure_bar(
    pressure: Any,
    pressure_unit: PressureUnit,
) -> float:
    """
    Checks the pressure and converts it to bar if necessary.

    """

    # the pressure and pressure unit are checked
    pressure = get_checked_real_numeric(value=pressure)
    pressure_unit_internal = pressure_unit.lower()
    if pressure_unit_internal not in {"bar", "mbar", "hpa", "kpa", "atm"}:
        raise ValueError(
            f"Expected pressure_unit to be one of 'bar', 'mbar', 'hPa', 'kPa', 'atm', "
            f"but got {pressure_unit}."
        )

    # if required, the pressure is converted to bar
    if pressure_unit_internal == "bar":
        return pressure

    return {
        "mbar": _MBAR_TO_BAR,
        "hpa": _HPA_TO_BAR,
        "kpa": _KPA_TO_BAR,
        "atm": _ATM_TO_BAR,
    }[pressure_unit_internal] * pressure


# === Concentrations ===


def get_validated_gas_mole_fraction(
    mole_fraction: Any,
    mole_fraction_unit: Optional[GasConcentrationUnit],
) -> float:
    """
    Checks the mole fraction of a gas and converts it to a dimensionless value if
    necessary.

    """

    # the mole fraction and mole fraction unit are checked
    mole_fraction = get_checked_real_numeric(value=mole_fraction)
    if mole_fraction_unit is not None:
        mole_fraction_unit_internal = mole_fraction_unit.lower()
    else:
        mole_fraction_unit_internal = "dimensionless"

    if mole_fraction_unit_internal not in {"ppmv", "percent", "%", "dimensionless"}:
        raise ValueError(
            f"Expected mole_fraction_unit to be one of 'ppmv', 'percent', '%', "
            f"'dimensionless', or None, but got {mole_fraction_unit}."
        )

    # if required, the mole fraction is converted to a dimensionless value
    if mole_fraction_unit_internal == "dimensionless":
        return mole_fraction

    return {
        "ppmv": _PPMV_TO_DIMENSIONLESS,
        "percent": _PERCENT_TO_DIMENSIONLESS,
        "%": _PERCENT_TO_DIMENSIONLESS,
    }[mole_fraction_unit_internal] * mole_fraction


# === Pathlengths ===


def get_checked_path_length_centimeters(
    path_length: Any,
    path_length_unit: PathlengthUnit,
) -> float:
    """
    Checks the pathlength and converts it to centimeters if necessary.

    """

    # the pathlength and pathlength unit are checked
    path_length = get_checked_real_numeric(value=path_length)
    pathlength_unit_internal = path_length_unit.lower()
    if pathlength_unit_internal not in {"m", "dm", "cm", "mm", "µm"}:
        raise ValueError(
            f"Expected pathlength_unit to be one of 'm', 'dm', 'cm', 'mm', 'µm', but "
            f"got {path_length_unit}."
        )

    # if required, the pathlength is converted to centimeters
    if pathlength_unit_internal == "cm":
        return path_length

    return {
        "m": _METERS_TO_CENTIMETERS,
        "dm": _DECIMETERS_TO_CENTIMETERS,
        "mm": _MILLIMETERS_TO_CENTIMETERS,
        "µm": _MICROMETERS_TO_CENTIMETERS,
    }[pathlength_unit_internal] * path_length
