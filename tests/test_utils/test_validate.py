"""
This test suite implements all tests for the module :mod:`pyftir._utils.validate`.

"""

# === Imports ===

from math import isclose as pyisclose
from typing import Any, Optional, Union

import numpy as np
import pytest

from pyftir._utils import get_validated_integer, get_validated_real_numeric

# === Tests ===


@pytest.mark.parametrize(
    "value, min_value, max_value, clip, expected",
    [
        (  # 0) a Python integer without any constraints
            1,
            None,
            None,
            False,
            1,
        ),
        (  # 1) a NumPy integer without any constraints
            np.int64(1),
            None,
            None,
            False,
            1,
        ),
        (  # 2) a float value without any constraints
            1.0,
            None,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 3) a Python integer with a minimum constraint that is satisfied
            1,
            0,
            None,
            False,
            1,
        ),
        (  # 4) a Python integer with a minimum constraint that leads to clipping
            1,
            2,
            None,
            True,
            2,
        ),
        (  # 5) a Python integer with a minimum constraint that leads to an exception
            1,
            2,
            None,
            False,
            ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        (  # 6) a Python integer with a maximum constraint that is satisfied
            1,
            None,
            2,
            False,
            1,
        ),
        (  # 7) a Python integer with a maximum constraint that leads to clipping
            3,
            None,
            2,
            True,
            2,
        ),
        (  # 8) a Python integer with a maximum constraint that leads to an exception
            3,
            None,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 9) a Python integer with both constraints that are satisfied
            1,
            0,
            2,
            False,
            1,
        ),
        (  # 10) a Python integer with both constraints that lead to clipping
            0,
            1,
            2,
            True,
            1,
        ),
        (  # 11) a Python integer with both constraints that lead to clipping
            3,
            0,
            2,
            True,
            2,
        ),
        (  # 12) a Python integer with both constraints that lead to an exception
            0,
            1,
            2,
            False,
            ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        (  # 13) a Python integer with both constraints that lead to an exception
            3,
            0,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 14) a Python integer with flipped constraints
            1,
            2,
            0,
            False,
            ValueError(
                "Expected 'min_value' for 'value' to be <= 'max_value', but got "
                "'min_value' = 2 and 'max_value' = 0."
            ),
        ),
        (  # 15) a Python float with a minimum constraint that is satisfied
            1.0,
            0,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 16) a Python float with a maximum constraint that is satisfied
            1.0,
            None,
            2,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 17) a Python float with both constraints that are satisfied
            1.0,
            0,
            2,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 18) a Python float with flipped constraints
            1.0,
            2,
            0,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
    ],
)
def test_integer_validation(
    value: Any,
    min_value: Optional[int],
    max_value: Optional[int],
    clip: bool,
    expected: Union[int, Exception],
) -> None:
    """
    Tests the function :func:`get_validated_integer` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = get_validated_integer(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = get_validated_integer(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert checked_value == expected

    return


@pytest.mark.parametrize(
    "value, min_value, max_value, clip, expected",
    [
        (  # 0) a Python float without any constraints
            1.0,
            None,
            None,
            False,
            1.0,
        ),
        (  # 1) a NumPy float without any constraints
            np.float64(1.0),
            None,
            None,
            False,
            1.0,
        ),
        (  # 2) a Python integer without any constraints
            1,
            None,
            None,
            False,
            1.0,
        ),
        (  # 3) a NumPy integer without any constraints
            np.int64(1),
            None,
            None,
            False,
            1.0,
        ),
        (  # 4) a complex value without any constraints
            1.0 + 1.0j,
            None,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
        (  # 5) a Python float with a minimum constraint that is satisfied
            1.0,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 6) a Python float with a minimum constraint that leads to clipping
            1.0,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 7) a Python float with a minimum constraint that leads to an exception
            1.0,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1.0."),
        ),
        (  # 8) a Python integer with a minimum constraint that is satisfied
            1,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 9) a Python integer with a minimum constraint that leads to clipping
            1,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 10) a Python integer with a minimum constraint that leads to an exception
            1,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1."),
        ),
        (  # 11) a Python float with a maximum constraint that is satisfied
            1.0,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 12) a Python float with a maximum constraint that leads to clipping
            3.0,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 13) a Python float with a maximum constraint that leads to an exception
            3.0,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 14) a Python integer with a maximum constraint that is satisfied
            1,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 15) a Python integer with a maximum constraint that leads to clipping
            3,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 16) a Python integer with a maximum constraint that leads to an exception
            3,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 17) a Python float with both constraints that are satisfied
            1.0,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 18) a Python float with both constraints that lead to clipping
            0.0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 19) a Python float with both constraints that lead to clipping
            3.0,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 20) a Python float with both constraints that lead to an exception
            0.0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0.0."),
        ),
        (  # 21) a Python float with both constraints that lead to an exception
            3.0,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 22) a Python integer with both constraints that are satisfied
            1,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 23) a Python integer with both constraints that lead to clipping
            0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 24) a Python integer with both constraints that lead to clipping
            3,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 25) a Python integer with both constraints that lead to an exception
            0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0."),
        ),
        (  # 26) a Python integer with both constraints that lead to an exception
            3,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 27) a Python float with flipped constraints
            1.0,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected 'min_value' for 'value' to be <= 'max_value', but got "
                "'min_value' = 2.0 and 'max_value' = 0.0."
            ),
        ),
        (  # 28) a Python integer with flipped constraints
            1,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected 'min_value' for 'value' to be <= 'max_value', but got "
                "'min_value' = 2.0 and 'max_value' = 0.0."
            ),
        ),
    ],
)
def test_real_numeric_validation(
    value: Any,
    min_value: Optional[float],
    max_value: Optional[float],
    clip: bool,
    expected: Union[float, Exception],
) -> None:
    """
    Tests the function :func:`get_validated_real_numeric` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = get_validated_real_numeric(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = get_validated_real_numeric(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert pyisclose(
        checked_value,
        expected,
        abs_tol=1e-15,
        rel_tol=1e-15,
    )

    return
