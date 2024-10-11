"""
Module :mod:`_utils.numba_helpers`

This module implements auxiliary functionalities to handle Numba-related tasks, such as

- checking whether Numba ``jit``-compilation has been explicitly specified to take no
    effect, e.g., for test coverage

"""

# === Imports ===

import os
from enum import Enum
from typing import Callable

# === Models ===

# an Enum that specifies the possible actions that can be taken regarding Numba
# ``jit``-compilation


class NumbaJitActions(Enum):
    """
    Specifies the possible actions that can be taken regarding Numba
    ``jit``-compilation.

    """

    NORMAL = "0"
    DEACTIVATE = "1"


# === Constants ===

# the runtime argument that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ARGV = "--no-jit"

# the environment variable that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ENV_KEY = "CUSTOM_NUMBA_NO_JIT"


# whether the environment variable is set to specify that Numba ``jit``-compilation
# should take effect or not in the current runtime environment
do_numba_normal_jit_action = (
    os.environ.get(NUMBA_NO_JIT_ENV_KEY, NumbaJitActions.NORMAL.value)
    == NumbaJitActions.NORMAL.value
)


# === Functions ===


def no_jit(*args, **kwargs) -> Callable:
    """
    Fake decorator that can be used to make sure that Numba ``jit``-compilation has no
    effect.

    Parameters
    ----------
    func : :class:`Callable`
        The function that is decorated.
    args : :class:`tuple`
        The fake positional arguments.
    kwargs : :class:`dict`
        The fake keyword arguments.

    Returns
    -------
    decorated_func : :class:`Callable`
        The decorated function.

    """

    def decorator(func: Callable) -> Callable:
        return func

    return decorator
