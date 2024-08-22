"""
Package ``pyftir``

A Python package for Fourier Transform Spectroscopy in Python.

"""

# === Imports ===

import os as _os

from .test_data import black_body_peak, black_body_spectrum  # noqa: F401

# === Package Metadata ===

_AUTHOR_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "AUTHORS.txt")
_VERSION_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "VERSION.txt")

with open(_AUTHOR_FILE_PATH, "r") as author_file:
    __author__ = author_file.read().strip()

with open(_VERSION_FILE_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
