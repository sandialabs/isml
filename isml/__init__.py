"""Experiments in In-Situ Machine Learning (ISML)

"""

import sys

if sys.version_info.major * 10 + sys.version_info.minor < 37:
    raise RuntimeError("Python >= 3.7 is required.")

__version__ = "0.1.0-dev"

import isml.data as data
import isml.netcdf_data as netcdf_data
import isml.decision as decision
import isml.feature as feature
import isml.measure as measure
import isml.signature as signature

