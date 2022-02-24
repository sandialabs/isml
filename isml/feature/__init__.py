"""Functionality for working with features.

"""

import numpy

def matrix(array):
    """Flatten a feature array into a matrix.

    Treating the feature array as a matrix loses spatial information, but makes
    it easier to work with tools (like Scikit Learn) that expect matrices as
    inputs.

    Parameters
    ----------
    array: :class:`numpy.ndarray` with two-or-more dimensions
        The array to be flattened.

    Returns
    -------
    matrix: :class:`numpy.ndarray` with two dimensions
    """
    return array.reshape((-1, array.shape[-1]))

import isml.feature.preprocess as preprocess

