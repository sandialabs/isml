"""Functionality to perform preprocessing on features.

This module provides functions that preprocess per-partition features before
their conversion to signatures.  These functions all take partitions as inputs
and produce modified partitions as outputs.

To preprocess, we use a Python callable (a function or a class that has a
__call__() method) that takes a list of :math:`P` arrays of shape :math:`\\langle shape
\\rangle \\times F`, one for each partition, where :math:`\\langle shape
\\rangle` is the shape of the simulation grid within a partition and :math:`F`
is the number of features at each grid point.

The callable must return a list of modified partitions, each with the same
shape as the corresponding input.

Note that implementations should not assume that :math:`\\langle shape
\\rangle` is consistent across all partitions, since the local domain might not
permit uniform decompositions.

Also, note that these functions receive and produce *local* partitions (local
to the current process), but implementations are free to perform communication
(e.g. using MPI) as needed.
"""

import logging

import numpy
import sklearn.preprocessing

import isml.feature

log = logging.getLogger(__name__)


def require_valid_partitions(partitions):
    """Sanity-check partitions before preprocessing."""
    if not isinstance(partitions, list):
        raise ValueError("Partitions must be a list.")
    for partition in partitions:
        if not isinstance(partition, numpy.ndarray):
            raise ValueError("Partitions must be a list of numpy arrays.")
        if partition.ndim < 2:
            raise ValueError("Partitions must be a list of numpy arrays with ndim >= 2.")


def identity(partitions):
    """Return partition features without modification.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.feature.preprocess` for details.

    Returns
    -------
    partitions: list of :class:`numpy.ndarray`
        See :mod:`isml.feature.preprocess` for details.
    """
    require_valid_partitions(partitions)
    return partitions


def local_min_max_scaling(partitions, output_range):
    """Scale partition features to the given range, based on the min and max across all partitions.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.feature.preprocess` for details.
    output_range: two-tuple, required
        Minimum and maximum output values.

    Returns
    -------
    partitions: list of :class:`numpy.ndarray`
        See :mod:`isml.feature.preprocess` for details.
    """
    require_valid_partitions(partitions)

    # Identify the current feature domain.
    model = sklearn.preprocessing.MinMaxScaler(output_range)
    for partition in partitions:
        model.partial_fit(isml.feature.matrix(partition))

    # Scale the features to the desired output range.
    scaled_partitions = [model.transform(isml.feature.matrix(partition)) for partition in partitions]
    scaled_partitions = [scaled_partition.reshape(partition.shape) for scaled_partition, partition in zip(scaled_partitions, partitions)]

    return scaled_partitions


def local_max_abs_scaling(partitions):
    """Scale partition features so the maximum absolute value for any feature is 1.0, across all partitions.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    partitions: list of :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)

    # Identify the current feature domain.
    model = sklearn.preprocessing.MaxAbsScaler()
    for partition in partitions:
        model.partial_fit(isml.feature.matrix(partition))

    # Scale the features to the desired output range.
    scaled_partitions = [model.transform(isml.feature.matrix(partition)) for partition in partitions]
    scaled_partitions = [scaled_partition.reshape(partition.shape) for scaled_partition, partition in zip(scaled_partitions, partitions)]

    return scaled_partitions


#def mpi_global_min_max_scaling(partitions, feature_range):
#    """Scale partition features to the given range, based on the global min and max across all partitions and processes.
#    By default global here means communicator=MPI.COMM_WORLD.
#    In future will allow a user passed in communicator that may be different.
#
#    Parameters
#    ----------
#    partitions: list of :class:`numpy.ndarray`, required
#        See :mod:`isml.signature` for details.
#    feature_range: two-tuple, required
#        Minimum and maximum output values.
#
#    Returns
#    -------
#    partitions: list of :class:`numpy.ndarray`
#        See :mod:`isml.signature` for details.
#    """
#
#    communicator = MPI.COMM_WORLD 
#
#    # Compute local feature min and max values.
#    local_min = numpy.amin(data)
#    local_max = numpy.amax(data)
#
#    # Get global min and max values.
#    global_min = communicator.allreduce(local_min, MPI.MIN)
#    global_max = communicator.allreduce(local_max, MPI.MAX)
#
#    # Normalize local data using the global min and max.
#    scaling = sklearn.preprocessing.MinMaxScaler(feature_range)
#    scaling.fit([[global_min], [global_max]])
#    data = scaling.transform(data.reshape(-1, 1)).reshape(data.shape)
#
#    return data
#
