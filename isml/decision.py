"""Functionality to decide when a partition should be considered "novel".

A decision is a per-partition boolean value that is `True` when the partition
should be considered "novel".

To compute decisions, we use a Python callable (a function or a class that has
a __call__() method) that takes a list of :math:`T` vectors of size :math:`P`,
containing the novelty measures for each of :math:`P` partitions across
:math:`T` timesteps.  The last vector in the list contains the measures for the
current timestep.

Implementations must return a size :math:`P` vector containing the novelty
decision for each partition.  Typically, they would use the last measure vector
in the given list to evaluate the current timestep, but they may choose to base
their decision on measures from previous timesteps if desired.
"""

import numpy


def require_valid_measures(measures):
    """Raise an exception if the input is not a valid list of per-pertition measures.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.

    Raises
    ------
    ValueError: if the list contents aren't valid measures.
    """
    if not isinstance(measures, list):
        raise ValueError("Measures must be a list.")
    for timestep in measures:
        if not isinstance(timestep, numpy.ndarray):
            raise ValueError("Measures must be a list of numpy arrays.")
        if timestep.ndim != 1:
            raise ValueError("Measures must be a list of 1D numpy arrays.")
        if timestep.shape != measures[0].shape:
            raise ValueError("Measure shapes must be consistent across time.")


def checkpoint(measures, *, timestep, n, offset):
    """Flag every partition as interesting if the current timestep mod `n` equals `offset`.

    This is useful for reproducing the behavior of simulation codes that simply
    checkpoint every nth timestep.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    timestep: integer, required
        Current timestep index.
    n: integer, required
    offset: integer, required

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    require_valid_measures(measures)
    current_measures = measures[-1]

    if timestep % n == offset:
        decisions = numpy.ones_like(current_measures, dtype="bool")
    else:
        decisions = numpy.zeros_like(current_measures, dtype="bool")
    return decisions


def greater_than(measures, *, threshold):
    """Flag partitions as interesting if their measure is above an absolute threshold.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    threshold: number, required
        Threshold value above which the measure is considered "interesting".

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    require_valid_measures(measures)
    current_measures = measures[-1]

    return current_measures > threshold


def less_than(measures, *, threshold):
    """Flag partitions as interesting if their measure is below an absolute threshold.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    threshold: number, required
        Threshold value below which the measure is considered "interesting".

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    require_valid_measures(measures)
    current_measures = measures[-1]

    return current_measures < threshold


def percentile(measures, *, percentile):
    """Flag partitions as interesting if their measure exceeds a percentile threshold.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    percentile: number in the range :math:`[0, 1)`, required
        Threshold percentile above which the measure is considered "interesting".

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    require_valid_measures(measures)
    current_measures = measures[-1]

    threshold = numpy.percentile(current_measures, percentile * 100.0)
    return current_measures > threshold


class MemoryDecision:
    """Wraps another decision function, forcing a minimum lifetime (in timesteps) for `True` decisions.

    Parameters
    ----------
    decision_func: callable, required
        Another decision function, see :mod:`isml.decision` for details.
    timesteps: integer, required
        Minimum lifetime of `True` decisions.

    Any `True` decision returned by the wrapped function will persist for `timesteps` iterations.
    """
    def __init__(self, decision_func, *, timesteps):
        self.decision_func = decision_func
        self.timesteps = timesteps
        self.decision_array = []

    def __call__(self, measures):
        decision = self.decision_func(measures)
        self.decision_array.append(decision)

        final_decision = numpy.array(self.decision_array)[-self.timesteps:].any(axis=0)

        return final_decision


def outside(measures, *, lower_threshold, upper_threshold, mode):
    """Flag partitions as interesting if their measure falls outside a range of values.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    lower_threshold: number, required
        Threshold value below which a metric is considered "interesting".
    upper_threshold: number, required
        Threshold value above which a metric is considered "interesting".
    mode: "or" or "and", required
        Controls whether the measure is considered interesting when outside one or both thresholds.

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    require_valid_measures(measures)
    current_measures = measures[-1]

    if mode == 'or':
        return (current_measures > upper_threshold) | (current_measures < lower_threshold)
    elif mode == 'and':
        return (current_measures > upper_threshold) & (current_measures < lower_threshold)
    else:
        raise ValueError(f"Unexpected mode: {mode}")


def percent_change(measures, *, threshold):
    """Flag a partition as interesting if its measures changes by more than the given percentage.

    Parameters
    ----------
    measures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.decision` for details.
    threshold: number, required
        Threshold value above which a metric is considered "interesting".

    Returns
    -------
    decisions: :class:`numpy.ndarray`
        See :mod:`isml.decision` for details.
    """
    if len(measures) < 2:
        return(measures[-1])

    current_measures = measures[-1].reshape(-1,1)
    previous_measures = measures[-2].reshape(-1,1)
    cp_measures = numpy.hstack((previous_measures, current_measures))
    dec = []
    for measure in cp_measures:
        a = measure[1]
        b = measure[0]

        pct = ((a-b)/b)
        dec.append(pct)
    return numpy.array(dec) > threshold
