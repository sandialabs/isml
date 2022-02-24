"""Functionality to perform postprocessing on signatures.

This module provides functionality to postprocess per-partition signatures
before they are used to compute novelty measures.  These callables all take
signatures as inputs (described in :mod:`isml.signature`) and produce modified
signatures as outputs.

Note that these callables receive and produce *local* signatures (local to the
current process), but the implementations are free to perform communication
(for example: using MPI) as needed.
"""

import numpy
import sklearn.manifold
import sklearn.random_projection


def require_valid_signatures(signatures):
    """Raise an exception if the input is not a valid list of per-partition signatures.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Raises
    ------
    ValueError: if the list contents aren't valid signatures.
    """
    if not isinstance(signatures, numpy.ndarray):
        raise ValueError("Signatures must be a numpy array.")
    if signatures.ndim != 2:
        raise ValueError("Signatures must be a 2D numpy array.")


def identity(signatures):
    """Do-nothing example of a signature postprocessor.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    return signatures


def tsne(signatures, *args, **kwargs):
    """Reduce the dimensionality of signatures using T-SNE.

    Note
    ----
    Accepts any of the parameters supported by :class:`sklearn.manifold.TSNE`.
    By default, signatures will be reduced to two dimensions.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    return sklearn.manifold.TSNE(*args, **kwargs).fit_transform(signatures)


def random_projection(signatures, *args, **kwargs):
    """Reduce the dimensionality of signatures using random projection.

    Note
    ----
    Accepts any of the parameters supported by :class:`sklearn.random_projection`.
    By default, signatures will be reduced to two dimensions.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    return sklearn.random_projection.GaussianRandomProjection(*args, **kwargs).fit_transform(signatures)


def mds(signatures, *args, **kwargs):
    """Reduce the dimensionality of signatures using MDS.

    Note
    ----
    Accepts any of the parameters supported by :class:`sklearn.manifold.MDS`.
    By default, signatures will be reduced to two dimensions.

    The current implementation discards signatures from prior timesteps,
    making it unsuitable for use with temporal measures / analysis.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    return sklearn.manifold.MDS(*args, **kwargs).fit_transform(signatures)


def isomap(signatures, *args, **kwargs):
    """Reduce the dimensionality of signatures using Isomap.

    Note
    ----
    Accepts any of the parameters supported by :class:`sklearn.manifold.Isomap`.
    By default, signatures will be reduced to two dimensions.

    The current implementation discards signatures from prior timesteps,
    making it unsuitable for use with temporal measures / analysis.

    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        see :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    return sklearn.manifold.Isomap(*args, **kwargs).fit_transform(signatures)


def pruning_2d(signatures, *, dimensions, point_density):
    """Partitions the 2D space into grid and calculates the density of each grid location using a KDTree.

    The points in regions with the highest density are then "removed" (replaced with zeros)


    Parameters
    ----------
    signatures: :class:`numpy.ndarray`, required
        see :mod:`isml.signature` for details.
    dimensions: :class:'numpy.ndarray', required
        :math: '1\\times 2' vector containing the number of rows and columns desired in each of the partitions

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_signatures(signatures)
    assert(isinstance(dimensions, tuple))
    dimensions = numpy.array(dimensions)
    assert(dimensions.shape == (2,))

    signatures = mds(signatures, n_components=2, n_init=1, max_iter=100)

    xd = dimensions[0]-2
    yd = dimensions[1]-2
    from sklearn.neighbors import KDTree
    xmin = numpy.amin(signatures[:,0])
    xmax = numpy.amax(signatures[:,0])
    ymin = numpy.amin(signatures[:,1])
    ymax = numpy.amax(signatures[:,1])
    rangex = numpy.ptp(signatures[:,0])
    rangey = numpy.ptp(signatures[:,1])

    gridx = numpy.append(numpy.append(xmin-(rangex/xd), numpy.arange(xmin, xmax, rangex/xd)), xmax+(rangex/xd))
    gridy = numpy.append(numpy.append(ymin-(rangex/yd), numpy.arange(ymin, ymax, rangey/yd)), ymax+(rangey/yd))

    xx, yy = numpy.meshgrid(gridx, gridy, sparse=True)
    point = numpy.empty([1,2], dtype=int)
    for xn in xx.T:
        for yn in yy:
            point = numpy.append(point, numpy.reshape(numpy.array([xn, yn]),(1,2)), axis = 0)

    points = numpy.delete(point, 0,0)

    tree = KDTree(points, leaf_size=2)

    dist, reg = tree.query(signatures, k=1)

    hist, edges = numpy.histogram(reg, (dimensions[0]*dimensions[1]), range=(0,(dimensions[0]*dimensions[1])))

    dense_points = numpy.array(numpy.where(hist>point_density))
    idx = numpy.isin(reg, dense_points)
    dense_space_points, ig = numpy.where(idx == True)
    zsigs = signatures[dense_space_points,:]

    del dist, edges, ig

    for sig in zsigs:
        numpy.place(signatures, signatures==sig, [0, 0])

    return signatures

