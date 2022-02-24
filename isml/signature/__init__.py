"""Functionality to compute per-partition signatures.

A signature is a small-ish vector that summarizes the state of a given
partition in a meaningful way.  By reducing a partition's state to a simpler
signature, we make it possible to perform inter-processor comparisons to
identify regions of spatial events.  Similarly, because signatures are small,
we can retain past timesteps' signatures on the local processor to identify
temporal events.

To compute a signature, we use a Python callable (a function or a class that
has a __call__() method) that takes a list of :math:`P` matrices, one per
partition, with shape :math:`\\langle shape \\rangle \\times F`, where
:math:`\\langle shape \\rangle` is the shape of the partition grid and
:math:`F` is the number of features at each grid point.

The callable must return a :math:`P \\times S` matrix containing the
:math:`S`-dimensional signature vector for each partition as output.

Note that implementations should not assume that :math:`\\langle shape
\\rangle` is the same for all partitions, since the local domain might not
permit uniform decompositions.
"""

import numpy
import sklearn.ensemble
import sklearn.neighbors
import sklearn.manifold
import scipy.linalg

import isml.feature
import isml.signature.postprocess as postprocess


def require_valid_partitions(partitions):
    """Sanity check partitions before conversion to signatures."""
    if not isinstance(partitions, list):
        raise ValueError("Partitions must be a list.")
    for partition in partitions:
        if not isinstance(partition, numpy.ndarray):
            raise ValueError("Partitions must be a list of numpy arrays.")
        if partition.ndim < 2:
            raise ValueError("Partitions must be a list of numpy arrays with ndim >= 2.")


def constant(partitions, value):
    """Signature implementation that returns a constant per-feature value, no matter the input.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)
    return numpy.array([numpy.repeat(value, partition.shape[-1]) for partition in partitions])


def mean(partitions):
    """Return a signature composed of per-feature mean values.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)
    return numpy.array([numpy.mean(isml.feature.matrix(partition), axis=0) for partition in partitions])


def percentile(partitions, *, q):
    """Return a signature based on per-feature percentile values.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        Sed :mod:`isml.signature` for details.
    percentiles: array-like, required
        The percentiles to extract for each feature.  For example, to
        obtain quartiles for each parameter, use [0, 25, 50, 75, 100]

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)

    signature = []
    for partition in partitions:
        percentiles = numpy.percentile(isml.feature.matrix(partition), q=q, axis=0)
        signature.append(numpy.concatenate(percentiles.T))
    return numpy.array(signature)


def fieda(partitions, *, random_state=None, n_estimators=500):
    """Return a per-processor-timestep signature using the FIEDA algorithm.

    Notes
    -----
    Assumes that the input partitions have already been normalized.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)

    signatures = []
    for partition in partitions:
        partition = isml.feature.matrix(partition)

        # Estimate the multi-dimensional log-pdf for this data.
        bandwidth = partition.shape[0] ** (-1 / (4 + partition.shape[1])) # Scott's rule
        kde = sklearn.neighbors.KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(partition)
        log_pdf = kde.score_samples(partition)

        # Train a random-forest regressor to estimate the log-pdf.
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(partition, log_pdf)

        signatures.append(model.feature_importances_)

    return numpy.array(signatures)


def fmm(partitions, *, moment):
    """Return a per-processor-timestep signature using the Feature-Moment-Metric (FMM) algorithm.

    The default metric will be based on the second moment i.e. covariance. A
    better metric for anomalies may be the fourth moment i.e. co-Kurtosis

    Notes
    -----
    Assumes that the input partitions have already been normalized.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    # TO DO: Currently only for 2nd moment. Flesh out later for moment=4
    assert(moment == 2)

    require_valid_partitions(partitions)

    signatures = []
    for partition in partitions:
        partition = isml.feature.matrix(partition)

        #Compute the SVD of the input partition. Cast the right SVD vectors and
        #singular values as desired principal vectors and prinicpal values.

        U, principal_values, principal_vecs = scipy.linalg.svd(partition, full_matrices=True)

        #Desired principal values are actually square of SVD singular values
        principal_values = numpy.square(principal_values)

        N = partition.shape[1]

        #Start the Feature moment metric computation

        #First compute sum of principal values, used to normalize
        sum_principal_vals = numpy.sum(principal_values)

        fmms = numpy.zeros(N)

        #The principal vectors should have been transposed after SVD
        #But I just take care of it in the way I index it
        #If there's a bug, this is the first place to look
        for i in range(N):
            for k in range(N):
                fmms[i] = fmms[i] + principal_values[k]*numpy.square(principal_vecs[k,i])


        fmms = numpy.divide(fmms, sum_principal_vals)

        del U
        signatures.append(fmms)

    return numpy.array(signatures)


def random_projection(partitions, *, n_dim, random_state):
    """Return a signature composed of the frequency with the highest
    power spectral density based on Welch's method.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.
    n_dim: number of dimensions to which the features in a partition should be reduced
    random_state: random state to use for creation of the projection matrix.
        See : `sklearn.random_projection` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    require_valid_partitions(partitions)

    from sklearn import random_projection
    transformer = random_projection.GaussianRandomProjection(n_components=n_dim,random_state=random_state)

    signatures = []
    for partition in partitions:
        partition = isml.feature.matrix(partition)

        new_signature=transformer.fit_transform(numpy.array([partition.flatten()])).squeeze()
        signatures.append(new_signature)

    signatures = numpy.array(signatures)

#     from sklearn.metrics import pairwise_distances
#     import numpy as np
#     distances=pairwise_distances(signatures)
#     distance_stats=np.array([[np.min(distances[i][np.nonzero(distances[i])]),np.mean(distances[i][np.nonzero(distances[i])]),np.max(distances[i])] for i in range(len(distances))])
#     sorted_min_indices=distance_stats.T[0].argsort()[::]
#     sorted_mean_indices=distance_stats.T[1].argsort()[::]
#     sorted_max_indices=distance_stats.T[2].argsort()[::]
#
#     print ("STATS")
#     print (distance_stats.T[0][sorted_min_indices][0],distance_stats.T[0][sorted_min_indices][255])
#     print (distance_stats.T[1][sorted_mean_indices][0],distance_stats.T[1][sorted_mean_indices][255])
#
#     print (distance_stats.T[2][sorted_max_indices][0],distance_stats.T[2][sorted_max_indices][255])

    return signatures


class DeepSig:
    """Loads a deep neural network, projecting the features through this to form a compressed representation.

    Parameters
    ----------
    decision_func: callable
        Another decision function, see :mod:`isml.decision` for details.
    timesteps: integer
        Minimum lifetime of `True` decisions.

    Any `True` decision returned by the wrapped function will persist for `timesteps` iterations.
    """
    def __init__(self, filepath, partition_dim_0, partition_dim_1, num_features, num_hidden_layers, encoding_dim):
        from keras.layers import Lambda, Input, Dense
        from keras.models import Model, load_model
        from keras.losses import mse, binary_crossentropy
        from keras import backend as K

        import numpy as np

        self.filepath = filepath
        self.partition_dim_0=partition_dim_0
        self.partition_dim_1=partition_dim_1
        self.num_features=num_features
        self.num_hidden_layers=num_hidden_layers
        self.encoding_dim = encoding_dim

        if self.filepath is None:
            self.encoder=self.build_encoder()
        else:
            self.encoder=self.load_encoder()

    def load_encoder(self):
        from keras.models import load_model
        return load_model(self.filepath)

    def build_encoder(self):
        from keras.layers import Lambda, Input, Dense
        from keras.models import Model
        features_shape=self.partition_dim_0*self.partition_dim_1*self.num_features
        input_shape = features_shape

        self.num_hidden_layers=2
        shape_diff=input_shape-self.encoding_dim
        shape_step=int(shape_diff/(self.num_hidden_layers+1))

        inputs = Input(shape=(features_shape,))
        x=inputs

        current_shape=input_shape

        for i in range(self.num_hidden_layers):
            current_shape=current_shape-shape_step
            x = Dense(current_shape,activation='elu')(x)
        outputs = Dense(self.encoding_dim,activation='linear')(x)

        encoder = Model(inputs=inputs, outputs=outputs)
        encoder.compile(optimizer='adadelta', loss="mse")

        return encoder

    def __call__(self, partitions):
        require_valid_partitions(partitions)

        signatures = []
        for partition in partitions:
            partition = isml.feature.matrix(partition)
            inputs=partition.flatten()
            signatures.append(self.encoder.predict(numpy.array([inputs])).squeeze())

        return numpy.array(signatures)


def svd(partitions):
    """Return a signature composed of singular values of each feature obtained through SVD.

    Parameters
    ----------
    partitions: list of :class:`numpy.ndarray`, required
        See :mod:`isml.signature` for details.

    Returns
    -------
    signatures: :class:`numpy.ndarray`
        See :mod:`isml.signature` for details.
    """
    return numpy.array([scipy.linalg.svd(isml.feature.matrix(partition), compute_uv=False, overwrite_a=True, check_finite=False) for partition in partitions])


