"""Functionality to compute novelty measures.

A measure is a per-partition scalar value that can be used to make a decision
about whether the partition should be considered "novel".

To compute measures, we use a Python callable (a function or a class that has a
__call__() method) that takes a list of :math:`T` signature matrics of shape
:math:`P \\times S` containing the :math:`S`-dimensional signatures for each of
:math:`P` partitions, across :math:`T` timesteps.  The last matrix in the list
contains the signatures for the current timestep.

Implementations can choose to compute spatial novelty measures by comparing
signatures from different partitions, or temporal measures by comparing
signatures from a single partition across all available times, or a hybrid
of the two.

Implementations must return a size-:math:`P` vector containing the measure value
for each partition.
"""

import numpy
import math

from sklearn.preprocessing import MinMaxScaler, Normalizer, normalize
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
from numpy.linalg import svd

#try:
#    from mpi4py import MPI
#except:
#    pass


def require_valid_signatures(signatures):
    """Raise an exception if the input is not a valid list of per-partition signatures.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Raises
    ------
    ValueError: if the list contents aren't valid signatures.
    """
    if not isinstance(signatures, list):
        raise ValueError("Signatures must be a list.")
    for timestep in signatures:
        if not isinstance(timestep, numpy.ndarray):
            raise ValueError("Signatures must be a list of numpy arrays.")
        if timestep.ndim != 2:
            raise ValueError("Signatures must be a list of 2D numpy arrays.")
        if timestep.shape != signatures[0].shape:
            raise ValueError("Signature shapes must be consistent across time.")


def constant(signatures, value):
    """Measure implementation that returns a constant value, no matter the input.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]
    return numpy.repeat(value, current_signatures.shape[0])


def random(signatures, low=0, high=1):
    """Measure implementation that returns random values, ignoring the input.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    low: number, optional
    high: number, optional

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]
    return numpy.random.uniform(low=low, high=high, size=current_signatures.shape[0])


def m1(signatures):
    """Return a measure of uniqueness among per-processor-timestep signatures using the m1 (spatial) metric.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    mean_signature = current_signatures.mean(axis=0)

    results = []
    for signature in current_signatures:
        diff = numpy.zeros(len(current_signatures))
        selection = numpy.nonzero(mean_signature)
        diff[selection] = numpy.square(signature[selection] - mean_signature[selection]) / numpy.square(mean_signature[selection])
        results.append(numpy.sqrt(numpy.mean(diff)))

    return numpy.array(results)


def dbscan(signatures, *, metric, eps, min_sample_ratio):
    """Return a measure of uniqueness among per-processor-timestep signatures using DBSCAN.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    from sklearn.cluster import DBSCAN
    min_samples=int(len(current_signatures) * min_sample_ratio)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clusters = model.fit_predict(current_signatures)
    measure = [1.0 if cluster < 0.0 else 0.0 for cluster in clusters]

    return numpy.array(measure)


def isolation_forest(signatures, *, max_samples):
    """Return a measure of uniqueness among per-processor-timestep signatures using an isolation forest.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(max_samples=max_samples)
    clf.fit(current_signatures)
    assignments=clf.predict(current_signatures)
    results=[1.0 if i<0.0 else 0.0 for i in assignments]

    return numpy.array(results)


def mean_square_distance(signatures, *, x=0):
    """Return a measure of anomalousness based on the square distance of each signature from the mean of all signatures.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    x: float, optional
        Value to add to the max mean during normalization to ensure there is no value of 1

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    mean_signatures = current_signatures.mean(axis=0)

    results = []
    for signature in current_signatures:
        results.append(numpy.linalg.norm(mean_signatures-signature))
    results = numpy.array(results)

    return numpy.interp(results, (results.min(), results.max()), (0,1))


def lof(signatures, *, neighbors):
    """Return a measure of anomalousness based on local density deviation using Local Outlier Factor estimation.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    neighbors: integer, required
        Number of neighbors to be considered when determining outliers

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]
    assert(isinstance(neighbors, int))

    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=neighbors)
    lof.fit_predict(current_signatures)
    results = numpy.array(lof.negative_outlier_factor_)

    feature_min = numpy.amin(numpy.row_stack(results), axis=0)
    feature_max = numpy.amax(numpy.row_stack(results), axis=0)

    # Normalize local data using the global min and max.
    scaling = MinMaxScaler((0,1))
    scaling.fit(numpy.row_stack((feature_min, feature_max)))
    results = scaling.transform(results.reshape(1, results.size))

    return 1-results.reshape(-1)


def lof_predict(signatures, *, neighbors):
    """Return a measure of anomalousness based on local density deviation using Local Outlier Factor estimation.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    neighbors: integer, required
        Number of neighbors to be considered when determining outliers

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]
    assert(isinstance(neighbors, int))

    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=neighbors)
    predictions = lof.fit_predict(current_signatures)
    results = numpy.array(lof.negative_outlier_factor_)

    #feature_min = numpy.amin(numpy.row_stack(results), axis=0)
    #feature_max = numpy.amax(numpy.row_stack(results), axis=0)

    ## Normalize local data using the global min and max.
    #scaling = MinMaxScaler((0,1))
    #scaling.fit(numpy.row_stack((feature_min, feature_max)))
    #results = scaling.transform(results.reshape(1, results.size))

    #results = results.reshape(-1)

    #Get max and min of scores only for those samples marked as outliers
    #i.e. those whose predictions are -1, the others have predictions = 1
    results = numpy.where(predictions == -1, -1.0*results, 1.0)
#HK
#HK    results = results * predictions
#HK
    feature_max = numpy.amax( results )
    feature_min = numpy.amin( results )

    results = (feature_max - results) / (feature_max - feature_min )


    return results


def signature_scaling(signatures):
    """Return a measure of anomalousness based on average of all values when scaled using the current signature

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    score_vector = []
    tmp_sig_avg = []
    current_signatures = numpy.abs(current_signatures)
    for signature in current_signatures:
        temp_sig_vector = numpy.true_divide(current_signatures, signature, numpy.zeros_like(current_signatures), where=signature!=0)
        tmp_sig_avg = numpy.mean(temp_sig_vector, axis=0)
        x = numpy.where(tmp_sig_avg>0)
        x = numpy.array(x)
        z = 0
        if x.size == 0:
            score_vector.append(z)
        else:
            score_vector.append(numpy.amin(tmp_sig_avg[x]))

    results = 1 - numpy.array(score_vector)

    return results


def hellinger(p, q):
    return numpy.sqrt( numpy.sum( (numpy.sqrt(p) - numpy.sqrt(q)) ** 2.0) ) / numpy.sqrt(2.0)


def m1_hellinger(signatures):
    """Return a measure of uniqueness among per-processor-timestep signatures using the Hellinger distance based m1 (spatial) metric.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)
    current_signatures = signatures[-1]

    mean_signature = current_signatures.mean(axis=0)

    results = []
    for signature in current_signatures:
        results.append(hellinger(signature, mean_signature))
    return numpy.array(results)


def maximum_change(signatures, *, window_size):
    """Return a measure of anomalousness based on per-feature
    maximum change in magnitude across any two consecutive
    timesteps.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """

    require_valid_signatures(signatures)

    spikes = []
    tmp_spikes = []
    b_spikes = []
    for temporal_block in signatures:
        for idx, feat in enumerate(temporal_block):
            feat = numpy.interp(feat, (feat.min(), feat.max()), (.38, .85))
            for idx2, f in enumerate(feat):
                if idx2 > 0:
                    tmp_spikes.append((f-feat[idx-1]))
                else:
                    tmp_spikes.append(0)
            b_spikes.append(tmp_spikes)
            tmp_spikes = []

        b_spikes = numpy.vstack(b_spikes)
        spikes.append(b_spikes)
        b_spikes = []

    results = numpy.amax(numpy.amax(numpy.array(spikes), axis = 2), axis=1)
    del window_size
    return results


def change_frequency(signatures, *, window_size, threshold):
    """Return a measure of anomalousness based on the number of changes
    (drastic drops or increases across any 2 timesteps) occurring
    across all features for each partition.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """

    require_valid_signatures(signatures)

    spikes = []
    tmp_spikes = []
    b_spikes = []
    for temporal_block in signatures:
        for idx, feat in enumerate(temporal_block):
            for idx2, f in enumerate(feat):
                if idx2 > 0:
                    tmp_spikes.append(f-feat[idx-1])
                else:
                    tmp_spikes.append(0)
            b_spikes.append(tmp_spikes)
            tmp_spikes = []

        b_spikes = numpy.vstack(b_spikes)
        spikes.append(b_spikes)

        spike_count = []
        for spike in spikes:
            s = numpy.ndarray.flatten(abs(spike))
            spike_count.append(len(numpy.where(s>threshold)[-1]))
        b_spikes = []
        results = numpy.array(spike_count)
    del window_size

    return results


def power_spectral_density(signatures, *, window_size, method, **kwargs):
    """Return a measure of anomalousness based on the maximum or average PSD across
    timesteps, based on Welch's method.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """

    require_valid_signatures(signatures)

    results = []

    for temporal_block in signatures:
        ps = []
        pf = []

        for feat in temporal_block:
            tmp_f, tmp_p = welch(x=feat, **kwargs)
            ps.append(numpy.interp(tmp_p, (tmp_p.min(), tmp_p.max()), (.54, .92)))
            pf.append(tmp_f)
            #if len(tmp_p[numpy.nonzero(tmp_p)])==0:
            #    ps.append(0)
            #    pf.append(tmp_f)
            #else:
            #    ps.append(1-(numpy.amin(tmp_p[numpy.nonzero(tmp_p)])/numpy.amax(tmp_p[numpy.nonzero(tmp_p)])))
            #    pf.append(tmp_f)
        ps = numpy.array(ps)
        pf = numpy.vstack(pf)
        if method == 'maximum':
            del tmp_f
            results.append(numpy.amax(ps, axis=1))
        elif method == 'average':
            del tmp_f
            results.append(ps.mean())
        elif method == 'frequency':
            results.append(numpy.mean(tmp_p[numpy.where(pf==pf.max())[-1]]))
        else:
            raise Exception("Invalid method specified.  Only 'maximum' and 'average' are supported")
    results = numpy.array(results).reshape(-1,)

    del window_size

    return numpy.interp(results, (results.min(), results.max()), (results.min(), 1))#1 - numpy.interp(results, (results.min(), results.max()), (0, 1))


def mse(signatures, *, window_size):
    """Return a measure of anomalousness based on mean squared
    error between the current and previous temporal block. The
    greater the error, the greater the change from one timestep
    to the next.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)

    results = []

    for block in signatures:
        block_a = block[:, -(window_size-1):]
        block_b = block[:, :window_size-1]
        results.append(mean_squared_error(block_a, block_b))
    return numpy.array(results)


def fuzzy(signatures, *, window_size):
    """Return a measure of anomalousness based on the application of
    a fuzzy logic, triangular membership function. Takes the maximum
    value of the fuzzy set to use as a measure.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """

    require_valid_signatures(signatures)

    results = []

    for block in signatures:
        a = block.min()
        b = block.mean()
        c = block.max()

        tmp = []

        for row in block:
            tmp.append(skfuzzy.trimf(row, (a,b,c)))

        results.append(numpy.array(tmp).max())
    del window_size

    return numpy.array(results)


def fuzzy2(signatures, *, window_size, no_mfx, mfx_params, mfx_type):
    """Return a measure of anomalousness based on the application of
    a fuzzy logic, triangular or trapezoidal membership function.
    ****WIP****

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    no_mfx: :class:`int`, required
        Number of membership functions you want to use
    mfx_params: list of :class:`numpy.ndarray`, required
        List of parameters for the membership function (a, b, d) for
        triangular and (a, b, c, d) for trapezoidal, where a and d
        represent the feet of the membership function, and b and c
        represent the peak (tri) or shoulders (trap) of the membership
        function
    mfx_type: :class: `str`, required
        String indicating the type of membership function(s) to use.
        Must be `tri` for triangular or `trap` for trapezoidal

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """

    require_valid_signatures(signatures)
    assert(isinstance(no_mfx, int))
    assert(isinstance(mfx_params, list))
    for params in mfx_params:
        if mfx_type == 'tri':
            assert(len(params)==3)
        elif mfx_type == 'trap':
            assert(len(params)==4)
        assert((params == numpy.sort(params)).all())
    assert(isinstance(mfx_type, str))
    assert(len(mfx_params == no_mfx))

    results = []

    for block in signatures:
        for feat in block:
            feat_fuzzy_sets = []
            for i in numpy.arange(no_mfx):
                tmp = []
                if mfx_type == 'tri':
                    tmp.append(skfuzzy.trimf(feat, mfx_params[i]))
                elif mfx_type == 'trap':
                    tmp.append(skfuzzy.trapmf(feat, mfx_params[i]))
                feat_fuzzy_sets.append(numpy.vstack[tmp])
        results.append(numpy.array(tmp).max())
    del window_size 
    return numpy.array(results)


class MovingAverage(object):
    """Reduce the signatures for the last N timesteps to an average signature.

    This adapter computes a moving average of the signatures for the past N
    timesteps.  The result is a per-partition "average" signature, suitable for
    use with measure functions that only consider a single timestep.


    Parameters
    ----------
    window: integer, required
        The number of past signatures to reduce.  Note that at the beginning of
        the simulation, shorter windows will be returned.
    measure: callable, required
        The measure function to convert the average signature into a measure.
    """
    def __init__(self, window, measure):
        self._window = window
        self._measure = measure

    def __call__(self, signatures):
        average = numpy.mean(signatures[-self._window : ], axis=0)
        return self._measure(signatures=[average])


class MovingWindow(object):
    """Apply specified measure to the signatures for the last N timesteps.

    This adapter applies a temporal measure on a per-partition basis to the
    signatures for the past N timesteps.  The result is a per-partition block
    of signatures, suitable for use with measure functions that consider a
    multiple timesteps.

    Parameters
    ----------
    window: integer, required
        The number of past signatures to reduce.  Note that at the beginning of
        the simulation, shorter windows will be returned.
    measure: callable, required
        The measure function to convert the average signature into a measure.
    """
    def __init__(self, window, measure):
        self._window = window
        self._measure = measure

    def __call__(self, signatures):
        if len(signatures) < self._window:
            results = numpy.ones((len(signatures[-1])))
        else:
            current_signatures = signatures[-self._window:]
            temporal_block = []
            for signature in current_signatures:
                tmp_sig = []
                for sig in signature:
                    tmp_sig.append(sig.reshape(-1, 1))
                temporal_block.append(numpy.array(tmp_sig))
            temporal_blocks = numpy.concatenate(temporal_block, axis=2)#.tolist()
            #sig = []
            #for block in temporal_blocks:
            #    sig.append(numpy.vstack(block))

            results = self._measure(signatures=temporal_blocks, window_size=self._window)
        return results


#def m1_distributed(signatures):
#    """Return a measure of uniqueness among per-processor-timestep signatures using the m1 (spatial) metric.
#
#    Parameters
#    ----------
#    signatures: list of :class:`numpy.ndarray`, required
#        See :mod:`isml.measure` for details.
#
#    Returns
#    -------
#    measures: :class:`numpy.ndarray`
#        See :mod:`isml.measure` for details.
#    """
#    require_valid_signatures(signatures)
#    current_signatures = signatures[-1]
#
#    communicator = MPI.COMM_WORLD  # @UndefinedVariable
#    nprocs = communicator.Get_size()
#    local_signatures = current_signatures.sum(axis=0)
#    sum_signatures = numpy.zeros(len(local_signatures))
#    MPI.COMM_WORLD.Allreduce(local_signatures, sum_signatures, op=MPI.SUM)  # @UndefinedVariable
#
#    mean_signature = sum_signatures/nprocs #current_signatures.mean(axis=0)
#
#    results = []
#    for signature in current_signatures:
#        diff = numpy.zeros(len(mean_signature)) #numpy.zeros(len(current_signatures))
#        selection = numpy.nonzero(mean_signature)
#        diff[selection] = numpy.square(signature[selection] - mean_signature[selection]) / numpy.square(mean_signature[selection])
#        results.append(numpy.sqrt(numpy.mean(diff)))
#
#    return numpy.array(results)


def svd(signatures, window_size, version):
    """Return a measure of anomalousness based on magnitude and angle
    of eigenvalues and singular vectors respectively.

    Parameters
    ----------
    signatures: list of :class:`numpy.ndarray`, required
        See :mod:`isml.measure` for details.
    window: integer, required
        The number of past signatures to reduce.  Note that at the beginning of
        the simulation, shorter windows will be returned.
    version: :class: `str` required,
        Specifies what feature of SVD is desired to define the measures,
        options are `angles` for change in angle of largest singular vectors
        (first and second), `ratio` for ratio of largest to smallest non-zero
        singular value magnitude, and `both` for both of the angles and ratio

    Returns
    -------
    measures: :class:`numpy.ndarray`
        See :mod:`isml.measure` for details.
    """
    require_valid_signatures(signatures)

    del window_size
    possible_versions = ['angles', 'ratio', 'both']
    assert(isinstance(version, str))
    assert(version in possible_versions)
    results = []
    f_angles = []
    s_angles = []
    for temporal_block in signatures:
        temporal_block = temporal_block.T
        f_angle = []
        s_angle = []
        fv = [] #list first vectors for each temporal block
        sv = [] #list of second vectors for each temporal block
        rv = [] #list of ratios for each temporal block
        #for signature in temporal_block:
        if version == 'angles':
            for signature in temporal_block:
                signature = signature.reshape(2,-1)
                first_vector = signature[0]
                second_vector = signature[1]
                fv.append(first_vector)
                sv.append(second_vector)
            fv = numpy.vstack(fv)
            sv = numpy.vstack(sv)

            fv0 = fv[0,:]
            fv1 = fv[1,:]

            f_angle = numpy.arccos((numpy.dot(fv0,fv1))/(numpy.dot(numpy.linalg.norm(fv0),numpy.linalg.norm(fv1))))
            idf = numpy.eye(len(fv0))

            sv0 = sv[0,:]
            sv1 = sv[1,:]

            s_angle = numpy.arccos((numpy.dot(sv0,sv1))/(numpy.dot(numpy.linalg.norm(sv0),numpy.linalg.norm(sv1))))
            ids = numpy.eye(len(sv0))

            for i, vec in enumerate(fv):
                idf[:, i] = vec

            for i, vec in enumerate(sv):
                ids[:, i] = vec

            fs = numpy.sign(numpy.linalg.det(idf)) # Get direction of the first angle change
            ss = numpy.sign(numpy.linalg.det(ids)) # Get direction of the second angle change

            fs_angle = fs*f_angle
            ss_angle = ss*s_angle

            ang_diff = numpy.abs(fs_angle-ss_angle)

            if fs < 0 and ss < 0 or ((numpy.rad2deg(ang_diff) > 0 and numpy.rad2deg(ang_diff)< 210)):
                results.append(.5)
            else:
                results.append(0)

        elif version == 'ratio':
            for signature in temporal_block:
                singular_values = signature
                if len(numpy.nonzero(singular_values)[0])>1:
                    ratio = ((numpy.amax(singular_values[numpy.nonzero(singular_values)])/numpy.amin(singular_values[numpy.nonzero(singular_values)])))
                    oom = int(math.log10(ratio))
                    rv.append(oom)
                else:
                    rv.append(numpy.array(0))
            results.append(numpy.mean(rv))
        elif version == 'both':
            for signature in temporal_block:
                signature = signature.reshape(3,-1)
                first_vector = signature[0]
                second_vector = signature[1]
                singular_values = signature[2]
                if len(numpy.nonzero(singular_values)[0])>1:
                    ratio = ((numpy.amax(singular_values[numpy.nonzero(singular_values)])/numpy.amin(singular_values[numpy.nonzero(singular_values)])))
                    oom = int(math.log10(ratio))
                    rv.append(oom)
                else:
                    rv.append(numpy.array(0))
                fv.append(first_vector)
                sv.append(second_vector)
                #tmp_r = numpy.mean(rv)
            fv = numpy.vstack(fv)
            sv = numpy.vstack(sv)

            fv0 = fv[0,:]
            fv1 = fv[1,:]

            f_angle = numpy.arccos((numpy.dot(fv0,fv1))/(numpy.dot(numpy.linalg.norm(fv0),numpy.linalg.norm(fv1))))
            idf = numpy.eye(len(fv0))

            sv0 = sv[0,:]
            sv1 = sv[1,:]

            s_angle = numpy.arccos((numpy.dot(sv0,sv1))/(numpy.dot(numpy.linalg.norm(sv0),numpy.linalg.norm(sv1))))
            ids = numpy.eye(len(sv0))

            for i, vec in enumerate(fv):
                idf[:, i] = vec

            for i, vec in enumerate(sv):
                ids[:, i] = vec

            fs = numpy.sign(numpy.linalg.det(idf)) # Get direction of the first angle change
            ss = numpy.sign(numpy.linalg.det(ids)) # Get direction of the second angle change

            fs_angle = fs*f_angle
            ss_angle = ss*s_angle

            ang_diff = numpy.abs(fs_angle-ss_angle)
            if numpy.mean(rv) < 2:
                tmp_r = 0.5
            else:
                tmp_r = 0

            if fs < 0 and ss < 0 or ((numpy.rad2deg(ang_diff) > 0 and numpy.rad2deg(ang_diff)< 210)):
                tmp_a = 0.5
            else:
                tmp_a = 0

            results.append(tmp_r + tmp_a)
#             #rv = numpy.array(rv)
#
#             fv0 = fv[0,:]
#             fv1 = fv[1,:]
#             f_angle = numpy.arccos((numpy.dot(fv0,fv1))/(numpy.dot(numpy.linalg.norm(fv0),numpy.linalg.norm(fv1))))
#
#             sv0 = sv[0,:]
#             sv1 = sv[1,:]
#             s_angle = numpy.arccos((numpy.dot(sv0,sv1))/(numpy.dot(numpy.linalg.norm(sv0),numpy.linalg.norm(sv1))))
#
#             ang_diff = numpy.amax(numpy.abs(f_angle), numpy.abs(s_angle))
#             r = min(numpy.abs((((.5*numpy.pi)-ang_diff),((1.5*numpy.pi)-ang_diff))))
#
#             if r < .2:
#                 tmpv = .5
#             else:
#                 tmpv = 0
#
#             if len(numpy.nonzero(singular_values)[0])>1:
#                 ratio = ((numpy.amax(singular_values[numpy.nonzero(singular_values)])/numpy.amin(singular_values[numpy.nonzero(singular_values)])))
#                 oom = int(math.log10(ratio))
#                 rv.append(oom)
#                 if oom > 4:
#                     tmpr = .5
#                 else:
#                     tmpr = 0
#             else:
#                 rv.append(numpy.array(0))
#                 tmpr = 0
#             results.append(tmpv+tmpr)
    if version == 'angles':
        results = numpy.array(results)
    elif version == 'ratio':
        results = numpy.array(results)
    elif version == 'both':
        results = numpy.array(results)

    return results


#def mean_square_distance_distributed(signatures, *, x=0):
#    """Return a measure of anomalousness based on the square distance of each signature from the mean of all signatures.
#    The mean of signatures is a global mean, computed across all MPI partitions.
#
#    Parameters
#    ----------
#    signatures: list of :class:`numpy.ndarray`, required
#        See :mod:`isml.measure` for details.
#    x: float, optional
#        Value to add to the max mean during normalization to ensure there is no value of 1
#
#    Returns
#    -------
#    measures: :class:`numpy.ndarray`
#        See :mod:`isml.measure` for details.
#    """
#    require_valid_signatures(signatures)
#    current_signatures = signatures[-1]
#
#    communicator = MPI.COMM_WORLD  # @UndefinedVariable
#    nprocs = communicator.Get_size()
#    local_signatures = current_signatures.sum(axis=0)
#    sum_signatures = numpy.zeros(len(local_signatures))
#    MPI.COMM_WORLD.Allreduce(local_signatures, sum_signatures, op=MPI.SUM)  # @UndefinedVariable
#
#    mean_signatures = sum_signatures/nprocs #current_signatures.mean(axis=0)
#
#    results = []
#    for signature in current_signatures:
#        results.append(numpy.linalg.norm(mean_signatures-signature))
#    results = numpy.array(results)
#
#    if(isinstance(x, float)):
#        results = numpy.true_divide(results, (numpy.amax(results, axis=0) + x)) 
#    else:
#        results = numpy.true_divide(results, numpy.amax(results, axis=0))
#
#    return results
#
