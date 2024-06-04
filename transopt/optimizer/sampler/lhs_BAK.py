import numpy as np
import scipy.stats.qmc as qmc
from scipy import spatial
from scipy import stats
from scipy import linalg
from numpy import ma

__all__ = ["lhs"]


def lhs(d, samples=None, criterion=None, iterations=5, correlation_matrix=None):
    """
    Generate a latin-hypercube design
    Parameters
    ----------
    d : int
        The number of factors to generate samples for
    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: d)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    correlation_matrix : ndarray
         Enforce correlation between factors (only used in lhsmu)
    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    """
    H = None
    if samples is None:
        samples = d

    if criterion is None:
        return _lhsclassic(d, samples)

    criterion = criterion.lower()
    if not criterion in ("center", "c", "maximin", "m", "centermaximin", "cm", "correlation", "corr", "lhsmu"):
        raise ValueError('Invalid value for "criterion": {}'.format(criterion))

    if criterion in ("center", "c"):
        H = _lhscentered(d, samples)
    elif criterion in ("maximin", "m"):
        H = _lhsmaximin(d, samples, iterations, "maximin")
    elif criterion in ("centermaximin", "cm"):
        H = _lhsmaximin(d, samples, iterations, "centermaximin")
    elif criterion in ("correlation", "corr"):
        H = _lhscorrelate(d, samples, iterations)
    elif criterion in ("lhsmu"):
        # as specified by the paper. M is set to 5
        H = _lhsmu(d, samples, correlation_matrix, M=5)

    return H


def _lhsclassic(d, samples):
    sampler = qmc.LatinHypercube(d=d)
    return sampler.random(n=samples)



def _lhscentered(d, samples):
    sampler = qmc.LatinHypercube(d=d)
    H = sampler.random(n=samples)
    H = (np.floor(H * samples) + 0.5) / samples
    return H


def _lhsmaximin(d, samples, iterations, lhstype):
    maxdist = 0
    best_sample = None

    for i in range(iterations):
        sampler = qmc.LatinHypercube(d=d)
        Hcandidate = sampler.random(n=samples)

        if lhstype != "maximin":
            Hcandidate = (np.floor(Hcandidate * samples) + 0.5) / samples

        d = spatial.distance.pdist(Hcandidate, "euclidean")
        min_d = np.min(d)
        if maxdist < min_d:
            maxdist = min_d
            best_sample = Hcandidate.copy()

    return best_sample if best_sample is not None else np.zeros((samples, d))


def _lhscorrelate(d, samples, iterations):
    mincorr = np.inf
    best_sample = None

    for i in range(iterations):
        sampler = qmc.LatinHypercube(d=d)
        Hcandidate = sampler.random(n=samples)

        R = np.corrcoef(Hcandidate.T)
        max_corr = np.max(np.abs(R - np.eye(d)))

        if max_corr < mincorr:
            mincorr = max_corr
            best_sample = Hcandidate.copy()

    return best_sample


def _lhsmu(d, samples=None, corr=None, M=5):
    if samples is None:
        samples = d

    I = M * samples

    rdpoints = np.random.uniform(size=(I, d))

    dist = spatial.distance.cdist(rdpoints, rdpoints, metric="euclidean")
    D_ij = ma.masked_array(dist, mask=np.identity(I))

    index_rm = np.zeros(I - samples, dtype=int)
    i = 0
    while i < I - samples:
        order = ma.sort(D_ij, axis=1)
        avg_dist = ma.mean(order[:, 0:2], axis=1)
        min_l = ma.argmin(avg_dist)

        D_ij[min_l, :] = ma.masked
        D_ij[:, min_l] = ma.masked

        index_rm[i] = min_l
        i += 1

    rdpoints = np.delete(rdpoints, index_rm, axis=0)

    if corr is not None:
        # check if covariance matrix is valid
        assert type(corr) == np.ndarray
        assert corr.ndim == 2
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == d

        norm_u = stats.norm().ppf(rdpoints)
        L = linalg.cholesky(corr, lower=True)

        norm_u = np.matmul(norm_u, L)

        H = stats.norm().cdf(norm_u)
    else:
        H = np.zeros_like(rdpoints, dtype=float)
        rank = np.argsort(rdpoints, axis=0)

        for l in range(samples):
            low = float(l) / samples
            high = float(l + 1) / samples

            l_pos = rank == l
            H[l_pos] = np.random.uniform(low, high, size=d)
    return H


if __name__ == "__main__":
    """
    Example
    -------
    A 3-factor design (defaults to 3 samples)::
        >>> lhs(3, random_state=42)
        array([[ 0.12484671,  0.95539205,  0.24399798],
               [ 0.53288616,  0.38533955,  0.86703834],
               [ 0.68602787,  0.31690477,  0.38533151]])
    A 4-factor design with 6 samples::
        >>> lhs(4, samples=6)
        array([[ 0.06242335,  0.19266575,  0.88202411,  0.89439364],
               [ 0.19266977,  0.53538985,  0.53030416,  0.49498498],
               [ 0.71737371,  0.75412607,  0.17634727,  0.71520486],
               [ 0.63874044,  0.85658231,  0.33676408,  0.31102936],
               [ 0.43351917,  0.45134543,  0.12199899,  0.53056742],
               [ 0.93530882,  0.15845238,  0.7386575 ,  0.09977641]])
    A 2-factor design with 5 centered samples::
        >>> lhs(2, samples=5, criterion='center', random_state=42)
        array([[ 0.1,  0.9],
               [ 0.5,  0.5],
               [ 0.7,  0.1],
               [ 0.3,  0.7],
               [ 0.9,  0.3]])
    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::
        >>> lhs(3, samples=4, criterion='maximin')
        array([[ 0.69754389,  0.2997106 ,  0.96250964],
               [ 0.10585037,  0.09872038,  0.73157522],
               [ 0.25351996,  0.65148999,  0.07337204],
               [ 0.91276926,  0.97873992,  0.42783549]])
    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::
        >>> lhs(4, samples=5, criterion='correlation', iterations=10)
        array([[ 0.72088348,  0.05121366,  0.97609357,  0.92487081],
               [ 0.49507404,  0.51265511,  0.00808672,  0.37915272],
               [ 0.22217816,  0.2878673 ,  0.24034384,  0.42786629],
               [ 0.91977309,  0.93895699,  0.64061224,  0.14213258],
               [ 0.04719698,  0.70796822,  0.53910322,  0.78857071]])
    """

    h1 = lhs(4, samples=5)
    print(h1)

    sampler = qmc.LatinHypercube(d=4)
    h2 = sampler.random(n=5)
    print(h2)
    
    d = 3
    samples = 10
    corr = np.array([[1.0, 0.5, 0.2],
                    [0.5, 1.0, 0.3],
                    [0.2, 0.3, 1.0]])
    sampled_data = _lhsmu(d, samples, corr, M=5)
    print("Generated samples with specified correlation:")
    print(sampled_data)