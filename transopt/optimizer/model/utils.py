import itertools
from typing import List, Union, Tuple

import numpy as np
import scipy
from GPy.kern import Fixed, BasisFuncKernel



def is_pd(a: np.ndarray) -> bool:
    """Check whether matrix `a` is positive definite via Cholesky decomposition.

    Args:
        a: Input matrix.

    Returns:
        `True` if input matrix is positive-definite, `False` otherwise.
    """

    try:
        _ = np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_pd(a: np.ndarray) -> np.ndarray:
    """Calculate the nearest positive-definite matrix to a given symmetric matrix `a`.

    Nearest is defined by the Frobenius norm.

    Args:
        a: Symmetric matrix. `shape = (n, n)`

    Returns:
        The nearest positive-definite matrix to the input symmetric matrix `a`.
        `shape = (n, n)`
    """
    # compute eigendecomposition of symmetric matrix a
    w, v = np.linalg.eigh(a)

    # account for floating-point accuracy
    spacing = np.spacing(np.linalg.norm(a))

    # clip the eigenvalues at zero
    wp = np.clip(w, spacing, None)

    return np.dot(v, np.dot(np.diag(wp), np.transpose(v)))


def compute_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Calculate the Cholesky decomposition of a matrix.

    If the matrix is singular, a small constant is added to the diagonal of the matrix.
    This Method is therefore useful for the calculation of GP posteriors.

    Args:
        matrix: The input matrix. `shape = (n_points, n_points)`

    Returns:
        The Cholesky decomposition stored in the lower triangle.
            `shape = (n_points, n_points)`
    """
    assert len(matrix.shape) <= 2, (
        "The matrix has more than two input dimensions. Cholesky decomposition"
        "impossible."
    )
    assert (
        matrix.shape[0] == matrix.shape[1]
    ), "The matrix is not square. Cholesky decomposition impossible."

    _matrix = np.copy(matrix)  # to avoid modifying the input
    for k in itertools.count(start=1):
        try:
            chol = scipy.linalg.cholesky(_matrix, lower=True)
        except scipy.linalg.LinAlgError:
            # Increase eigenvalues of matrix
            np.fill_diagonal(_matrix, _matrix.diagonal() + 10 ** k * 1e-8)
        else:
            return chol


class FixedKernel(Fixed):
    """Fixed covariance kernel. Serializable version of the Fixed Kernel from `GPy`.

    Serialization is required to initialize a `gpy_adapter` `Model` using this kernel.
    """

    def __init__(
        self,
        input_dim: int,
        covariance_matrix: np.ndarray,
        active_dims: List[int] = None,
        name="PosteriorCov",
    ):
        """Initialize the kernel.

        Args:
            input_dim: Input dimension of the training data.
            covariance_matrix: The fixed covariance matrix.
            active_dims: Active dimensions.
            name: Name of the kernel.
        """
        super(FixedKernel, self).__init__(
            input_dim=input_dim,
            variance=1.0,
            covariance_matrix=covariance_matrix,
            active_dims=active_dims,
            name=name,
        )
        self.variance.fix()

    def to_dict(self) -> dict:
        """Save the kernel as a dictionary."""
        input_dict = super(Fixed, self)._save_to_input_dict()
        input_dict["covariance_matrix"] = self.fixed_K
        input_dict["class"] = "GPy.kern.Fixed"
        input_dict.pop("useGPU")

        return input_dict


def compute_alpha(model: "GP", x) -> np.ndarray:
    r"""Calculate the $\alpha(x)$ Woodbury vector used for computing the boosted
    covariance.

    $$
        \alpha(x) = k(x, X)\left(k(\X, \X) +\sigma^2\mathbb 1\right)^{-1}$,
    $$

    where $k$ is the kernel of `model`, $X$ is the training data of `model`, and
    $\sigma$ is the standard deviation of the observational noise.

    Args:
        model: The Gaussian-process model.
        x: The input data. `shape = (n_points, n_features)`

    Returns:
        The $\alpha$ vector. `shape = (n_points, n_training_points)`
    """
    L = model._gpy_model.posterior.woodbury_chol
    X = model.X
    k = model.compute_kernel(X, x)
    return scipy.linalg.solve_triangular(
        L.T, scipy.linalg.solve_triangular(L, k, lower=True)
    )


class CrossTaskKernel(BasisFuncKernel):
    """A kernel that is one iff the X-task corresponds to one of the `task_indices`."""

    def __init__(
        self,
        task_indices: Union[Tuple[int, int], int, np.ndarray],
        index_dim: int,
        variance=1.0,
        name="task_domain",
    ):
        super().__init__(
            input_dim=1,
            variance=variance,
            active_dims=(index_dim,),
            ARD=False,
            name=name,
        )
        self.task_indices = np.atleast_2d(np.asarray(task_indices, dtype=int))
        assert self.task_indices.size >= 1, "Need at least one task."

    def _phi(self, X: np.ndarray) -> np.ndarray:
        # atol maps our floats to tasks
        is_domain_task = np.isclose(X, self.task_indices, atol=0.5, rtol=0)
        return is_domain_task.any(axis=-1, keepdims=True)
