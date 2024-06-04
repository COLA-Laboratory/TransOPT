# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import numpy as np
from typing import Dict, Hashable, Union, Sequence, Tuple, List

from GPy.kern import RBF
from GPy.kern import Kern, RBF
from transopt.optimizer.model.gp import GP
from transopt.optimizer.model.model_base import Model
from transopt.agent.registry import model_registry

@model_registry.register("MHGP")
class MHGP(Model):
    """Stack of Gaussian processes.

    Transfer Learning model based on [Golovin et al: Google Vizier: A Service for
    Black-Box Optimization](https://dl.acm.org/doi/abs/10.1145/3097983.3098043).
    Given a list of source data sets, the
    transfer to the target data set is done by training a separate GP for each data set
    whose prior mean function is the posterior mean function of the previous GP in the
    stack.
    """

    def __init__(self,         
        kernel: Kern = None,
        noise_variance: float = 1.0,
        normalize: bool = True,
        **options: dict):
        """Initialize the Method.

        Args:
            n_features: Number of input parameters of the data.
            within_model_normalize: Normalize each GP internally. Helpful for
                numerical stability.
        """
        super().__init__()

        self._normalize = normalize
        self._kernel = kernel
        self._noise_variance = noise_variance
        self.n_samples = 0

        self.source_gps = []

        # GP on difference between target data and last source data set
        self.target_gp = None

    def _compute_residuals(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Determine the difference between given y-values and the sum of predicted
        values from the models in 'source_gps'.

        Args:
            data: Observation (input and target) data.
                Input data: ndarray, `shape = (n_points, n_features)`
                Target data: ndarray, `shape = (n_points, 1)`

        Returns:
            Difference between observed values and sum of predicted values
            from `source_gps`. `shape = (n_points, 1)`
        """
        if self.n_features != X.shape[1]:
            raise ValueError("Number of features in model and input data mismatch.")

        if not self.source_gps:
            return Y

        predicted_y = self.predict_posterior_mean(
            X, idx=len(self.source_gps) - 1
        )

        residuals = Y - predicted_y

        return residuals

    def _update_meta_data(self, *gps: GP):
        """Cache the meta data after meta training."""
        for gp in gps:
            self.source_gps.append(gp)

    def _meta_fit_single_gp(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        optimize: bool,
    ) -> GP:
        """Train a new source GP on `data`.

        Args:
            data: The source dataset.
            optimize: Switch to run hyperparameter optimization.

        Returns:
            The newly trained GP.
        """
        self.n_features = X.shape[1]
        
        residuals = self._compute_residuals(X, Y)
        
        kernel = RBF(self.n_features, ARD=True)
        new_gp = GP(
            kernel, noise_variance=self._noise_variance
        )
        new_gp.fit(
            X = X,
            Y = residuals,
            optimize = optimize,
        )
        return new_gp

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        optimize: Union[bool, Sequence[bool]] = True,
    ):
        """Train the source GPs on the given source data.

        Args:
            source_datasets: Dictionary containing the source datasets. The stack of GPs
                are trained on the residuals between two consecutive data sets in this
                list.
            optimize: Switch to run hyperparameter optimization.
        """
        assert isinstance(optimize, bool) or isinstance(optimize, list)
        if isinstance(optimize, list):
            assert len(source_X) == len(optimize)
        optimize_flag = copy.copy(optimize)

        if isinstance(optimize_flag, bool):
            optimize_flag = [optimize_flag] * len(source_X)

        for i in range(len(source_X)):
            new_gp = self._meta_fit_single_gp(
                source_X[i],
                source_Y[i],
                optimize=optimize_flag[i],
            )
            self._update_meta_data(new_gp)



    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimize: bool = False,
    ):
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )

        self._X = copy.deepcopy(X)
        self._y = copy.deepcopy(Y)
        
        self.n_samples, n_features = self._X.shape
        if self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")
        
        if self.target_gp is None:
            self.target_gp = GP(
            RBF(self.n_features, ARD=True),
            noise_variance=0.1,
        )

        residuals = self._compute_residuals(X, Y)

        self.target_gp.fit(X, residuals, optimize)

    def predict(
        self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )

        # returned mean: sum of means of the predictions of all source and target GPs
        mu = self.predict_posterior_mean(X)

        # returned variance is the variance of target GP
        _, var = self.target_gp.predict(
            X, return_full=return_full, with_noise=with_noise
        )

        return mu, var

    def predict_posterior_mean(self, X: np.ndarray, idx: int = None) -> np.ndarray:
        """Predict the mean function for given test point(s).

        For `idx=None` returns the same as `self.predict(data)[0]` but avoids the
        overhead coming from predicting the variance. If `idx` is specified, returns
        the sum of all the means up to the `idx`-th GP. Useful for inspecting the inner
        state of the stack.

        Args:
            data: Input data to predict on.
                Data is provided as ndarray with shape = (n_points, n_features).
            idx: Integer of the GP in the stack. Counting starts from the bottom at
                zero. If `None`, the mean prediction of the entire stack is returned.

        Returns:
            Predicted mean for every input. `shape = (n_points, 1)`
        """

        all_gps = self.source_gps + [self.target_gp]

        if idx is None:  # if None, the target GP is considered
            idx = len(all_gps) - 1

        mu = np.zeros((X.shape[0], 1))
        # returned mean is a sum of means of the predictions of all GPs below idx
        for model in all_gps[: idx + 1]:
            mu += model.predict_posterior_mean(X)

        return mu

    def predict_posterior_covariance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Posterior covariance between two inputs.

        Args:
            x1: First input to be queried. `shape = (n_points_1, n_features)`
            x2: Second input to be queried. `shape = (n_points_2, n_features)`

        Returns:
            Posterior covariance at `(x1, x2)`. `shape = (n_points_1, n_points_2)`
        """
        return self.target_gp.predict_posterior_covariance(x1, x2)
    
    def get_fmin(self):

        return np.min(self._y)