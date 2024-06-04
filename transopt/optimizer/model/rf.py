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
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler

from transopt.optimizer.model.model_base import  Model
from transopt.agent.registry import model_registry
from sklearn.ensemble import RandomForestRegressor


@model_registry.register('RF')
class RF(Model):
    def __init__(
        self,
        name = 'RandomForest',
        num_estimators = 100,
        seed = 0,
        normalize: bool = True,
        **options: dict
    ):
        """Initialize the Method.
        """
        super().__init__()
        self.name = name
        self.num_estimators = num_estimators

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_features='sqrt',
            bootstrap=True,
            random_state=seed
        )

        self._normalize = normalize
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None

        self._options = options

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimize: bool = True,
    ):
        self._X = np.copy(X)
        self._y = np.copy(Y)
        self._Y = np.copy(Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)

        if self._normalize:
            _X = self._x_normalizer.fit_transform(_X)
            _y = self._y_normalizer.fit_transform(_y)
        self.model.fit(_X, _y)


    def predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, var = self._raw_predict(X, return_full, with_noise)

        return mean, var

    def _raw_predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict functions distribution(s) for given test point(s) without taking into
        account data normalization. If `self._normalize` is `False`, return the same as
        `self.predict()`.

        Same input/output as `self.predict()`.
        """
        _X_test = X.copy()

        mu = self.model.predict(_X_test)
        cov = self._raw_predic_var(_X_test, self.model, mu)
        return mu[:,np.newaxis], cov[:,np.newaxis]

    def _raw_predic_var(self, X, trees, predictions, min_variance=0.0):
        # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
        std = np.zeros(len(X))

        for tree in trees:
            var_tree = tree.tree_.impurity[tree.apply(X)]

            var_tree[var_tree < min_variance] = min_variance
            mean_tree = tree.predict(X)
            std += var_tree + mean_tree ** 2

        std /= len(trees)
        std -= predictions ** 2.0
        std[std < 0.0] = 0.0
        std = std ** 0.5
        return std

    def sample(
        self, X, size: int = 1, with_noise: bool = False
    ) -> np.ndarray:
        """Perform model inference.

        Sample functions from the posterior distribution for the given test points.

        Args:
            data: Input data to predict on. `shape = (n_points, n_features)`
            size: Number of functions to sample.
            with_noise: If `False`, the latent function `f` is considered. If `True`,
                the observed function `y` that includes the noise variance is
                considered.

        Returns:
            Sampled function value for every input. `shape = (n_points, size)`
        """
        mean, cov = self.predict(X, return_full=True, with_noise=with_noise)
        mean = mean.flatten()
        sample = np.random.multivariate_normal(mean, cov, size).T
        return sample

    def get_fmin(self):
        if self._normalize:
            return np.min(self._y_normalizer.inverse_transform(self._y))
        else:
            return np.min(self._y)