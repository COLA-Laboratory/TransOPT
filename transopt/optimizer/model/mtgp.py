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
from typing import Dict, List, Tuple

import numpy as np
from GPy.kern import Kern, RBF
from GPy.models import GPCoregionalizedRegression
from GPy.util.multioutput import ICM

from transopt.optimizer.model.gp import GP
from transopt.optimizer.model.utils import is_pd, nearest_pd
from transopt.agent.registry import model_registry

@model_registry.register("MTGP")
class MTGP(GP):
    r"""Multi-Task-Single-k GP, a GP-based transfer-learning algorithm.

    Multi-Task-Single-kGP models the source and target data on an equal footing with no
    explicit hierarchy. Correlations within tasks are assumed to be different than
    those across tasks. Also known as coregionalized regression model,
    Multi-Task-Single-k GP models the data with a kernel of the form

    $$
        \begin{bmatrix}
        k((x, s), (x', s)) &  k((x, s), (x', t)) \\
        k((x, t), (x', s)) &  k((x, t), (x', t))
        \end{bmatrix}
        =
        \begin{bmatrix}
        W_{ss} & W_{st} \\
        W_{st} & W_{tt}
        \end{bmatrix}
        k(x, x'),
    $$

    where $\mathbf{W}$ is a positive semi-definite matrix also known as
    coregionalization matrix.

    Multi-Task-Single-k GP is a powerful but computationally expensive Method since (i)
    it scales cubically with the total number of data points and (ii) the number of
    hyperparameters scales quadratically with the number of tasks.
    """

    def __init__(
        self,
        kernel: Kern = None,
        noise_variance: float = 1.0,
        normalize: bool = True,
        **options: dict,
    ):
        super().__init__(kernel, noise_variance, normalize, **options)
        self._normalize = normalize
        self._kernel = kernel
        self._multikernel = None
        self._gpy_model = None
        self._noise_variance = []
        self.n_sources = None
        self.n_features = None

        self._metadata_x = []
        self._metadata_y = []

        self._options = options

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        data_X = copy.deepcopy(source_X)
        data_Y = copy.deepcopy(source_Y)
        self.n_sources = len(data_X)

        # create list of input/observed values from source data
        for i in range(self.n_sources):
            self._metadata_x = self._metadata_x + data_X
            self._metadata_y = self._metadata_y + data_Y
        self.n_features = self._metadata_x[0].shape[-1]

    def fit(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        optimize: bool = False,
    ):
        if not self._metadata_x:
            raise ValueError(
                "Error: source data not available. Forgot to call `meta_fit`."
            )

        self._X = np.copy(X)
        self._y = np.copy(Y)

        # add target data to the list of input/observed values
        x_list = copy.deepcopy(self._metadata_x)
        y_list = copy.deepcopy(self._metadata_y)
        x_list.append(X)
        y_list.append(Y)

        if self._normalize:
            # add source order to data lists
            for i in range(len(x_list)):
                x_list[i] = np.hstack(
                    [x_list[i], np.zeros((x_list[i].shape[0], 1)) + i]
                )
                y_list[i] = np.hstack(
                    [y_list[i], np.zeros((y_list[i].shape[0], 1)) + i]
                )
            # merge all data into one array, normalize data
            x_all = np.vstack(x_list)
            x_all[:, :-1] = self._x_normalizer.fit_transform(x_all[:, :-1])
            y_all = np.vstack(y_list)
            y_all[:, :-1] = self._y_normalizer.fit_transform(y_all[:, :-1])
            # transform data back to original list of arrays
            for i in range(len(x_list)):
                x_list[i] = x_all[np.where(x_all[:, -1] == i)][:, :-1]
                y_list[i] = y_all[np.where(y_all[:, -1] == i)][:, :-1]

        # define multiple output kernel
        if self._kernel is None:
            self._kernel = RBF(self.n_features)
        multikernel = ICM(
            input_dim=self.n_features,
            num_outputs=self.n_sources + 1,
            kernel=self._kernel,
        )

        # fit model to current data
        self._gpy_model = GPCoregionalizedRegression(x_list, y_list, kernel=multikernel)

        if optimize:
            optimize_restarts_options = self._options.get(
                "optimize_restarts_options", {}
            )

            kwargs = copy.deepcopy(optimize_restarts_options)

            if "verbose" not in optimize_restarts_options:
                kwargs["verbose"] = False

            self._gpy_model.optimize_restarts(**kwargs)

        self._multikernel = self._gpy_model.kern.copy()

        # noise variance: each element corresponds to the noise of one output
        self._noise_variance = self._gpy_model.likelihood.param_array

    def _raw_predict(
        self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        _X = X.copy()

        if self._X is None:
            mu = np.zeros((_X.shape[0], 1))
            cov = self._kernel.K(_X)
            var = np.diag(cov)[:, None]
            return mu, cov if return_full else var

        if self._normalize:
            _X = self._x_normalizer.transform(_X)

        # predictions are made for the last output, which corresponds to the target;
        # prepare extended input format + associated noise model
        _X_test = np.hstack([_X, np.ones((_X.shape[0], 1)) * self.n_sources])
        noise_dict = {"output_index": _X_test[:, -1:].astype(int)}

        # ensure that no negative variance is predicted
        mu, cov = self._gpy_model.predict(
            _X_test,
            full_cov=return_full,
            include_likelihood=with_noise,
            Y_metadata=noise_dict,
        )
        if return_full:
            if not is_pd(cov):
                cov = nearest_pd(cov)
        else:
            cov = np.clip(cov, 1e-20, None)
        return mu, cov
