import copy
from typing import Dict, List, Sequence, Union

import GPy
import numpy as np
from GPy.kern import RBF, Kern

from transopt.agent.registry import model_registry
from transopt.optimizer.model.gp import GP
from transopt.optimizer.model.model_base import Model


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)

@model_registry.register("SGPT")
class SGPT(Model):
    def __init__(
            self,
            kernel: Kern = None,
            noise_variance: float = 1.0,
            normalize: bool = True,
            Seed = 0,
            bandwidth: float = 1,
            **options: dict,
    ):
        super().__init__()
        # GP on difference between target data and last source data set
        self._noise_variance = noise_variance
        self._metadata = {}
        self._source_gps = {}
        self._source_gp_weights = {}
        self._normalize = normalize
        self.Seed = Seed
        self.rng = np.random.RandomState(self.Seed)
        
        self._metadata = {}
        self._source_gps = {}
        self._source_gp_weights = {}
        self.bandwidth =bandwidth

        self._target_model = None
        self._target_model_weight = 1
    
    
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
                
        kernel = RBF(self.n_features, ARD=True)
        new_gp = GP(
            kernel, noise_variance=self._noise_variance
        )
        new_gp.fit(
            X = X,
            Y = Y,
            optimize = optimize,
        )
        return new_gp
    
    def meta_fit(self,
            source_X : List[np.ndarray],
            source_Y : List[np.ndarray],
            optimize: Union[bool, Sequence[bool]] = True):
        # metadata, _ = SourceSelection.the_k_nearest(source_datasets)

        self._metadata = {'X': source_X, 'Y':source_Y}
        self._source_gps = {}
        
        
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
            self._source_gps[i] = new_gp

        self._calculate_weights()


    def fit(self, 
            X: np.ndarray,
            Y: np.ndarray,
            optimize: bool = False):

        self._X = copy.deepcopy(X)
        self._Y = copy.deepcopy(Y)

        self.n_samples, n_features = self._X.shape
        if self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")

        kern = GPy.kern.RBF(self.n_features, ARD=False)

        self._target_model = GPy.models.GPRegression(self._X, self._Y, kernel=kern)
        self._target_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)

        try:
            self._target_model.optimize_restarts(num_restarts=1, verbose=False, robust=True)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')

        self._calculate_weights()


    def predict(self, X, return_full: bool = False, with_noise: bool = False):
        X_test = X
        n_models = len(self._source_gp_weights)
        if self._target_model_weight > 0:
            n_models += 1
        n_sample = X_test.shape[0]
        means = np.empty((n_models, n_sample, 1))
        weights = np.empty((n_models, n_sample))
        if return_full == False:
            vars_ = np.empty((n_models, n_sample, 1))
        else:
            vars_ = np.empty((n_models, n_sample, n_sample))
        for task_uid, weight in enumerate(self._source_gp_weights):
            means[task_uid], vars_[task_uid] = self._source_gps[task_uid].predict(X_test)
            weights[task_uid] = weight
        if self._target_model_weight > 0:
            means[-1], vars_[-1] = self._target_model.predict(X_test)
            weights[-1] = self._target_model_weight

        weights = weights[:,:,np.newaxis]
        mean = np.sum(weights * means, axis=0)
        return mean, vars_[-1]

    def Epanechnikov_kernel(self, X1, X2):
        diff_matrix = X1 - X2
        u = np.linalg.norm(diff_matrix, ord=2) / self.bandwidth**2  # 计算归一化距离
        if u < 1:
            weight = 0.75 * (1 - u**2)  # 根据 Epanechnikov 核计算权重
        else:
            weight = 0 
        return weight
    
    def _calculate_weights(self, alpha: float = 0.0):
        if self._X is None:
            weight = 1 / len(self._source_gps)
            self._source_gp_weights = [weight for task_uid in self._source_gps]
            self._target_model_weight = 0
            return

        predictions = []
        for model_idx in range(len(self._source_gps)):
            model = self._source_gps[model_idx]
            predictions.append(model.predict(self._X)[0].flatten())  # ndarray(n,)


        predictions.append(self._target_model.predict(self._X)[0].flatten())
        predictions = np.array(predictions)

        bootstrap_indices = self.rng.choice(predictions.shape[1],
                                            size=(self.n_samples, predictions.shape[1]),
                                            replace=True)

        bootstrap_predictions = []
        bootstrap_targets = self._Y[bootstrap_indices].reshape((self.n_samples, len(self._Y)))
        for m in range(len(self._source_gps) + 1):
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((len(self._source_gps) + 1, self.n_samples))
        for i in range(len(self._source_gps)):
            for j in range(1, len(self._Y)):
                ranking_losses[i] += np.sum(
                    (
                        ~(roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                       ), axis=1

                )
        for j in range(1, len(self._Y)):
            ranking_losses[-1] += np.sum(
                (
                        ~((roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets))
                ), axis=1
            )
        total_compare = len(self._Y) *(len(self._Y - 1))
        ranking_loss = np.array(ranking_losses) / total_compare

        weights = [self.Epanechnikov_kernel(ranking_loss[task_uid], ranking_loss[-1]) for task_uid in self._source_gps]
        weights.append(1.0)
        weights = np.array(weights)/np.sum(weights)
        self._source_gp_weights = [weights[task_uid] for task_uid in self._source_gps]
        self._target_model_weight = weights[-1]

    def posterior_samples_f(self,X, size=10, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """


        predict_kwargs["full_cov"] = True  # Always use the full covariance for posterior samples.
        m, v = self._raw_predict(X,  **predict_kwargs)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """


        fsim = self.posterior_samples_f(X, size, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = likelihood.samples(fsim[:, d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim

    def get_fmin(self):

        return np.min(self._Y)