import numpy as np
from GPy.kern import RBF, Kern
from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import model_registry
from transopt.optimizer.model.gp import GP
from transopt.optimizer.model.model_base import Model
from transopt.utils.weights import init_weight, tchebycheff


@model_registry.register("MOEAD-EGO")
class MoeadEGO(Model):
    def __init__(
        self,
        num_objective: int,
        name="MoeadEGO",
        num_weights=10,
        seed=0,
        normalize: bool = True,
        **options: dict
    ):
        super().__init__()
        self.name = name
        self.num_weights = num_weights
        self.num_objective = num_objective
        self.normalize = normalize
        self.seed = seed
        self.weights = init_weight(self.num_objective, self.num_weights)
        self.models = []
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None
        self._options = options
        self._initialize_weights()

    def fit(self, X, Y):
        self._X = np.copy(X)
        self._Y = np.copy(Y)
        if self.normalize:
            X = self._x_normalizer.fit_transform(X)
            Y = self._y_normalizer.fit_transform(Y)
        self._update_model(X, Y)

    def predict(self, X, full_cov=False):
        return self._make_prediction(X, full_cov)

    def _create_model(self, X, Y):
        self.models = []
        ideal_point = np.min(Y.T, axis=0)
        for i, weight in enumerate(self.weights):
            kernel = RBF(input_dim=X.shape[1])
            Y_weighted = tchebycheff(Y.T, weight, ideal=ideal_point)
            model = GP(
            kernel, noise_variance=self._noise_variance
            )
            model.fit(
                X = X,
                Y = Y_weighted,
                optimize = True,
            )
            model[".*Gaussian_noise.variance"].constrain_fixed(1.0e-4)
            model[".*rbf.variance"].constrain_fixed(1.0)
            self.models.append(model)

    def _update_model(self, X, Y):
        if not self.models:
            self._create_model(X, Y)
        else:
            ideal_point = np.min(Y.T, axis=0)
            for i, model in enumerate(self.models):
                Y_weighted = tchebycheff(Y.T, self.weights[i], ideal=ideal_point)
                model.set_XY(X, Y_weighted[:, np.newaxis])

        try:
            for model in self.models:
                model.optimize_restarts(
                    num_restarts=1, verbose=self.verbose, robust=True
                )
        except np.linalg.linalg.LinAlgError as e:
            print("Error during model optimization: ", e)

    def _make_prediction(self, X, full_cov=False):
        pred_mean = np.zeros((X.shape[0], 0))
        pred_var = (
            np.zeros((X.shape[0], 0))
            if not full_cov
            else np.zeros((0, X.shape[0], X.shape[0]))
        )

        for model in self.model_list:
            mean, var = model.predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        return pred_mean, pred_var

    def _make_prediction_by_id(self, X, idx, full_cov=False):
        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))
            mean, var = self.model_list[idx].predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        return pred_mean, pred_var
