import GPy
import numpy as np
from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import model_registry
from transopt.optimizer.model.model_base import Model


@model_registry.register("SMSEGO")
class SMSEGO(Model):
    def __init__(self, seed=0, normalize=True, **options):
        super().__init__()
        self.seed = seed
        self.normalize = normalize
        self.models = []
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None
        self._options = options
        np.random.seed(self.seed)

    def fit(self, X, Y):
        self._X = np.copy(X)
        self._Y = np.copy(Y)
        if self.normalize:
            X = self._x_normalizer.fit_transform(X)
            Y = self._y_normalizer.fit_transform(Y.T).T  # Transpose Y to normalize across objectives
        self._create_model(X, Y)

    def predict(self, X, full_cov=False):
        return self._make_prediction(X, full_cov)

    def _create_model(self, X, Y):
        for i in range(self.num_objective):
            kernel = GPy.kern.RBF(input_dim=X.shape[1])
            model = GPy.models.GPRegression(X, Y[i][:, np.newaxis], kernel=kernel)
            model[".*Gaussian_noise.variance"].constrain_fixed(1.0e-4)
            model[".*rbf.variance"].constrain_fixed(1.0)
            self.models.append(model)

    def _update_model(self, X, Y):
        if not self.models:
            self._create_model(X, Y)
        else:
            for i, model in enumerate(self.models):
                model.set_XY(X, Y[i][:, np.newaxis])
        
        try:
            for model in self.models:
                model.optimize_restarts(num_restarts=1, verbose=self._options.get("verbose", False), robust=True)
        except np.linalg.linalg.LinAlgError as e:
            print("Error during model optimization: ", e)

    def _make_prediction(self, X, full_cov=False):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        pred_mean = np.zeros((X.shape[0], 0))
        pred_var = np.zeros((X.shape[0], 0)) if not full_cov else np.zeros((0, X.shape[0], X.shape[0]))
        
        for model in self.models:
            mean, var = model.predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        
        return pred_mean, pred_var
