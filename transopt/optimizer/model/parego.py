import GPy
import numpy as np
from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import model_registry
from transopt.optimizer.model.model_base import Model


@model_registry.register("ParEGO")
class ParEGO(Model):
    def __init__(self, seed=0, normalize=True, **options):
        super().__init__()
        self.seed = seed
        self.normalize = normalize
        self.models = None
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None
        self._options = options
        self.rho = 0.1

    def fit(self, X, Y):
        self._X = np.copy(X)
        self._Y = np.copy(Y)
        if self.normalize:
            X = self._x_normalizer.fit_transform(X)
            Y = self._y_normalizer.fit_transform(Y)
        self._update_model(X, Y)

    def predict(self, X, full_cov=False):
        return self._make_prediction(X, full_cov)

    def _scalarization(self, Y: np.ndarray, rho):
        theta = np.random.random_sample(Y.shape[0])
        sum_theta = np.sum(theta)
        theta = theta / sum_theta

        theta_f = Y.T * theta
        max_k = np.max(theta_f, axis=1)
        rho_sum_theta_f = rho * np.sum(theta_f, axis=1)

        return max_k + rho_sum_theta_f
    
    def _create_model(self, X, Y):
        kernel = GPy.kern.RBF(input_dim=X.shape[1])
        model = GPy.models.GPRegression(X, Y, kernel=kernel, normalizer=None)
        model[".*Gaussian_noise.variance"].constrain_fixed(1.0e-4)
        model[".*rbf.variance"].constrain_fixed(1.0)
        self.model = model

    def _update_model(self, X, Y):
        Y_scalar = self._scalarization(Y, self.rho)[:, np.newaxis]
        
        if not self.model:
            self._create_model(X, Y_scalar)
        else:
            self.model.set_XY(X, Y_scalar)
         
        try:
            self.model.optimize_restarts(num_restarts=1, verbose=self._options.get("verbose", False), robust=True)
        except np.linalg.linalg.LinAlgError as e:
            print("Error during model optimization: ", e)
    
    def _make_prediction(self, X, full_cov=False):
        pred_mean = np.zeros((X.shape[0], 0))
        pred_var = np.zeros((X.shape[0], 0)) if not full_cov else np.zeros((0, X.shape[0], X.shape[0]))
        
        if self.model:
            mean, var = self.model.predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        
        return pred_mean, pred_var
 