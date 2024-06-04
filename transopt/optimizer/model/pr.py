import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from transopt.optimizer.model.model_base import  Model
from transopt.optimizer.model.utils import is_pd, nearest_pd
from transopt.agent.registry import model_registry
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


@model_registry.register('PR')
class PR(Model):
    def __init__(
        self,
        degree: int = 10,
        normalize: bool = True,
        **options: dict
    ):
        super().__init__()
        self._degree = degree
        self._pr_model = None

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

        if self._pr_model is None:
            self._poly_features = PolynomialFeatures(degree=self._degree)
            X_poly = self._poly_features.fit_transform(_X)
            self._pr_model = LinearRegression()
            self._pr_model.fit(X_poly, _y)
        else:
            X_poly = self._poly_features.fit_transform(_X)
            self._pr_model.fit(X_poly, _y)

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_poly = self._poly_features.transform(X)
        Y = self._pr_model.predict(X_poly)
        return Y, None