import GPy
import numpy as np
import scipy.optimize as opt
from scipy.stats import *
from scipy.spatial import distance
from GPyOpt.acquisitions.base import AcquisitionBase

from agent.registry import acf_registry
from transopt.utils.hypervolume import calc_hypervolume


@acf_registry.register("SMSEGO")
class SMSEGO:
    def __init__(self, model, space, optimizer, config):
        self.optimizer = optimizer
        self.model = model
        self.const = 1 / norm.cdf(0.5 + 1 / 2**self.model.num_objective)
        self.current_hypervolume = None
        self.w_ref = None

    def _compute_acq(self, x):
        if self.w_ref is None:
            self.w_ref = self.model._Y.max(axis=1) + 1.0e2
        if self.current_hypervolume is None:
            self.current_hypervolume = calc_hypervolume(self.model._Y.T, self.w_ref)

        if np.any(np.all(self.model._X == x, axis=1)):
            return 1.0e5
        else:
            mean, var = self.model.predict(x)
            lcb = mean - self.const * np.sqrt(var)
            new_y_train = np.hstack((self.model._Y, lcb.T)).T

            new_hypervolume = calc_hypervolume(new_y_train, self.w_ref)
            smsego = self.current_hypervolume - new_hypervolume
            # print(smsego)
            return smsego

    def optimize(self, duplicate_manager=None):
        x_bounds = self.model._get_var_bound("search")
        default = np.array([(v[1] + v[0]) / 2 for _, v in x_bounds.items()])
        bounds = [(v[0], v[1]) for _, v in x_bounds.items()]
        result = opt.minimize(
            self._compute_acq, x0=default, bounds=bounds, method="L-BFGS-B"
        )
        return result.x[np.newaxis, :], result.fun
