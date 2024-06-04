import GPy
import numpy as np
import scipy.optimize as opt
from scipy.stats import *
from scipy.spatial import distance
from GPyOpt.util.general import get_quantiles

from agent.registry import acf_registry
from transopt.utils.hypervolume import calc_hypervolume
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer


@acf_registry.register("MOEADEGO")
class MOEADEGO:
    def __init__(self, model, space, optimizer, config):
        self.optimizer = optimizer
        self.model = model
        self.model_id = 0
        if 'jitter' in config:
            self.jitter = config['jitter']
        else:
            self.jitter = 0.1

        if 'threshold' in config:
            self.threshold = config['threshold']
        else:
            self.threshold = 0
    def _compute_acq(self, x):
        m, s = self.model.predict_by_id(x, self.model_id)
        fmin = self.model.get_fmin_by_id(self.model_id)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu_ei = s * (u * Phi + phi)

        return -f_acqu_ei

    def set_model_id(self, idx):
        self.model_id = idx
    def optimize(self, duplicate_manager=None):
        space = self.model.search_space
        self.acquisition_optimizer = AcquisitionOptimizer(space, 'lbfgs')  ## more arguments may come here
        suggested_sample = []
        suggested_acfvalue = []
        for i in range(len(self.model.model_list)):
            self.set_model_id(i)
            suggest_x, acf_value = self.acquisition_optimizer.optimize(self._compute_acq)
            suggested_sample.append(suggest_x)
            suggested_acfvalue.append(acf_value)
        suggested_sample = np.vstack(suggested_sample)
        suggested_acfvalue = np.vstack(suggested_acfvalue)
        return suggested_sample, suggested_acfvalue

