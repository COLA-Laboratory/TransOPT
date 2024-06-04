import copy

import numpy as np
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles

from transopt.agent.registry import acf_registry
from transopt.optimizer.acquisition_function.acf_base import AcquisitionBase


@acf_registry.register('TAF')
class AcquisitionTAF(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, config):
        super(AcquisitionTAF, self).__init__()
        if 'jitter' in config:
            self.jitter = config['jitter']
        else:
            self.jitter = 0.01

        if 'threshold' in config:
            self.threshold = config['threshold']
        else:
            self.threshold = 0

        self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):
        n_sample = len(x)
        source_num = len(self.model._source_gps)
        n_models = source_num + 1
        acf_ei = np.empty((n_models, n_sample, 1))

        for task_uid in range(source_num):
            m, s = self.model._source_gps[task_uid].predict(x)
            _X = self.model._source_gps[task_uid]._X
            fmin = self.model._source_gps[task_uid].predict(_X)[0].min()
            phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
            acf_ei[task_uid] =  s * (u * Phi + phi)
        m,s = self.model.predict(x)
        for task_uid in range(source_num):
            acf_ei[task_uid] = acf_ei[task_uid] * self.model._source_gp_weights[task_uid]
        acf_ei[-1] = self.model._target_model_weight
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        acf_ei[-1] = acf_ei[-1] * (s * (u * Phi + phi))
        f_acqu_ei = np.sum(acf_ei, axis=0)

        return f_acqu_ei

    def _compute_acq_withGradients(self, x):
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        raise NotImplementedError()

