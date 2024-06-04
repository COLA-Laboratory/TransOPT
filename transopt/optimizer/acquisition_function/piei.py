import copy

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
import numpy as np
from scipy.stats import norm
from GPyOpt.acquisitions.LCB import AcquisitionLCB

class AcquisitionpiEI(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, Model, space, optimizer, cost_withGradients=None, jitter=0.01, threshold=0.):
        self.optimizer = optimizer
        super(AcquisitionpiEI, self).__init__(Model, space, optimizer)
        self.Model = Model
        self.jitter = jitter
        self.threshold = threshold
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('EIC acquisition does now make sense with cost at present. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):

        m, s = self.Model.predict(x)
        # fmin = self.CBOModel.get_valid_fmin()
        fmin = self.Model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu_ei = s * (u * Phi + phi)

        return f_acqu_ei * self._compute_prior(x)

    def _compute_prior(self, x):
        return 1

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
