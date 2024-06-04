# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase

from transopt.agent.registry import acf_registry
from transopt.optimizer.acquisition_function.acf_base import AcquisitionBase

@acf_registry.register('LCB')
class AcquisitionLCB(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function with constant exploration weight.
    See:

    Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
    Srinivas et al., Proc. International Conference on Machine Learning (ICML), 2010

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = False

    def __init__(self, config):
        super(AcquisitionLCB, self).__init__()
        if 'exploration_weight' in config:
            self.exploration_weight = config['exploration_weight']
        else:
            self.exploration_weight = 1

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x)
        f_acqu = -m + self.exploration_weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu

