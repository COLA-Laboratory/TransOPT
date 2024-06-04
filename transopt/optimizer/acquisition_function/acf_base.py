# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
import scipy
from GPyOpt import Design_space
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.util import epmgp


class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer

    """

    analytical_gradient_prediction = False

    def __init__(self, cost_withGradients=None, **kwargs):
        self.analytical_gradient_acq = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

        if 'optimizer_name' in kwargs:
            self.optimizer_name = kwargs['optimizer']
        else:
            self.optimizer_name = 'lbfgs'

        if cost_withGradients is  None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()
    
    def link(self, model, space):
        self.link_model(model=model)
        self.link_space(space=space)
    
    def link_model(self, model):
        self.model = model
        
    def link_space(self, space):
        opt_space = []
        for var_name in space.variables_order:
            var_dic = {
                'name': var_name,
                'type': 'continuous',
                'domain': space[var_name].search_space_range,
            }
            if space[var_name].type == 'categorical' or 'integer':
                var_dic['type'] = 'discrete'

            opt_space.append(var_dic.copy())
            
        self.space = Design_space(opt_space)
        self.optimizer = AcquisitionOptimizer(self.space, self.optimizer_name)
    

    def acquisition_function(self,x):
        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)
        x_z = x if self.space.model_dimensionality == self.space.objective_dimensionality else self.space.zip_inputs(x)
        return -(f_acqu*self.space.indicator_constraints(x_z))/cost_x


    def acquisition_function_withGradients(self, x):
        """
        Takes an acquisition and it gradient and weights it so the domain and cost are taken into account.
        """
        f_acqu,df_acqu = self._compute_acq_withGradients(x)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        x_z = x if self.space.model_dimensionality == self.space.objective_dimensionality else self.space.zip_inputs(x)
        return -f_acq_cost*self.space.indicator_constraints(x_z), -df_acq_cost*self.space.indicator_constraints(x_z)

    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        if not self.analytical_gradient_acq:
            out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager)
        else:
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients, duplicate_manager=duplicate_manager)
        return out

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')
