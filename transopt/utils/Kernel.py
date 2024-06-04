import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.mappings.constant import Constant
from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference



def construct_multi_objective_kernel(input_dim, output_dim, base_kernel='RBF', Q=1, rank=2):
    # Choose the base kernel. Note: This part can be improved since it currently always chooses RBF.
    k = GPy.kern.RBF(input_dim=input_dim)

    kernel_list = [k] * Q
    j = 1
    kk = kernel_list[0]
    K = kk.prod(
        GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None, kappa=None,
                               name='B'), name='%s%s' % ('ICM', 0))
    for kernel in kernel_list[1:]:
        K += kernel.prod(
            GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], rank=rank, W=None,
                                   kappa=None, name='B'), name='%s%s' % ('ICM', j))
        j += 1
    return K