'''
Pareto-related tools.
'''

import numpy as np
from collections.abc import Iterable
from pymoo.indicators.hv import Hypervolume


def convert_minimization(Y, obj_type=None):
    '''
    Convert maximization to minimization.

    Example usage:
    Y = np.array([[1, 4, 3], [2, 1, 4], [3, 2, 2]])
    obj_type = ['min', 'max', 'min']
    Y_minimized = convert_minimization(Y, obj_type)
    '''
    if obj_type is None: 
        return Y

    if isinstance(obj_type, str):
        obj_type = [obj_type] * Y.shape[1]
    assert isinstance(obj_type, Iterable), f'Objective type {type(obj_type)} is not supported'

    maxm_idx = np.array(obj_type) == 'max'
    Y = Y.copy()
    Y[:, maxm_idx] = -Y[:, maxm_idx]

    return Y

def find_pareto_front(Y, return_index=False, obj_type=None, eps=1e-8):
    '''
    Find pareto front (undominated part) of the input performance data.
    '''
    if len(Y) == 0: return np.array([])

    Y = convert_minimization(Y, obj_type)

    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y[idx] - Y > -eps).all(axis=1), (Y[idx] - Y > eps).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = np.atleast_2d(Y[pareto_indices].copy())

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front
    

def check_pareto(Y, obj_type=None):
    '''
    Check pareto optimality of the input performance data

    Example usage:
    Y = np.array([[1, 2], [2, 1], [1.5, 1.5]])
    pareto_optimal = check_pareto(Y)
    '''
    Y = convert_minimization(Y, obj_type)

    # find pareto indices
    sorted_indices = np.argsort(Y.T[0])
    pareto = np.zeros(len(Y), dtype=bool)
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto[idx] = True
    return pareto


def calc_hypervolume(Y, ref_point, obj_type=None):
    '''
    Calculate hypervolume

    Example usage:
    Y = np.array([[1, 2], [2, 1], [1.5, 1.5]])
    ref_point = np.array([2.5, 2.5])
    hypervolume = calc_hypervolume(Y, ref_point)
    '''
    Y = convert_minimization(Y, obj_type)

    return Hypervolume(ref_point=ref_point).do(Y)


def calc_pred_error(Y, Y_pred_mean, average=False):
    '''
    Calculate prediction error
    '''
    assert len(Y.shape) == len(Y_pred_mean.shape) == 2
    pred_error = np.abs(Y - Y_pred_mean)
    if average:
        pred_error = np.sum(pred_error, axis=0) / len(Y)
    return pred_error