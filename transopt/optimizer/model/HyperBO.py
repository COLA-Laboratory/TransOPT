import random
import time

from external.hyperbo.basics import definitions as defs
from external.hyperbo.basics import params_utils
from external.hyperbo.gp_utils import gp
from external.hyperbo.gp_utils import kernel
from external.hyperbo.gp_utils import mean
from external.hyperbo.gp_utils import utils
from external.hyperbo.bo_utils import data
from external.hyperbo.gp_utils import objectives as obj
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Tuple, Union

font = {
    'family': 'serif',
    'weight': 'normal',
    'size': 7,
}
axes = {'titlesize': 7, 'labelsize': 7}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
SubDataset = defs.SubDataset

class hyperbo():
    def __init__(self, seed = 0):
        self.mean_func = mean.constant
        self.cov_func = kernel.squared_exponential
        self.warp_func = DEFAULT_WARP_FUNC
        self.key = jax.random.PRNGKey(seed)
        self._X = None
        self._Y = None

        self.params = GPParams(
            model={
                'constant': 5.,
                'lengthscale': 1.,
                'signal_variance': 1.0,
                'noise_variance': 0.01,
            },
            config={
                'Method': 'adam',
                'learning_rate': 1e-5,
                'beta': 0.9,
                'max_training_step': 1,
                'batch_size': 100,
                'retrain': 1,
            })

    def pretrain(self, Meta_data, Target_data):
        dataset = {}
        num_train_functions = len(Meta_data['X'])
        for sub_dataset_id in range(num_train_functions):
            x = jax.numpy.array(Meta_data['X'][sub_dataset_id])
            y = jax.numpy.array(Meta_data['Y'][sub_dataset_id])
            dataset[str(sub_dataset_id)] = SubDataset(x, y)

        self.target_dataset_id = num_train_functions
        self._X = Target_data['X']
        self._Y = Target_data['Y']
        x = jax.numpy.array(self._X)
        y = jax.numpy.array(self._Y)
        dataset[str(self.target_dataset_id)] = SubDataset(x, y)

        self.model = gp.GP(
            dataset=dataset,
            params=self.params,
            mean_func=self.mean_func,
            cov_func=self.cov_func,
            warp_func=self.warp_func,
        )
        assert self.key is not None, ('Cannot initialize with '
                                             'init_random_key == None.')
        key, subkey = jax.random.split(self.key)
        self.model.initialize_params(subkey)
        # Infer GP parameters.
        key, subkey = jax.random.split(self.key)
        self.model.train(subkey)

    def retrain(self, Target_data):
        self._X = Target_data['X']
        self._Y = Target_data['Y']
        x = jax.numpy.array(self._X)
        y = jax.numpy.array(self._Y)
        dataset =  SubDataset(x, y)

        self.model.update_sub_dataset(
            dataset, sub_dataset_key=str(self.target_dataset_id), is_append=False)

        retrain_condition = 'retrain' in self.model.params.config and self.model.params.config[
            'retrain'] > 0 and self.model.dataset[str(self.target_dataset_id)].x.shape[0] > 0
        if not retrain_condition:
            return
        if self.model.params.config['objective'] in [obj.regkl, obj.regeuc]:
            raise ValueError('Objective must include NLL to retrain.')
        max_training_step = self.model.params.config['retrain']
        self.model.params.config['max_training_step'] = max_training_step
        key, subkey = jax.random.split(self.key)
        self.model.train(subkey)

    def predict(self, X, subset_data_id:Union[int, str] = 0):
        _X = jnp.array(X)
        mu, var = self.model.predict(_X, subset_data_id)

        return mu, var
