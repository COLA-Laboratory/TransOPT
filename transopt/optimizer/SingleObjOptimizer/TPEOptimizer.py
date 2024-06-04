import numpy as np
from typing import Dict, List, Union
from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register

from transopt.utils.Normalization import get_normalizer


@optimizer_register('TPE')
class TPEOptimizer(BOBase):
    def __init__(self, config:Dict, **kwargs):
        super(TPEOptimizer, self).__init__(config=config)

        self.init_method = 'Random'
        self.model = None

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = True

        if 'init_number' in config:
            self.ini_num = config['init_number']
        else:
            self.ini_num = None

        if 'acf' in config:
            self.acf = config['acf']
        else:
            self.acf = 'EI'

        self.obj_model = None



    def initial_sample(self):
        return self.random_sample(self.ini_num)

    def suggest(self, n_suggestions:Union[None, int] = None) ->List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if 'normalize' in self.config:
                self.normalizer = get_normalizer(self.config['normalize'])


            Data = {'Target':{'X':self._X, 'Y':self._Y}}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'), suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def update_model(self, Data):
        assert 'Target' in Data
        target_data = Data['Target']
        X = target_data['X']
        Y = target_data['Y']

        if self.normalizer is not None:
            Y = self.normalizer(Y)

        if self.obj_model == None:
            self.create_model(X, Y)
        else:
            self.obj_model.set_XY(X, Y)

        try:
            self.obj_model.optimize_restarts(num_restarts=1, verbose=self.verbose, robust=True)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')

    def create_model(self, X, Y):
        self.obj_model = TPEOptimizer()



    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]

        m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v


    def random_sample(self, num_samples: int) -> List[Dict]:
        """
        Initialize random samples.

        :param num_samples: Number of random samples to generate
        :return: List of dictionaries, each representing a random sample
        """
        if self.input_dim is None:
            raise ValueError("Input dimension is not set. Call set_search_space() to set the input dimension.")

        random_samples = []
        for _ in range(num_samples):
            sample = {}
            for var_info in self.search_space.config_space:
                var_name = var_info['name']
                var_domain = var_info['domain']
                # Generate a random floating-point number within the specified range
                random_value = np.random.uniform(var_domain[0], var_domain[1])
                sample[var_name] = random_value
            random_samples.append(sample)

        random_samples = self.inverse_transform(random_samples)
        return random_samples

    def model_reset(self):
        self.obj_model = None

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self.obj_model.X)

        return m.min()

    def posterior_samples(self, X, model_id, size=10):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        fsim = self.posterior_samples_f(X, model_id=model_id, size=size)

        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = self.samples(fsim[:, d])
        else:
            fsim = self.samples(fsim)
        return fsim