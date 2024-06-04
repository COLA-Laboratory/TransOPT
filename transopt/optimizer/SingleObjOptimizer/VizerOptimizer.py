import numpy as np

from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer
from transopt.optimizer.model.MHGP import MHGP
from typing import Dict, Union, List, Tuple
from transopt.optimizer.optimizer_base import BOBase


@optimizer_register('vizer')
class Vizer(BOBase):

    def __init__(self, config: Dict, **kwargs):
        super(Vizer, self).__init__(config=config)
        self.init_method = 'Random'

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


    def model_reset(self):
        if self.obj_model is None:
            self.obj_model = MHGP(n_features=self.input_dim)
        if self.obj_model.target_gp is not None:
            self.meta_update()
            self.obj_model.fit({'X':self._X, 'Y':self._Y})
        if self.obj_model.target_gp is None and self._X.size != 0:
            self.obj_model.fit({'X':self._X, 'Y':self._Y})

    def initial_sample(self):
        return self.random_sample(self.ini_num)

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

    def suggest(self, n_suggestions:Union[None, int] = None) ->List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if 'normalize' in self.config:
                self.normalizer = get_normalizer(self.config['normalize'])

            Data = {}
            Data['Target'] = {'X':self._X, 'Y':self._Y}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'), suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def meta_update(self):
        self.obj_model.meta_update()

    def meta_add(self, Data:List[Dict]):
        self.obj_model.meta_add(Data)

    def create_model(self):
        self.obj_model = MHGP(self.input_dim)


    def update_model(self, Data):
        ## Train target model
        if self.obj_model is None:
            self.create_model(Data['Target'])
        elif self.obj_model is not None:
            self.obj_model.set_XY(Data['Target'])
        else:
            self.obj_model.set_XY(Data['Target'])

        ## Train target model
        self.obj_model.fit(Data['Target'], optimize=True)


    def MetaFitModel(self, metadata):

        if self.model_name == 'SHGP' or \
            self.model_name == 'MHGP' or \
            self.model_name == 'BHGP':

            self.obj_model.meta_fit(metadata)

    # def meta_add(self, meta_data:Dict):
    #     self.obj_model.meta_add(meta_data:Dict)


    def get_train_time(self):
        return self.fit_time

    def get_fit_time(self):
        return self.acf_time


    def predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        # returned mean: sum of means of the predictions of all source and target GPs
        mu, var = self.obj_model.predict(X, return_full=return_full, with_noise=with_noise)

        return mu, var


    def obj_posterior_samples(self, X, sample_size):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'SHGP' or \
                self.model_name == 'HGP' or \
                self.model_name == 'MHGP' or \
                self.model_name == 'BHGP' or \
                self.model_name == 'RPGE':
            samples_obj = self.posterior_samples(X, model_id=0,size=sample_size)
        elif self.model_name == 'MOGP':
            noise_dict = {'output_index': np.array([task_id] * X.shape[0])[:, np.newaxis].astype(int)}
            X_zip = np.hstack((X, noise_dict['output_index']))

            samples_obj = self.obj_model.posterior_samples(X_zip, size=sample_size, Y_metadata=noise_dict) # grid * 1 * sample_num

        else:
            raise NameError

        return samples_obj

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self.obj_model.X)

        return m.min()

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        # orig_shape = gp.shape
        gp = gp.flatten()
        Ysim = np.array([np.random.normal(gpj, scale=np.sqrt(1e-2), size=1) for gpj in gp])
        return Ysim.reshape(orig_shape)


    def posterior_samples_f(self,X, model_id, size=10):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """
        m, v = self.obj_model.predict(X, return_full=True)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


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