import numpy as np
import GPy

from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer
from typing import Dict, Union, List, Tuple
from transopt.optimizer.optimizer_base import BOBase
from optimizer.model.rgpe import RGPE

@optimizer_register("RGPE")
class RGPEOptimizer(BOBase):
    def __init__(self, config: Dict, **kwargs):
        super(RGPEOptimizer, self).__init__(config=config)
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

    def model_reset(self):
        if self.obj_model is None:
            self.obj_model = RGPE(n_features=self.input_dim)
        if self.obj_model.target_model is not None:
            self.meta_update()
        if self._X.size != 0:
            self.obj_model.fit({'X':self._X, 'Y':self._Y})


    def meta_update(self):
        self.obj_model.meta_update()
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

    def create_model(self):
        self.obj_model = RGPE(self.input_dim)

    def create_model(self, model_name, Source_data, Target_data):
        self.model_name = model_name
        source_num = len(Source_data['Y'])
        self.output_dim = source_num + 1

        ##Meta Date
        meta_data = {}
        for i in range(source_num):
            meta_data[i] = TaskData(X=Source_data['X'][i], Y=Source_data['Y'][i])

        ###Construct objective model
        if self.model_name == 'RGPE':
            self.obj_model = get_model('RGPE', self.model_space)
            self.obj_model.meta_fit(meta_data)
            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)
        elif self.model_name == 'SGPT_POE':
            self.obj_model = SGPT_POE(n_features=self.Xdim, beta=1)
            self.obj_model.meta_fit(meta_data)
            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)
        elif self.model_name == 'SGPT_M':
            self.obj_model = SGPT_M(n_features=self.Xdim)
            self.obj_model.meta_fit(meta_data)
            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)
        else:
            if self.kernel == None or self.kernel == 'RBF':
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            else:
                kern = GPy.kern.RBF(self.Xdim, ARD=True)
            X = Target_data['X']
            Y = Target_data['Y']

            self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1, verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

    def updateModel(self, Target_data):
        ## Train target model
        if self.model_name == 'RGPE' or \
            self.model_name == 'SGPT_POE' or self.model_name == 'SGPT_M':
            self.obj_model.fit(TaskData(Target_data['X'], Target_data['Y']), optimize=True)

        else:
            X = Target_data['X']
            Y = Target_data['Y']
            self.obj_model.set_XY(X, Y)
            try:
                self.obj_model.optimize_restarts(messages=True, num_restarts=1,
                                                 verbose=self.verbose)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

            return None

    def reset_target(self):
        self.obj_model.reset_target()

    def meta_add(self, meta_data):
        self.obj_model.meta_add(meta_data)

    def resetModel(self, Source_data, Target_data):
        ## Train target model
        pass

    def get_train_time(self):
        return self.fit_time

    def get_fit_time(self):
        return self.acf_time


    def predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        # returned mean: sum of means of the predictions of all source and target GPs
        mu, var = self.obj_model.predict(X, return_full=return_full)

        return mu, var


    def obj_posterior_samples(self, X, sample_size):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'SGPT_POE' or self.model_name == 'SGPT_M' or\
                self.model_name == 'RGPE':
            samples_obj = self.posterior_samples(X, model_id=0,size=sample_size)

        else:
            raise NameError

        return samples_obj

    def update_model(self, Data: Dict):
        ## Train target model
        if self.obj_model is None:
            self.create_model(Data['Target'])
        elif self.obj_model is not None:
            self.obj_model.set_XY(Data['Target'])
        else:
            self.obj_model.set_XY(Data['Target'])

        ## Train target model
        self.obj_model.fit(Data['Target'], optimize=True)

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self.obj_model.X)

        return m.min()

    def set_XY(self, X=None, Y=None):
        if isinstance(X, list):
            X, _, self.obj_model.output_index = util.multioutput.build_XY(X, None)
        if isinstance(Y, list):
            _, Y, self.obj_model.output_index = util.multioutput.build_XY(Y, Y)

        self.obj_model.update_model(False)
        if Y is not None:
            self.obj_model.Y = ObsAr(Y)
            self.obj_model.Y_normalized = self.obj_model.Y
        if X is not None:
            self.obj_model.X = ObsAr(X)

        self.obj_model.Y_metadata = {'output_index': self.obj_model.output_index, 'trials': np.ones(self.obj_model.output_index.shape)}
        if isinstance(self.obj_model.inference_method, expectation_propagation.EP):
            self.obj_model.inference_method.reset()
        self.obj_model.update_model(True)

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #orig_shape = gp.shape
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