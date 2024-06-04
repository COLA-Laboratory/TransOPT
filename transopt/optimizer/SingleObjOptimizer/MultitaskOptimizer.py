import numpy as np
import GPy
from typing import Dict, Union, List
from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from paramz import ObsAr
from transopt.utils.Normalization import get_normalizer
from GPy import util
from transopt.utils.Kernel import construct_multi_objective_kernel
from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise
from transopt.optimizer.model.MPGP import MPGP

@optimizer_register('MTBO')
class MultitaskBO(BOBase):
    def __init__(self, config:Dict, **kwargs):
        super(MultitaskBO, self).__init__(config=config)
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


            if len(self.aux_data):
                Data = self.aux_data
            else:
                Data = {}
            Data['Target'] = {'X':self._X, 'Y':self._Y}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'), suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def update_model(self, Data):
        assert 'Target' in Data
        X_list = []
        Y_list = []

        if 'History' in Data:
            history_data = Data['History']
            X_list.extend(list(history_data['X']))
            Y_list.extend(list(history_data['Y']))


        target_data = Data['Target']
        X_list.append(target_data['X'])
        Y_list.append(target_data['Y'])

        if self.normalizer is not None:
            Y_list = self.normalizer(Y_list)

        self.output_dim = len(Y_list)
        self.task_id = self.output_dim - 1

        if self.obj_model == None:
            self.create_model(X_list, Y_list)
        else:
            if self.output_dim > 1:
                self.set_XY(X_list, Y_list)
            else:
                self.obj_model.set_XY(X_list[0], Y_list[0])

        try:
            self.obj_model.optimize_restarts(num_restarts=1, verbose=self.verbose, robust=True)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')

    def create_model(self, X_list, Y_list, mf=None, prior:list=[]):
        if self.output_dim > 1:
            X, Y, output_index = util.multioutput.build_XY(X_list, Y_list)

            #Set inference Method
            inference_method = ExactGaussianInference()
            ## Set likelihood
            likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                                zip(Y, range(self.output_dim))]
            likelihood = MixedNoise(likelihoods_list=likelihoods_list)

            kernel = construct_multi_objective_kernel(self.input_dim, output_dim=self.output_dim, base_kernel='RBF', rank=self.output_dim)
            self.obj_model = MPGP(X, Y, kernel, likelihood, Y_metadata={'output_index': output_index}, inference_method=inference_method, name=f'OBJ MPGP')

        else:
            if 'kernel' in self.config:
                kern = GPy.kern.RBF(self.input_dim, ARD=False)
            else:
                kern = GPy.kern.RBF(self.input_dim, ARD=False)
            X = X_list[0]
            Y = Y_list[0]

            self.obj_model = GPy.models.GPRegression(X, Y, kernel=kern)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)


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

    def model_reset(self):
        self.obj_model = None

    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]

        if self.output_dim > 1:
            noise_dict  = {'output_index': np.array([self.task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self.obj_model.X)

        return m.min()