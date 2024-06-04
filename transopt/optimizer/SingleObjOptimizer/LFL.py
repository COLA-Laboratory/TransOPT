import numpy as np
import GPy
from paramz import ObsAr
from optimizer.acquisition_function.get_acf import get_ACF
from transopt.optimizer.acquisition_function.sequential import Sequential
from typing import Dict, Union, List
from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Kernel import construct_multi_objective_kernel
from transopt.optimizer.model.MPGP import MPGP
from optimizer.model.GP_BAK import PriorGP
from transopt.utils import Prior

from GPy import util
from GPy.inference.latent_function_inference import expectation_propagation
from GPy.inference.latent_function_inference import ExactGaussianInference
from GPy.likelihoods.multioutput_likelihood import MixedNoise

from transopt.utils.Normalization import get_normalizer

@optimizer_register('LFL')
class LFLOptimizer(BOBase):
    def __init__(self, config:Dict, **kwargs):
        super(LFLOptimizer, self).__init__(config=config)

        self.init_method = 'LFL'
        self.knowledge_num = 2
        self.ini_quantile = 0.5
        self.anchor_points = None
        self.anchor_num = None
        self.model = None
        self.output_dim = None

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        if 'init_number' in config:
            self.ini_num = config['init_number']
        else:
            self.ini_num = None

        if 'acf' in config:
            self.acf = config['acf']
        else:
            self.acf = 'EI'


    def reset(self, design_space:Dict, search_sapce:Union[None, Dict] = None):
        self.set_space(design_space, search_sapce)
        self.obj_model = None
        self.var_model = None
        self.output_dim = None
        self.acqusition = get_ACF(self.acf, model=self, search_space=self.search_space, config=self.config)
        self.evaluator = Sequential(self.acqusition)


    def initial_sample(self):
        if self.anchor_points is None:
            self.anchor_num = int(self.ini_quantile * self.ini_num)
            self.anchor_points  = self.random_sample(self.anchor_num)

        random_samples = self.random_sample(self.ini_num - self.anchor_num)
        samples = self.anchor_points.copy()
        samples.extend(random_samples)

        return samples

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



    def combine_data(self):
        if len(self.aux_data) == 0:
            return {'Target':{'X':self._X, 'Y':self._Y}}
        else:
            return {}

    def suggest(self, n_suggestions:Union[None, int] = None)->List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if 'normalize' in self.config:
                self.normalizer = get_normalizer(self.config['normalize'])

            if self.aux_data is not None:
                pass
            else:
                self.aux_data = {}

            Data = self.combine_data()
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(self._get_var_name('search'),suggested_sample)
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def create_model(self, X_list, Y_list, mf=None, prior:list=[]):
        X, Y, output_index = util.multioutput.build_XY(X_list, Y_list)

        if self.output_dim > 1:
            K = construct_multi_objective_kernel(self.input_dim, self.output_dim, base_kernel='RBF', Q=1, rank=2)
            inference_method = ExactGaussianInference()
            likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_obj_%s" % j) for y, j in
                                zip(Y, range(self.output_dim))]
            likelihood = MixedNoise(likelihoods_list=likelihoods_list)

            self.obj_model = MPGP(X, Y, K, likelihood, Y_metadata={'output_index': output_index},
                                  inference_method=inference_method, mean_function=mf, name=f'OBJ MPGP')

            self.obj_model['mixed_noise.Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            # self.obj_model['constmap.C'].constrain_fixed(0)
            self.obj_model['ICM0.B.kappa'].constrain_fixed(np.zeros(shape=(self.output_dim,)))

        else:
            if 'kernel' in self.config:
                kern = GPy.kern.RBF(self.input_dim, ARD=False)
            else:
                kern = GPy.kern.RBF(self.input_dim, ARD=False)
            X = X_list[0]
            Y = Y_list[0]

            self.obj_model = PriorGP(X, Y, kernel=kern, mean_function = mf)
            self.obj_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)


        if len(prior) == 0:
            self.prior_list = []
            self.prior_list.append(Prior.LogGaussian(1, 2, 'lengthscale'))
            self.prior_list.append(Prior.LogGaussian(0.5, 2, 'variance'))
        else:
            self.prior_list = prior

        for i in range(len(self.prior_list)):
            self.obj_model.set_prior(self.prior_list[i])

    def update_model(self, Data):
        ## Train target model
        assert 'Target' in Data
        target_data = Data['Target']
        X_list = []
        Y_list = []

        if 'History' in Data:
            history_data = Data['History']
            X_list.extend(list(history_data['X']))
            Y_list.extend(list(history_data['Y']))
            source_num = len(history_data['Y'])
        else:
            source_num = 0
            history_data = {}

        if 'Gym' in Data:
            Gym_data = Data['Gym']
            gym_num = len(Gym_data['Gym'])
            X_list.extend(list(Gym_data['X']))
            Y_list.extend(list(Gym_data['Y']))
        else:
            gym_num = 0
            Gym_data = {}

        output_dim = gym_num + source_num + 1

        X_list.append(target_data['X'])
        Y_list.append(target_data['Y'])

        if self.normalizer is not None:
            Y_list = self.normalizer(Y_list)

        if self.output_dim != output_dim:
            self.output_dim = output_dim
            self.create_model(X_list, Y_list, prior=[])
        else:
            self.set_XY(X_list, Y_list)
            if self.var_model is not None:
                self.var_model.set_XY(target_data['X'][0], target_data['Y'][0])

        try:
            self.obj_model.optimize_restarts(messages=False, num_restarts=1,
                                             verbose=self.verbose)
            if self.var_model is not None:
                self.var_model.optimize_restarts(messages=False, num_restarts=1,
                                                verbose=self.verbose)
        except np.linalg.linalg.LinAlgError as e:
            # break
            print('Error: np.linalg.linalg.LinAlgError')


    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.output_dim >1:
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            m, v = self.obj_model.predict(X, Y_metadata=noise_dict, full_cov=False, include_likelihood=True)
            v = np.clip(v, 1e-10, np.inf)

        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v

    def var_predict(self, X):
        if X.ndim == 1:
            X = X[None,:]
        task_id = self.output_dim - 1

        if self.model_name == 'MOGP':
            noise_dict  = {'output_index': np.array([task_id] * X.shape[0])[:,np.newaxis].astype(int)}
            X = np.hstack((X, noise_dict['output_index']))

            _, v1 = self.var_model.predict(X)
            v1 = np.clip(v1, 1e-10, np.inf)
            v = v1
        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return v

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

    def get_model_para(self):

        if self.model_name == 'MOGP':
            lengthscale = self.obj_model['.*lengthscale'][0]
            variance = self.obj_model['.*rbf.*variance'][0]
        else:
            lengthscale = self.obj_model['rbf.*lengthscale'][0]
            variance = self.obj_model['rbf.*variance'][0]

        return lengthscale, variance

    def update_prior(self, parameters):
        for k, v in parameters.items():
            prior = self.obj_model.get_prior(k)
            cur_stat = prior.getstate()
            # mu = (self.kappa * cur_stat[0] + v) / (self.kappa + 1)
            # var = cur_stat[1] + (self.kappa * (v - cur_stat[0]) ** 2) / (2.0 * (self.kappa + 1.0))
            mu = np.mean(parameters[k])
            var = np.var(parameters[k])

            self.obj_model.update_prior(k, [mu, var])