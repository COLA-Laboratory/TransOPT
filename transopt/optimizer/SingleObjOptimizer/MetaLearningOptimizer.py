import numpy as np
import GPy
import GPyOpt
from GPy import util
from paramz import ObsAr

from GPy.inference.latent_function_inference import expectation_propagation
from transopt.optimizer.optimizer_base import OptimizerBase
# from Model.HyperBO import hyperbo

from external.transfergpbo import models

from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace

from external.FSBO.fsbo_modules import FSBO, DeepKernelGP
from external.FSBO.fsbo_utils import totorch
import os

from external.transfergpbo.models import (
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)


def get_model(
    model_name: str, space: ParameterSpace
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = GPy.kern.RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)

    return model

class MetaBOOptimizer(OptimizerBase):
    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, Xdim, bounds, kernel='RBF', likelihood=None, model_name='MOGP', acf_name='EI',
                 optimizer='bfgs',  verbose=True, seed = 0):
        self.kernel = kernel
        self.likelihood = likelihood
        self.Xdim = Xdim
        self.bounds = bounds
        self.acf_name = acf_name
        self.Seed = seed
        self.name = 'meta'

        # Set decision space
        Variables = []
        task_design_space = []
        for var in range(Xdim):
            v_n = f'x{var + 1}'
            Variables.append(ContinuousParameter(v_n, self.bounds[0][var], self.bounds[1][var]))
            var_dic = {'name': f'var_{var}', 'type': 'continuous',
                       'domain': tuple([self.bounds[0][var], self.bounds[1][var]])}
            task_design_space.append(var_dic.copy())
        self.model_space = ParameterSpace(Variables)
        self.acf_space = GPyOpt.Design_space(space=task_design_space)

        self.optimizer = optimizer
        self.verbose = verbose


    def create_model(self, model_name, Meta_data, Target_data):
        self.model_name = model_name
        source_num = len(Meta_data['Y'])
        self.output_dim = source_num + 1

        ###Construct objective model
        if self.model_name == 'HyperBO':
            self.obj_model = hyperbo()
            self.obj_model.pretrain(Meta_data, Target_data)

        elif self.model_name == 'FSBO':
            checkpoint_path = './External/FSBO/checkpoints/'
            self.training_model = FSBO(input_size=self.Xdim, checkpoint_path = checkpoint_path, batch_size=len(Meta_data['X'][0]))
            train_data = {}
            for i in range(source_num):
                train_data[i] = {'X':Meta_data['X'][i], 'y':Meta_data['Y'][i]}
            self.training_model.set_data(train_data=train_data)
            self.training_model.meta_train(epochs=1000)
            log_dir = os.path.join(checkpoint_path, "log.txt"),
            self.obj_model = DeepKernelGP(epochs = 1000, input_size=self.Xdim, checkpoint = checkpoint_path + f'Seed_{self.Seed}_{source_num+1}', log_dir= log_dir, seed=self.Seed)
            self.device = 'cpu'
            self.obj_model.X_obs, self.obj_model.y_obs = totorch(Target_data['X'], self.device), totorch(Target_data['Y'], self.device).reshape(-1)
            self.obj_model.train()

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
        ###Construct objective model
        if self.model_name == 'HyperBO':
            self.obj_model.retrain(Target_data)
        elif self.model_name == 'FSBO':
            self.obj_model.X_obs, self.obj_model.y_obs = totorch(Target_data['X'], self.device), totorch(
                Target_data['Y'], self.device).reshape(-1)
            self.obj_model.train()
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

    def resetModel(self, Source_data, Target_data):
        ## Train target model
        pass



    def predict(self, X):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """

        if self.model_name == 'HyperBO':
            m, v = self.obj_model.predict(X)
            m = np.array(m)
            v = np.array(v)

        elif self.model_name == 'FSBO':
            X = totorch(X, self.device)
            m,v = self.obj_model.predict(X)
            m = m[:,np.newaxis]
            v = v[:,np.newaxis]
        else:
            m, v = self.obj_model.predict(X)

        # We can take the square root because v is just a diagonal matrix of variances
        return m, v


    #
    # def obj_posterior_samples(self, X, sample_size):
    #     if X.ndim == 1:
    #         X = X[None,:]
    #     task_id = self.output_dim - 1
    #
    #     if self.model_name == 'WSGP' or \
    #             self.model_name == 'HGP':
    #         samples_obj = self.posterior_samples(X, model_id=0,size=sample_size)
    #     elif self.model_name == 'MOGP':
    #         noise_dict = {'output_index': np.array([task_id] * X.shape[0])[:, np.newaxis].astype(int)}
    #         X_zip = np.hstack((X, noise_dict['output_index']))
    #
    #         samples_obj = self.obj_model.posterior_samples(X_zip, size=sample_size, Y_metadata=noise_dict) # grid * 1 * sample_num
    #
    #     else:
    #         raise NameError
    #
    #     return samples_obj

    def get_fmin(self):
        "Get the minimum of the current model."
        if self.model_name == 'HyperBO':
            m = np.array(self.obj_model._Y)
            return np.min(m)
        elif self.model_name == 'FSBO':
            m = self.obj_model.y_obs.detach().to("cpu").numpy().reshape(-1,)
            return np.min(m)
        else:
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