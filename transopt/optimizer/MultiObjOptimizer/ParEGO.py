import numpy as np
import GPy
from typing import Dict, Union, List

from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer


@optimizer_register("ParEGO")
class ParEGO(BOBase):
    def __init__(self, config: Dict, **kwargs):
        super(ParEGO, self).__init__(config=config)

        self.init_method = "Random"

        if "verbose" in config:
            self.verbose = config["verbose"]
        else:
            self.verbose = True

        if "init_number" in config:
            self.ini_num = config["init_number"]
        else:
            self.ini_num = None

        self.acf = "EI"
        self.rho = 0.1

    def scalarization(self, Y: np.ndarray, rho):
        """
        scalarize observed output data
        """
        theta = np.random.random_sample(Y.shape[0])
        sum_theta = np.sum(theta)
        theta = theta / sum_theta

        theta_f = Y.T * theta
        max_k = np.max(theta_f, axis=1)
        rho_sum_theta_f = rho * np.sum(theta_f, axis=1)

        return max_k + rho_sum_theta_f

    def initial_sample(self):
        return self.random_sample(self.ini_num)

    def suggest(self, n_suggestions: Union[None, int] = None) -> List[Dict]:
        return self.random_sample(self.ini_num)
        
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if "normalize" in self.config:
                self.normalizer = get_normalizer(self.config["normalize"])

            Data = {"Target": {"X": self._X, "Y": self._Y}}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(
                None, context_manager=None
            )
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(
                self._get_var_name("search"), suggested_sample
            )
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def update_model(self, Data):
        Target_Data = Data["Target"]
        assert "X" in Target_Data

        X = Target_Data["X"]
        Y = Target_Data["Y"]
        assert Y.shape[0] == self.num_objective

        if self.normalizer is not None:
            Y_norm = np.array([self.normalizer(y) for y in Y])

        Y_scalar = self.scalarization(Y_norm, 0.1)[:, np.newaxis]

        if len(self.model_list) == 0:
            self.create_model(X, Y_scalar)
        else:
            self.model_list[0].set_XY(X, Y_scalar)

        try:
            self.model_list[0].optimize_restarts(
                num_restarts=1, verbose=self.verbose, robust=True
            )
        except np.linalg.linalg.LinAlgError as e:
            # break
            print("Error: np.linalg.linalg.LinAlgError")

    def create_model(self, X, Y):
        assert self.num_objective is not None

        kernel = GPy.kern.RBF(input_dim=self.input_dim)
        model = GPy.models.GPRegression(X, Y, kernel=kernel, normalizer=None)
        model[".*Gaussian_noise.variance"].constrain_fixed(1.0e-4)
        model[".*rbf.variance"].constrain_fixed(1.0)
        self.kernel_list.append(model.kern)
        self.model_list.append(model)
        print("model state")
        for i, model in enumerate(self.model_list):
            print("--------model for {}th object--------".format(i))
            print(model)

    def predict(self, X, full_cov=False):
        # X_copy = np.array([X])
        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))
        for model in self.model_list:
            mean, var = model.predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        return pred_mean, pred_var

    def random_sample(self, num_samples: int) -> List[Dict]:
        """
        Initialize random samples.

        :param num_samples: Number of random samples to generate
        :return: List of dictionaries, each representing a random sample
        """
        if self.input_dim is None:
            raise ValueError(
                "Input dimension is not set. Call set_search_space() to set the input dimension."
            )

        random_samples = []
        for _ in range(num_samples):
            sample = {}
            for var_info in self.search_space.config_space:
                var_name = var_info["name"]
                var_domain = var_info["domain"]
                # Generate a random floating-point number within the specified range
                random_value = np.random.uniform(var_domain[0], var_domain[1])
                sample[var_name] = random_value
            random_samples.append(sample)

        random_samples = self.inverse_transform(random_samples)
        return random_samples

    def model_reset(self):
        self.model_list = []
        self.kernel_list = []

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self._X)

        return m.min()
