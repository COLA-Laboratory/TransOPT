import GPy, GPyOpt
import numpy as np
from typing import Dict, Union, List

from transopt.optimizer.optimizer_base import BOBase
from transopt.utils.serialization import ndarray_to_vectors
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer
from transopt.utils.weights import init_weight, tchebycheff

# from utils.common import findKBest
# from revision.multiobjective_bayesian_optimization import MultiObjectiveBayesianOptimization
# from revision.weighted_gpmodel import WeightedGPModel
# from revision.multiobjective_EI import MultiObjectiveAcquisitionEI

@optimizer_register("MoeadEGO")
class MoeadEGO(BOBase):
    def __init__(self, config: Dict, **kwargs):
        super(MoeadEGO, self).__init__(config=config)

        self.init_method = "Random"
        self.verbose = config.get("verbose", True)
        self.n_weight = config.get("n_weight", 10)
        self.pop_size = config.get("pop_size", self.n_weight)

        if self.pop_size > self.n_weight:
            self.pop_size = self.n_weight

        self.ini_num = self.pop_size
        
        self.model = []
        self.acf = "MOEADEGO"
        self.weight = None

    def initial_sample(self):
        return self.random_sample(self.ini_num)

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


    def update_model(self, Data):
        Target_Data = Data["Target"]
        assert "X" in Target_Data

        X = Target_Data["X"]
        Y = Target_Data["Y"]
        assert Y.shape[0] == self.num_objective

        if self.normalizer is not None:
            Y_norm = np.array([self.normalizer(y) for y in Y])

        if len(self.model_list) == 0:
            self.create_model(X, Y_norm)
        else:
            ideal_point = np.min(Y_norm.T, axis=0)
            for i in range(len(self.model_list)):
                Y_weighted = tchebycheff(Y.T, self.weight[i], ideal=ideal_point)
                self.model_list[i].set_XY(X, Y_weighted)

        try:
            for i in range(len(self.model_list)):
                self.model_list[i].optimize_restarts(
                    num_restarts=1, verbose=self.verbose, robust=True
                )
        except np.linalg.linalg.LinAlgError as e:
            # break
            print("Error: np.linalg.linalg.LinAlgError")

    def create_model(self, X, Y):
        assert self.num_objective is not None

        ideal_point = np.min(Y.T, axis=0)
        self.weight = init_weight(self.num_objective, self.n_weight)
        self.n_weight = self.weight.shape[0]

        for i in range(self.n_weight):
            kernel = GPy.kern.RBF(input_dim = self.input_dim)

            Y_weighted = tchebycheff(Y.T, self.weight[i], ideal=ideal_point)

            model = GPy.models.GPRegression(X, Y_weighted, kernel=kernel, normalizer=None)
            model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
            model['.*rbf.variance'].constrain_fixed(1.0)
            self.kernel_list.append(model.kern)
            self.model_list.append(model)

    def suggest(self, n_suggestions: Union[None, int] = None) -> List[Dict]:
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

    def predict_by_id(self, X, idx, full_cov=False):
        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))
            mean, var = self.model_list[idx].predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis=0)
            else:
                pred_var = np.append(pred_var, var, axis=1)
        return pred_mean, pred_var

    def model_reset(self):
        self.model_list = []
        self.kernel_list = []

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self._X)

        return m.min()

    def get_fmin_by_id(self, idx):
        "Get the minimum of the current model."
        m, v = self.predict_by_id(self._X, idx)

        return m.min()
