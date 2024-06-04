import logging
import numpy as np
import ConfigSpace as CS
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from typing import Union, Tuple, Dict, List

from transopt.benchmark.problem_base import NonTabularProblem
from agent.registry import benchmark_register


logger = logging.getLogger("MovingPeakBenchmark")


class MovingPeakGenerator:
    def __init__(
        self,
        n_var,
        shift_length=3.0,
        height_severity=7.0,
        width_severity=1.0,
        lam=0.5,
        n_peak=4,
        n_step=11,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.n_var = n_var
        self.shift_length = shift_length
        self.height_severity = height_severity
        self.width_severity = width_severity

        # lambda determines whether there is a direction of the movement, or whether they are totally random.
        # For lambda = 1.0 each move has the same direction, while for lambda = 0.0, each move has a random direction
        self.lam = lam

        # number of peaks in the landscape
        self.n_peak = n_peak

        self.var_bound = np.array([[0, 100]] * n_var)

        self.height_bound = np.array([[30, 70]] * n_peak)

        self.width_bound = np.array([[1.0, 12.0]] * n_peak)

        self.n_step = n_step

        self.t = 0

        self.bounds = np.array(
            [[-1.0] * self.n_var, [1.0] * self.n_var], dtype=np.float64
        )

        current_peak = np.random.random(size=(n_peak, n_var)) * np.tile(
            self.var_bound[:, 1] - self.var_bound[:, 0], (n_peak, 1)
        ) + np.tile(self.var_bound[:, 0], (n_peak, 1))

        current_width = (
            np.random.random(size=(n_peak,))
            * (self.width_bound[:, 1] - self.width_bound[:, 0])
            + self.width_bound[:, 0]
        )

        current_height = (
            np.random.random(size=(n_peak,))
            * (self.height_bound[:, 1] - self.height_bound[:, 0])
            + self.height_bound[:, 0]
        )

        previous_shift = normalize(
            np.random.random(size=(n_peak, n_var)), axis=1, norm="l2"
        )

        self.peaks = []
        self.widths = []
        self.heights = []

        self.peaks.append(current_peak)
        self.widths.append(current_width)
        self.heights.append(current_height)

        for t in range(1, n_step):
            peak_shift = self.cal_peak_shift(previous_shift)
            width_shift = self.cal_width_shift()
            height_shift = self.cal_height_shift()
            current_peak = current_peak + peak_shift
            current_height = current_height + height_shift.squeeze()
            current_width = current_width + width_shift.squeeze()
            for i in range(self.n_peak):
                self._fix_bound(current_peak[i, :], self.var_bound)
            self._fix_bound(current_width, self.width_bound)
            self._fix_bound(current_height, self.height_bound)
            previous_shift = peak_shift
            self.peaks.append(current_peak)
            self.widths.append(current_width)
            self.heights.append(current_height)

    def get_MPB(self):
        return self.peaks, self.widths, self.heights

    def cal_width_shift(self):
        width_change = np.random.random(size=(self.n_peak, 1))
        return self.width_severity * width_change

    def cal_height_shift(self):
        height_change = np.random.random(size=(self.n_peak, 1))
        return self.height_severity * height_change

    def cal_peak_shift(self, previous_shift):
        peak_change = np.random.random(size=(self.n_peak, self.n_var))
        return (1 - self.lam) * self.shift_length * normalize(
            peak_change - 0.5, axis=1, norm="l2"
        ) + self.lam * previous_shift

    def change(self):
        if self.t < self.n_step - 1:
            self.t += 1

    def current_optimal(self, peak_shape=None):
        current_peak = self.peaks[self.t]
        current_height = self.heights[self.t]
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_y = self.f(optimal_x, peak_shape)
        return optimal_x, optimal_y

    def transfer(self, X):
        return (X + 1) * (self.var_bound[:, 1] - self.var_bound[:, 0]) / 2 + (
            self.var_bound[:, 0]
        )

    def normalize(self, X):
        return (
            2
            * (X - (self.var_bound[:, 0]))
            / (self.var_bound[:, 1] - self.var_bound[:, 0])
            - 1
        )

    @property
    def optimizers(self):
        current_peak = self.peaks[self.t]
        current_height = self.heights[self.t]
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_x = self.normalize(optimal_x)
        return optimal_x

    @staticmethod
    def _fix_bound(data, bound):
        for i in range(data.shape[0]):
            if data[i] < bound[i, 0]:
                data[i] = 2 * bound[i, 0] - data[i]
            elif data[i] > bound[i, 1]:
                data[i] = 2 * bound[i, 1] - data[i]
            while data[i] < bound[i, 0] or data[i] > bound[i, 1]:
                data[i] = data[i] * 0.5 + bound[i, 0] * 0.25 + bound[i, 1] * 0.25


@benchmark_register("MPB")
class MovingPeakBenchmark(NonTabularProblem):
    def __init__(
        self,
        task_name,
        budget,
        peak,
        height,
        width,
        seed,
        input_dim,
        task_type="non-tabular",
    ):
        self.dimension = input_dim
        self.peak = peak
        self.height = height
        self.width = width
        self.n_peak = len(peak)
        super(MovingPeakBenchmark, self).__init__(
            task_name=task_name, seed=seed, task_type=task_type, budget=budget
        )

    def peak_function_cone(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(self.height - self.width * distance)

    def peak_function_sharp(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(self.height / (1 + self.width * distance * distance))

    def peak_function_hilly(self, x):
        distance = np.linalg.norm(np.tile(x, (self.n_peak, 1)) - self.peak, axis=1)
        return np.max(
            self.height
            - self.width * distance * distance
            - 0.01 * np.sin(20.0 * distance * distance)
        )

    def objective_function(
        self,
        configuration: Union[CS.Configuration, Dict],
        fidelity: Union[Dict, CS.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        if "peak_shape" not in kwargs:
            peak_shape = "cone"
        else:
            peak_shape = kwargs["peak_shape"]

        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])
        if peak_shape == "cone":
            peak_function = self.peak_function_cone
        elif peak_shape == "sharp":
            peak_function = self.peak_function_sharp
        elif peak_shape == "hilly":
            peak_function = self.peak_function_hilly
        else:
            # print("Unknown shape, set to default")
            peak_function = self.peak_function_cone
        y = peak_function(X)

        return {"function_value": float(y), "info": {"fidelity": fidelity}}

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                CS.UniformFloatHyperparameter(f"x{i}", lower=0.0, upper=100.0)
                for i in range(self.dimension)
            ]
        )

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}
