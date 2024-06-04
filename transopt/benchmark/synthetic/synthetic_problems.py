# %matplotlib notebook

import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict
from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace


logger = logging.getLogger("SyntheticBenchmark")

class SyntheticProblemBase(NonTabularProblem):
    problem_type = "synthetic"
    num_variables = []
    num_objectives = 1
    workloads = []
    fidelity = None
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        super(SyntheticProblemBase, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def get_fidelity_space(self) -> FidelitySpace:
        fs = FidelitySpace([])
        return fs

    def get_objectives(self) -> Dict:
        return {'f1':'minimize'}

    def get_problem_type(self):
        return "synthetic"
    


@problem_registry.register("Sphere")
class SphereOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        super(SphereOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.sum((X) ** 2, axis=1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.12, 5.12)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Rastrigin")
class RastriginOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 2.0)
        self.dtype = np.float64

        super(RastriginOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.4)

        n = X.shape[0]
        d = X.shape[1]

        pi = np.array([math.pi], dtype=self.dtype)
        y = 10.0 * self.input_dim + np.sum((X) ** 2 - 10.0 * np.cos(pi * (X)), axis=1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.12, 5.12)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Schwefel")
class SchwefelOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(420.9687 - self.shift)
        self.dtype = np.float64

        super(SchwefelOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 420 - np.sum(
            np.multiply(X, np.sin(np.sqrt(abs(self.stretch * X - self.shift)))), axis=1
        )
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-500.0, 500.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss



@problem_registry.register("LevyR")
class LevyROptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 1.0)
        self.dtype = np.float64

        super(LevyROptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.1)

        n = X.shape[0]
        d = X.shape[1]

        w = 1.0 + X / 4.0
        pi = np.array([math.pi], dtype=self.dtype)
        part1 = np.sin(pi * w[..., 0]) ** 2
        part2 = np.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 5.0 * np.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            axis=1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (1.0 + np.sin(2 * math.pi * w[..., -1]) ** 2)
        y = part1 + part2 + part3
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-10.0, 10.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Griewank")
class GriewankOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        super(GriewankOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        div = np.arange(start=1, stop=d + 1, dtype=self.dtype)
        part1 = np.sum(X**2 / 4000.0, axis=1)
        part2 = -np.prod(np.cos(X / np.sqrt(div)), axis=1)
        y = part1 + part2 + 1.0
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-100.0, 100.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Rosenbrock")
class RosenbrockOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        super(RosenbrockOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            axis=-1,
        )
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 10.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("DropwaveR")
class DropwaveROptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 3.3)
        self.dtype = np.float64

        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([2 * math.pi], dtype=self.dtype)

        super(DropwaveROptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.33)

        n = X.shape[0]
        d = X.shape[1]

        part1 = np.linalg.norm(X, axis=1)
        y = -(3 + np.cos(part1)) / (0.1 * np.power(part1, 1.5) + 1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-10.0, 10.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Langermann")
class LangermannOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        self.c = np.array([1, 2, 5])
        self.m = 3
        self.A = np.random.randint(1, 10, (self.m, self.input_dim))

        super(LangermannOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 0
        for i in range(self.m):
            part1 = np.exp(-np.sum(np.power(X - self.A[i], 2), axis=1) / np.pi)
            part2 = np.cos(np.sum(np.power(X - self.A[i], 2), axis=1) * np.pi)
            y += part1 * part2 * self.c[i]
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (0.0, 10.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("RotatedHyperEllipsoid")
class RotatedHyperEllipsoidOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 32.75)
        self.dtype = np.float64

        super(RotatedHyperEllipsoidOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift + 0.5)

        n = X.shape[0]
        d = X.shape[1]

        div = np.arange(start=d, stop=0, step=-1, dtype=self.dtype)
        y = np.sum(div * X**2, axis=1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-65.536, 65.536)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("SumOfDifferentPowers")
class SumOfDifferentPowersOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift + 0.238)
        self.dtype = np.float64

        super(SumOfDifferentPowersOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.238)

        n = X.shape[0]
        d = X.shape[1]

        y = np.zeros(shape=(n,), dtype=self.dtype)
        for i in range(d):
            y += np.abs(X[:, i]) ** (i + 1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-1.0, 1.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("StyblinskiTang")
class StyblinskiTangOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 2.903534)
        self.dtype = np.float64

        super(StyblinskiTangOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = 0.5 * (X**4 - 16 * X**2 + 5 * X).sum(axis=1)
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Powell")
class PowellOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [tuple(0.0 for _ in range(self.input_dim))]
        self.dtype = np.float64

        super(PowellOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.zeros_like(X[..., 0])
        for i in range(self.input_dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            y += part1 + part2 + part3 + part4
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-4.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("DixonPrice")
class DixonPriceOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.input_dim + 1)
            )
        ]
        self.dtype = np.float64

        super(DixonPriceOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        part1 = (X[..., 0] - 1) ** 2
        i = np.arange(start=2, stop=d + 1, step=1)
        i = np.tile(i, (n, 1))
        part2 = np.sum(i * (2.0 * X[..., 1:] ** 2 - X[..., :-1]) ** 2, axis=1)
        y = part1 + part2
        # y +=  self.noise(n)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-10.0, 10.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("cp")
class cpOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.input_dim + 1)
            )
        ]
        self.dtype = np.float64

        super(cpOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array(
            [[configuration[k] for idx, k in enumerate(configuration.keys())]]
        )[0]

        part1 = np.sin(6 * X[0]) + X[1] ** 2
        part2 = 0.1 * X[0] ** 2 + 0.1 * X[1] ** 2

        if self.task_id == 1:
            part3 = 0.1 * ((3) * (X[0] + 0.3)) ** 2 + 0.1 * ((3) * (X[1] + 0.3)) ** 2
        else:
            part3 = 0.1 * ((3) * (X[0] - 0.3)) ** 2 + 0.1 * ((3) * (X[1] - 0.3)) ** 2

        y = part1 + part3 + part2

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-1.0, 1.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("mpb")
class mpbOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.input_dim + 1)
            )
        ]
        self.dtype = np.float64

        super(mpbOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array(
            [[configuration[k] for idx, k in enumerate(configuration.keys())]]
        )[0]

        n_peak = 2
        self.peak = np.ndarray([[-0.5, -0.5], [0.2, 0.2], []])

        if self.task_id == 0:
            distance = np.linalg.norm(np.tile(X, (n_peak, 1)) - self.peak[0], axis=1)
        elif self.task_id == 1:
            distance = np.linalg.norm(np.tile(X, (n_peak, 1)) - self.peak[0], axis=1)
        else:
            distance = np.linalg.norm(np.tile(X, (n_peak, 1)) - self.peak[0], axis=1)

        y = np.max(self.height - self.width * distance)

        return {"f1": float(y), "info": {"fidelity": fidelity}}

def get_configuration_space(self) -> SearchSpace:
        
        variables =  [Continuous(f'x{i}', (-32.768, 32.768)) for i in range(self.input_dim)]
        
        ss = SearchSpace(variables)

        return ss


@problem_registry.register("Ackley")
class Ackley(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift - 12)
        self.dtype = np.float64

        self.a = np.array([20], dtype=self.dtype)
        self.b = np.array([0.2], dtype=self.dtype)
        self.c = np.array([0.3 * math.pi], dtype=self.dtype)

        super(Ackley, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift - 0.73)

        n = X.shape[0]
        d = X.shape[1]
        a, b, c = self.a, self.b, self.c

        part1 = -a * np.exp(-b / math.sqrt(d) * np.linalg.norm(X, axis=-1))
        part2 = -(np.exp(np.mean(np.cos(c * X), axis=-1)))
        y = part1 + part2 + a + math.e

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-32.768, 32.768)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss

    
    
@problem_registry.register("Ellipsoid")
class EllipsoidOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        self.condition = 1e6

        super(EllipsoidOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = x[0] * x[0]
            for i in range(1, d):
                exponent = 1.0 * i / (d - 1)
                temp += pow(self.condition, exponent) * x[i] * x[i]
            y = np.append(y, temp)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Discus")
class DiscusOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        self.condition = 1e6

        super(DiscusOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = self.condition * x[0] * x[0]
            for i in range(1, d):
                temp += x[i] * x[i]
            y = np.append(y, temp)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("BentCigar")
class BentCigarOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        self.condition = 1e6

        super(BentCigarOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = x[0] * x[0]
            for i in range(1, d):
                temp += self.condition * x[i] * x[i]
            y = np.append(y, temp)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("SharpRidge")
class SharpRidgeOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        self.alpha = 100.0

        super(SharpRidgeOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        d_vars_40 = d / 40.0
        vars_40 = int(math.ceil(d_vars_40))
        y = np.array([])
        for x in X:
            temp = 0
            for i in range(vars_40, d):
                temp += x[i] * x[i]
            temp = self.alpha * math.sqrt(temp / d_vars_40)
            for i in range(vars_40):
                temp += x[i] * x[i] / d_vars_40
            y = np.append(y, temp)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("GriewankRosenbrock")
class GriewankRosenbrockOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        super(GriewankRosenbrockOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            temp = 0
            for i in range(len(x) - 1):
                temp1 = x[i] * x[i] - x[i + 1]
                temp2 = 1.0 - x[i]
                temp3 = 100.0 * temp1**2 + temp2**2
                temp += temp3 / 4000.0 - math.cos(temp3)
            y = np.append(y, temp)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results


    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


@problem_registry.register("Katsuura")
class KatsuuraOptBenchmark(SyntheticProblemBase):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        assert "params" in kwargs
        parameters = kwargs["params"]
        self.input_dim = parameters["input_dim"]

        if "shift" in parameters:
            self.shift = parameters["shift"]
        else:
            shift = np.random.random(size=(self.input_dim, 1)).T
            self.shift = (shift * 2 - 1) * 0.02

        if "stretch" in parameters:
            self.stretch = parameters["stretch"]
        else:
            self.stretch = np.array([1] * self.input_dim, dtype=np.float64)

        self.optimizers = tuple(self.shift)
        self.dtype = np.float64

        super(KatsuuraOptBenchmark, self).__init__(
            task_name=task_name,
            seed=seed,
            workload=workload,
            budget_type=budget_type,
            budget=budget,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])

        X = self.stretch * (X - self.shift)

        n = X.shape[0]
        d = X.shape[1]

        y = np.array([])
        for x in X:
            result = 1.0
            for i in range(len(x)):
                temp = 0.0
                for j in range(1, 33):
                    temp1 = 2.0**j
                    temp += abs(temp1 * x[i] - round(temp1 * x[i])) / temp1
                temp = 1.0 + (i + 1) * temp
                result *= temp ** (10.0 / (len(x) ** 1.2))
            y = np.append(y, result)

        results = {list(self.objective_info.keys())[0]: float(y)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results


    def get_configuration_space(self) -> SearchSpace:
        variables =  [Continuous(f'x{i}', (-5.0, 5.0)) for i in range(self.input_dim)]
        ss = SearchSpace(variables)
        return ss


# def plot_true_function(obj_fun_list, Dim, dtype, Exper_folder=None, plot_type="1D"):
#     for fun in obj_fun_list:
#         obj_fun = get_problem(fun, seed=0, Dim=Dim)

#         if Exper_folder is not None:
#             if not os.path.exists(f"{Exper_folder}/true_f/{plot_type}/"):
#                 os.makedirs(f"{Exper_folder}/true_f/{plot_type}/")
#             name = obj_fun.task_name
#             if "." in obj_fun.task_name:
#                 name = name.replace(".", "|")
#             save_load = f"{Exper_folder}/true_f/{plot_type}/{name}"

#         if plot_type == "1D":
#             fig, ax = plt.subplots(1, 1, figsize=(16, 6))
#             test_x = np.arange(-1, 1, 0.005, dtype=dtype)
#             test_x = test_x[:, np.newaxis]
#             dic_list = []
#             for i in range(len(test_x)):
#                 for j in range(Dim):
#                     dic_list.append({f"x{j}": test_x[i][j]})
#             test_y = []
#             for j in range(len(test_x)):
#                 test_y.append(obj_fun.f(dic_list[j])["f1"])
#             test_y = np.array(test_y)
#             ax.plot(test_x, test_y, "r-", linewidth=1, alpha=1)
#             ax.legend(["True f(x)"])
#             ax.set_title(fun)
#             plt.savefig(save_load)
#             plt.close(fig)
#         elif plot_type == "2D":
#             x = np.linspace(-1, 1, 101)
#             y = np.linspace(-1, 1, 101)
#             X, Y = np.meshgrid(x, y)
#             all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
#             v_name = [f"x{i}" for i in range(Dim)]
#             dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
#             Z_true = []
#             for j in range(len(all_sample)):
#                 Z_true.append(obj_fun.f(dic_list[j])["f1"])
#             Z_true = np.asarray(Z_true)
#             Z_true = Z_true[:, np.newaxis]
#             Z_true = Z_true.reshape(X.shape)

#             optimizers = obj_fun.optimizers
#             fig = plt.figure(figsize=(10, 8))
#             ax = plt.subplot(111)
#             box = ax.get_position()
#             ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
#             a = plt.contourf(X, Y, Z_true, 100, cmap=plt.cm.summer)
#             plt.colorbar(a)
#             plt.title(fun)
#             plt.draw()
#             plt.savefig(save_load, dpi=300)
#             plt.close(fig)
#         elif plot_type == "3D":
#             fig = plt.figure()
#             ax = plt.axes(projection="3d")
#             x = np.linspace(-1, 1, 101)
#             y = np.linspace(-1, 1, 101)
#             X, Y = np.meshgrid(x, y)
#             all_sample = np.array(np.c_[X.ravel(), Y.ravel()])
#             v_name = [f"x{i}" for i in range(Dim)]
#             dic_list = [dict(zip(v_name, sample)) for sample in all_sample]
#             Z_true = []
#             for j in range(len(all_sample)):
#                 Z_true.append(obj_fun.f(dic_list[j])["f1"])
#             Z_true = np.asarray(Z_true)
#             Z_true = Z_true[:, np.newaxis]
#             Z_true = Z_true.reshape(X.shape)
#             a = ax.plot_surface(X, Y, Z_true, cmap=plt.cm.summer)
#             plt.colorbar(a)
#             plt.draw()
#             plt.savefig(save_load, dpi=300)
#             plt.close(fig)


# def get_problem(fun, seed, Dim):
#     # 从注册表中获取任务类
#     task_class = g_problem_registry.get(fun)

#     if task_class is not None:
#         problem = task_class(
#             task_name=f"{fun}_{1}",
#             task_id=1,
#             budget=100000,
#             seed=seed,
#             params={"input_dim": Dim},
#         )
#     return problem


# if __name__ == "__main__":
#     Dim = 2
#     obj_fun_list = [
#         # 'Sphere',
#         # 'Rastrigin',
#         # 'Ackley',
#         # 'Schwefel',
#         # 'LevyR',
#         # 'Griewank',
#         # 'Rosenbrock',
#         # 'DropwaveR',
#         # 'Langermann',
#         # 'RotatedHyperEllipsoid',
#         # 'SumOfDifferentPowers',
#         # 'StyblinskiTang',
#         # 'Powell',
#         # 'DixonPrice',
#         "cp"
#     ]



    # from transopt.Benchmark.BenchBase.TransferOptBenchmark import TransferOptBenchmark
    #
    # seed=0
    # test_suits = TransferOptBenchmark(seed=seed)
    #
    # for fun in obj_fun_list:
    #     shift = np.zeros(Dim)
    #     stretch = np.ones(Dim)
    #     problem = getProblem(fun, seed, Dim)
    #     test_suits.add_task(problem)
    #
    # test_x = np.arange(-5, 5.05, 0.005, dtype=np.float64)
    # test_x = test_x[:, np.newaxis]
    # dic_list = []
    #
    # for i in range(len(test_x)):
    #     for j in range(Dim):
    #         dic_list.append({f'x{j}':test_x[i][j]})
    # for i in obj_fun_list:
    #     test_y = []
    #     for j in range(len(test_x)):
    #         test_y.append(test_suits.f(dic_list[j])['f1'])
    #     test_y = np.array(test_y)
    #     f, ax = plt.subplots(1, 1, figsize=(16, 6))
    #     ax.plot(test_x, test_y, 'r-', linewidth=1, alpha=1)
    #     plt.show()
    #
    #     test_suits.roll()
