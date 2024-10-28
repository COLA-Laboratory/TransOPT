import os
import re
import subprocess
from typing import Dict, Union

import numpy as np

from transopt.agent.registry import problem_registry
from transopt.benchmark.CPD.Absolut.absolut_container import \
    AbsolutContainerManager
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


@problem_registry.register("Absolut")
class Absolut(NonTabularProblem):
    problem_type = "CPD"
    num_variables = 11
    num_objectives = 1
    workloads = np.arange(194).tolist()
    fidelity = None
    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        self.manager = AbsolutContainerManager()

        # Select antigens based on workload. The workload is an integer between 0 and 193.
        antigens = self.manager.get_available_workloads()
        self.antigen = antigens[workload]   

        # Prepare the antigen data
        self.manager.prepare_antigen(self.antigen)

        super(Absolut, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )

    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        CDR3 = ''.join(configuration.values())
        energy = self.manager.predict_energy(self.antigen, CDR3)

        results = {list(self.objective_info.keys())[0]: float(energy)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    def get_configuration_space(self) -> SearchSpace:
        variables = [Categorical(
            f'aa{i}',
            ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        ) for i in range(11)]
        ss = SearchSpace(variables)
        return ss

    def get_fidelity_space(self) -> FidelitySpace:
        fs = FidelitySpace([])
        return fs
    
    def get_objectives(self) -> Dict:
        return {"energy":'minimize'}
    
    def get_problem_type(self) -> str:
        return "CPD"
    
    