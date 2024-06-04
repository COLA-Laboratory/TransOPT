import os
import re
import math
import logging
import subprocess
import numpy as np
from typing import Union, Dict
from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace


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
        self.src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Absolut/src")

        # Select antigens based on workload. The workload is an integer between 0 and 193.
        command = "./AbsolutNoLib listAntigens"
        result = subprocess.run(command, cwd=self.src_path, capture_output=True, text=True, shell=True).stdout.split("\n")
        antigens = [result[i].split("\t")[1] for i in range(194)]
        self.antigen = antigens[workload]

        # Download the antigen data
        command = "./AbsolutNoLib info_filenames " + self.antigen
        result = subprocess.run(command, cwd=self.src_path, capture_output=True, text=True, shell=True).stdout
        url_pattern = r'(https?://\S+\.zip)'
        url = re.findall(url_pattern, result)[0]
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(self.src_path, filename)):
            command = "wget " + url
            result = subprocess.run(command, cwd=self.src_path, capture_output=True, text=True, shell=True)
            command = "unzip " + filename
            result = subprocess.run(command, cwd=self.src_path, capture_output=True, text=True, shell=True)

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
        command = "./AbsolutNoLib singleBinding " + self.antigen + " " + CDR3
        result = subprocess.run(command, cwd=self.src_path, capture_output=True, text=True, shell=True).stdout
        energy = result.split("\t")[-3]

        results={list(self.objective_info.keys())[0]: float(energy)}
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
    
    