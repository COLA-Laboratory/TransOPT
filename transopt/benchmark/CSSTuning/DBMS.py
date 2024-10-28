import numpy as np
from csstuning.dbms.dbms_benchmark import MySQLBenchmark

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


@problem_registry.register("DBMS_MySQL")
class MySQLTuning(NonTabularProblem):
    problem_type = 'dbms'
    workloads = MySQLBenchmark.AVAILABLE_WORKLOADS
    num_variables = 197
    num_objectives = 2
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, knobs=None, **kwargs):        
        self.workload = workload or MySQLBenchmark.AVAILABLE_WORKLOADS[0]

        self.benchmark = MySQLBenchmark(workload=self.workload)
        self.knobs = self.benchmark.get_config_space()
        self.num_variables = len(self.knobs)
        
        super().__init__(task_name, budget_type, budget, workload, seed)
        np.random.seed(seed)


    def get_configuration_space(self):
        variables = []
        
        for knob_name, knob_details in self.knobs.items():
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                if range_[1] > np.iinfo(np.int64).max:
                    variables.append(ExponentialInteger(knob_name, range_))
                else:
                    variables.append(Integer(knob_name, range_))

        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            "latency": "minimize",
            "throughput": "maximize",
        }
        
    def get_problem_type(self):
        return self.problem_type
        
    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):
        try:
            perf = self.benchmark.run(configuration)
            return {obj: perf.get(obj, 1e10) for obj in self.get_objectives()}
        except Exception as e:
            return {obj: 1e10 for obj in self.get_objectives()}

if __name__ == "__main__":
    # a = DBMSTuning("1", 121, 0, 1)
    pass
