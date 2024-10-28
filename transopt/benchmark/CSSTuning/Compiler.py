import numpy as np
from csstuning.compiler.compiler_benchmark import GCCBenchmark, LLVMBenchmark

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *


@problem_registry.register("Compiler_GCC")
class GCCTuning(NonTabularProblem):
    problem_type = 'compiler'
    workloads = GCCBenchmark.AVAILABLE_WORKLOADS
    num_variables = 104
    num_objectives = 3
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, knobs=None, **kwargs):        
        self.workload = workload or GCCBenchmark.AVAILABLE_WORKLOADS[0]
        self.benchmark = GCCBenchmark(workload=self.workload)
        
        all_knobs = self.benchmark.get_config_space()
        self.knobs = {k: all_knobs[k] for k in (knobs or all_knobs)}
        self.num_variables = len(self.knobs)
        
        super().__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        np.random.seed(seed)

    def get_configuration_space(self):
        variables = []
        for knob_name, knob_details in self.knobs.items():
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                variables.append(Integer(knob_name, range_))

        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            "execution_time": "minimize",
            "compilation_time": "minimize",
            "file_size": "minimize",
            # "maxrss": "minimize",
            # "PAPI_TOT_CYC": "minimize",
            # "PAPI_TOT_INS": "minimize",
            # "PAPI_BR_MSP": "minimize",
            # "PAPI_BR_PRC": "minimize",
            # "PAPI_BR_CN": "minimize",
            # "PAPI_MEM_WCY": "minimize",
        }
    
    def get_problem_type(self):
        return self.problem_type
    
    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):        
        try:
            perf = self.benchmark.run(configuration)
            return {obj: perf.get(obj, 1e10) for obj in self.get_objectives()}
        except Exception as e:
            return {obj: 1e10 for obj in self.get_objectives()}
        


@problem_registry.register("Compiler_LLVM")
class LLVMTuning(NonTabularProblem):
    problem_type = 'compiler'
    workloads = LLVMBenchmark.AVAILABLE_WORKLOADS
    num_variables = 82
    num_objectives = 3
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, knobs=None, **kwargs):
        self.workload = workload or LLVMBenchmark.AVAILABLE_WORKLOADS[0]
        self.benchmark = LLVMBenchmark(workload=self.workload)
        
        all_knobs = self.benchmark.get_config_space()
        self.knobs = {k: all_knobs[k] for k in (knobs or all_knobs)}
        self.num_variables = len(self.knobs)
        
        super().__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        np.random.seed(seed)

    def get_configuration_space(self):
        variables = []
        for knob_name, knob_details in self.knobs.items():
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                variables.append(Integer(knob_name, range_))

        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            "execution_time": "minimize",
            "compilation_time": "minimize",
            "file_size": "minimize",
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
    benchmark = GCCBenchmark(workload="cbench-automotive-bitcount")
    conf = {
        
    }
    benchmark.run(conf)

