import ConfigSpace as cs
import time
import numpy as np
from typing import Any, Dict, List, Optional, Protocol, Tuple

from tpe.optimizer import TPEOptimizer

from transopt.benchmark.HPO.HPO import HPO_ERM
from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc

# Create a single HPO_ERM instance
hpo = HPO_ERM(task_name='tpe_optimization', budget_type='FEs', budget=100, seed=42, workload=0, optimizer='tpe')

class formal_obj(ObjectiveFunc):
    def __init__(self, f):
        self.f = f
    
    def __call__(self, eval_config: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        start = time.time()
        results = self.f(eval_config)
        return {'loss': 1 - results['function_value']}, time.time() - start

# Create an instance of formal_obj with hpo.objective_function

# Define the configuration space
def get_configspace():
    original_ranges = hpo.configuration_space.original_ranges
    hyperparameters = [cs.UniformFloatHyperparameter(param_name, lower=param_range[0], upper=param_range[1]) for param_name, param_range in original_ranges.items() ]
    space = cs.ConfigurationSpace(hyperparameters)
    
    return space

if __name__ == "__main__":
    # Create the configuration space
    config_space = get_configspace()
    obj_f = formal_obj(hpo.objective_function)

    # Initialize TPE Optimizer
    opt = TPEOptimizer(obj_func=obj_f, config_space=config_space, n_init=10, max_evals=100, resultfile='tpe_results.json')
    
    # Run optimization
    best_config, best_value = opt.optimize()
