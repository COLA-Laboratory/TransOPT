from bohb import BOHB
import bohb.configspace as cs
from transopt.benchmark.HPO.HPO import HPO_ERM
import numpy as np

# Create a single HPO_ERM instance
hpo = HPO_ERM(task_name='bohb_optimization', budget_type='FEs', budget=2000, seed=42, workload=0,algorithm='ERM',architecture='resnet', model_size=18, optimizer='bohb')

# Define the objective function
def objective(config, budget):
    result = hpo.objective_function(configuration=config, fidelity={'epoch': int(budget)})
    return 1 - result['function_value']  # BOHB minimizes, so we return the function value directly

# Define the configuration space
def get_configspace():
    original_ranges = hpo.configuration_space.original_ranges
    hyperparameters = [cs.UniformHyperparameter(param_name, lower=param_range[0], upper=param_range[1]) for param_name, param_range in original_ranges.items() ]
    space = cs.ConfigurationSpace(hyperparameters)
    
    return space

if __name__ == "__main__":
    # Create the configuration space
    config_space = get_configspace()
    
    # Initialize BOHB
    bohb = BOHB(configspace=config_space,
                eta=3, min_budget=1, max_budget=50, n_samples=200,
                evaluate=objective)
    
    # Run optimization
    results = bohb.optimize()
    