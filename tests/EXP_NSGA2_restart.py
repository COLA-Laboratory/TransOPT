import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from transopt.benchmark.HPO.HPO import HPO_ERM
import os
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt  # 添加這行
from pymoo.core.population import Population



class HPOProblem(Problem):
    def __init__(self, task_name, budget_type, budget, seed, workload, data_file):
        self.hpo = HPO_ERM(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, algorithm='ERM',architecture='resnet', model_size=18, optimizer='nsga2_augment_true')
        original_ranges = self.hpo.configuration_space.original_ranges
        n_var = len(original_ranges)
        xl = np.array([original_ranges[key][0] for key in original_ranges])
        xu = np.array([original_ranges[key][1] for key in original_ranges])
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

        # Load data from specified file
        self.data = {}
        for filename in os.listdir(data_file):
            file_path = os.path.join(data_file, filename)
            if os.path.isfile(file_path):
                import json
                import re
                with open(file_path, 'r') as f:
                    
                    content = json.load(f)
                    # Extract decision variables from filename
                    x = []
                    for key in original_ranges.keys():
                        pattern = rf'({key}_)([\d.e-]+)'
                        match = re.search(pattern, filename)
                        if match:
                            value = float(match.group(2))
                            if key in ['l', 'weight_decay']:
                                value = np.log10(value)
                            x.append(value)
                        elif key == 'epoch':
                            # Special handling for 'epoch' which is an integer
                            epoch_match = re.search(r'epoch_(\d+)', filename)
                            if epoch_match:
                                x.append(int(epoch_match.group(1)))
                        elif key in ['data_augmentation', 'class_balanced', 'nonlinear_classifier']:
                            # Special handling for boolean values
                            bool_match = re.search(rf'{key}_(True|False)', filename)
                            if bool_match:
                                x.append(bool_match.group(1) == 'True')
                    self.data[filename] = {
                        'x': x,
                        'test_standard_acc': content['test_standard_acc'],
                        'test_robust_acc': np.mean([v for k, v in content.items() if k.startswith('test_') and k != 'test_standard_acc'])
                    }
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for x in X:
            config = {}
            for i, param_name in enumerate(self.hpo.configuration_space.original_ranges):
                if param_name == 'epoch':
                    config[param_name] = int(x[i])
                else:
                    config[param_name] = x[i]
            val_acc = self.hpo.objective_function(config)
            f1.append(1 - val_acc['test_standard_acc'])  # Minimize 1 - accuracy
            f2.append(1- np.mean([v for k, v in val_acc.items() if k.startswith('test_') and k != 'test_standard_acc']))  # Minimize number of epochs
        out["F"] = np.column_stack([f1, f2])

if __name__ == "__main__":
    data_file = '/home/haxx/transopt_tmp/output/results/nsga2_false_augment_ERM_resnet_18_RobCifar10_0'
    problem = HPOProblem(task_name='test_task', budget_type='FEs', budget=3000, seed=0, workload=0, 
                         data_file=data_file)
    
    # Extract objectives from the loaded data
    F = np.array([[1 - data['test_standard_acc'], 1 - data['test_robust_acc']] for data in problem.data.values()])
    
    # Perform non-dominated sorting
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    
    # Initialize lists for the initial population
    initial_X = []
    initial_F = []
    pop_size = 40  # Assuming a population size of 40, adjust as needed

    # Iterate through fronts and add solutions layer by layer
    for front in fronts:
        front_solutions = [list(problem.data.values())[i] for i in front]
        
        if len(initial_X) + len(front_solutions) <= pop_size:
            initial_X.extend([sol['x'] for sol in front_solutions])
            initial_F.extend([[1 - sol['test_standard_acc'], 1 - sol['test_robust_acc']] for sol in front_solutions])
        else:
            remaining_slots = pop_size - len(initial_X)
            if remaining_slots > 0:
                # Use niching to select the most diverse solutions from the current front
                front_x = np.array([sol['x'] for sol in front_solutions])
                from scipy.spatial.distance import cdist
                distances = cdist(front_x, front_x)
                selected_indices = []
                
                while len(selected_indices) < remaining_slots:
                    if len(selected_indices) == 0:
                        selected_indices.append(np.random.choice(len(front_x)))
                    else:
                        min_distances = np.min(distances[:, selected_indices], axis=1)
                        min_distances[selected_indices] = -np.inf
                        selected_indices.append(np.argmax(min_distances))
                
                initial_X.extend([front_solutions[i]['x'] for i in selected_indices])
                initial_F.extend([[1 - front_solutions[i]['test_standard_acc'], 1 - front_solutions[i]['test_robust_acc']] for i in selected_indices])
            break

    # Create the initial population with X, F, and set evaluated correctly
    initial_pop = Population.new(X=np.array(initial_X), F=np.array(initial_F))

    for ind in initial_pop:
        ind.evaluated = {"F", "CV"}  # Set evaluated to include both F and CV

    # 創建NSGA2算法
    algorithm = NSGA2(pop_size=len(initial_pop))
    
    # 設置總迭代次數
    total_evaluations = 2000
    current_evaluations = len(problem.data)
    remaining_evaluations = total_evaluations - current_evaluations
    remaining_generations = max(1, remaining_evaluations // pop_size)

    # 繼續優化
    res = minimize(problem, algorithm, ('n_gen', remaining_generations), seed=1, verbose=True)
    
    print("Best solutions found:")
    for i in range(len(res.X)):
        print(f"Solution {i+1}: {res.X[i]}, Objectives: {res.F[i]}")
