import collections
import os
import random
import time
import json
from typing import Dict, Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transopt.benchmark.HPO import datasets

import transopt.benchmark.HPO.misc as misc
from transopt.agent.registry import problem_registry
from transopt.benchmark.HPO.fast_data_loader import (FastDataLoader,
                                                     InfiniteDataLoader)
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPO import algorithms
from transopt.benchmark.HPO.hparams_registry import get_hparam_space
from transopt.benchmark.HPO.networks import SUPPORTED_ARCHITECTURES

class HPO_base(NonTabularProblem):
    problem_type = 'hpo'
    num_variables = 4
    num_objectives = 1
    workloads = []
    fidelity = None
    
    ALGORITHMS = [
        'ERM',
        # 'BayesianNN',
        # 'GLMNet'
    ]
    
    ARCHITECTURES = SUPPORTED_ARCHITECTURES
    
    DATASETS = [
    "RobCifar10",
    # "RobCifar100",
    # "RobImageNet",
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, algorithm, architecture, model_size, **kwargs
        ):
        
        # Check if algorithm is valid
        if algorithm not in HPO_base.ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {HPO_base.ALGORITHMS}")
        self.algorithm_name = algorithm

        # Check if workload is valid
        if workload < 0 or workload >= len(HPO_base.DATASETS):
            raise ValueError(f"Invalid workload: {workload}. Must be between 0 and {len(HPO_base.DATASETS) - 1}")
        self.dataset_name = HPO_base.DATASETS[workload]

        # Check if architecture is valid
        if architecture not in HPO_base.ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {architecture}. Must be one of {list(HPO_base.ARCHITECTURES.keys())}")
        if model_size not in HPO_base.ARCHITECTURES[architecture]:
            raise ValueError(f"Invalid model_size: {model_size} for architecture: {architecture}. Must be one of {HPO_base.ARCHITECTURES[architecture]}")
        self.architecture = architecture
        self.model_size = model_size
        
        self.hpo_optimizer = kwargs.get('optimizer', 'random')

        super(HPO_base, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        
        self.query_counter = kwargs.get('query_counter', 0)
        self.trial_seed = seed
        self.hparams = {}
        
        base_dir = kwargs.get('base_dir', os.path.expanduser('~'))
        print(base_dir)
        self.data_dir = os.path.join(base_dir, 'transopt_tmp/data/')
        self.model_save_dir  = os.path.join(base_dir, f'transopt_tmp/output/models/{self.hpo_optimizer}_{self.algorithm_name}_{self.architecture}_{self.model_size}_{self.dataset_name}_{seed}/')
        self.results_save_dir  = os.path.join(base_dir, f'transopt_tmp/output/results/{self.hpo_optimizer}_{self.algorithm_name}_{self.architecture}_{self.model_size}_{self.dataset_name}_{seed}/')
        
        print(f"Selected algorithm: {self.algorithm_name}, dataset: {self.dataset_name}")
        print(f"Model architecture: {self.architecture}")
        if hasattr(self, 'model_size'):
            print(f"Model size: {self.model_size}")
        else:
            print("Model size not specified")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get the GPU ID from hparams, default to 0 if not specified
        gpu_id = kwargs.get('gpu_id', 0)
        
        if torch.cuda.is_available():
            # Check if the specified GPU exists
            if gpu_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not found. Defaulting to CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # 将最终使用的设备写入hparams
        self.hparams['device'] = str(self.device)
        
        print(f"Using device: {self.device}")
        
        if self.dataset_name in vars(datasets):
            self.dataset = vars(datasets)[self.dataset_name](root=self.data_dir, augment=kwargs.get('augment', None))
        else:
            raise NotImplementedError
        if self.hparams.get('augment', None) == 'mixup':
            self.mixup = True
        else:
            self.mixup = False
        
        print(f"Using augment: {kwargs.get('augment', None)}")
        
        self.eval_loaders, self.eval_loader_names = self.create_test_loaders(128)


        self.checkpoint_vals = collections.defaultdict(lambda: [])
        
    def create_train_loaders(self, batch_size):
        if not hasattr(self, 'dataset') or self.dataset is None:
            raise ValueError("Dataset not initialized. Please ensure self.dataset is set before calling this method.")
        
        train_loaders = FastDataLoader(
            dataset=self.dataset.datasets['train'],
            batch_size=batch_size,
            num_workers=2)  # Assuming N_WORKERS is 2, adjust if needed
        
        val_loaders = FastDataLoader(
            dataset=self.dataset.datasets['val'],
            batch_size=batch_size,
            num_workers=2)  # Assuming N_WORKERS is 2, adjust if needed

        return train_loaders, val_loaders
    

    def create_test_loaders(self, batch_size):
        if not hasattr(self, 'dataset') or self.dataset is None:
            raise ValueError("Dataset not initialized. Please ensure self.dataset is set before calling this method.")
        
        eval_loaders = []
        eval_loader_names = []

        # Get all available test set names
        available_test_sets = self.dataset.get_available_test_set_names()

        for test_set_name in available_test_sets:
            if test_set_name.startswith('test_'):
                eval_loaders.append(FastDataLoader(
                    dataset=self.dataset.datasets[test_set_name],
                    batch_size=batch_size,
                    num_workers=2))  # Assuming N_WORKERS is 2, adjust if needed
                eval_loader_names.append(test_set_name)

        return eval_loaders, eval_loader_names
    

    def save_checkpoint(self, filename):
        save_dict = {
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(self.model_save_dir, filename))
        
    def get_configuration_space(
        self, seed: Union[int, None] = None):

        hparam_space = get_hparam_space(self.algorithm_name, self.model_size, self.architecture)
        variables = []

        for name, (hparam_type, range) in hparam_space.items():
            if hparam_type == 'categorical':
                variables.append(Categorical(name, range))
            elif hparam_type == 'float':
                variables.append(Continuous(name, range))
            elif hparam_type == 'int':
                variables.append(Integer(name, range))
            elif hparam_type == 'log':
                variables.append(LogContinuous(name, range))

        ss = SearchSpace(variables)
        return ss
    
    def get_fidelity_space(
        self, seed: Union[int, None] = None):

        fs = FidelitySpace([
            Integer("epoch", [1, 100])  # Adjust the range as needed
        ])
        return fs
    
    def train(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.epoches = configuration['epoch']
        print(f"Total epochs: {self.epoches}")
                
        self.train_loader, self.val_loader = self.create_train_loaders(self.hparams['batch_size'])
        
        self.hparams['nonlinear_classifier'] = True
    
        for epoch in range(self.epoches):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            self.algorithm.train()
            total_batches = len(self.train_loader)
            for x, y in tqdm(self.train_loader, total=total_batches, desc=f"Epoch {epoch+1}/{self.epoches}", unit="batch"):
                step_start_time = time.time()
                minibatches_device = [(x.to(self.device), y.to(self.device))]

                step_vals = self.algorithm.update(minibatches_device)
                self.checkpoint_vals['step_time'].append(time.time() - step_start_time)

                for key, val in step_vals.items():
                    self.checkpoint_vals[key].append(val)
                
                # Update epoch statistics
                epoch_loss += step_vals.get('loss', 0.0)
                epoch_correct += step_vals.get('correct', 0)
                epoch_total += sum(len(x) for x, _ in minibatches_device)

            # Compute and print epoch metrics
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            epoch_loss /= len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.epoches} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate on validation set
        val_acc = self.evaluate_loader(self.val_loader)

        # Calculate final results after all epochs
        results = {
            'epoch': self.epoches,
            'epoch_time': time.time() - epoch_start_time,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_acc': val_acc,
        }

        # Evaluate on all test loaders
        for name, loader in zip(self.eval_loader_names, self.eval_loaders):
            results[f'{name}_acc'] = self.evaluate_loader(loader)

        # Calculate memory usage
        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.**3)

        results['hparams'] = self.hparams
        
        return results

    def save_epoch_results(self, results):
        epoch_path = os.path.join(self.results_save_dir, f"epoch_{results['epoch']}.json")
        with open(epoch_path, 'w') as f:
            json.dump(results, f, indent=2)

    def evaluate_loader(self, loader):
        self.algorithm.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                p = self.algorithm.predict(x)
                correct += (p.argmax(1).eq(y) if p.size(1) != 1 else p.gt(0).eq(y)).float().sum().item()
                total += len(x)
        self.algorithm.train()
        return correct / total

    def get_score(self, configuration: dict):
        for key, value in configuration.items():
            self.hparams[key] = value
        
        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes, self.architecture, self.model_size, self.mixup, self.device, self.hparams)
        self.algorithm.to(self.device)
        
        self.query_counter += 1
        results = self.train(configuration)
        
        # Construct filename with query and all hyperparameters
        filename_parts = [f"{self.query_counter}"]
        for key, value in configuration.items():
            filename_parts.append(f"{key}_{value}")
        filename = "_".join(filename_parts)

        # Save results
        epochs_path = os.path.join(self.results_save_dir, f"{filename}.jsonl")
        with open(epochs_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final checkpoint and mark as done
        self.save_checkpoint(f"{filename}_model.pkl")
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')

        val_acc = results['val_acc']
        
        return val_acc, results
        

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

        if fidelity is None:
            fidelity = {"epoch": 50}
        
        # Convert log scale values back to normal scale
        c = self.configuration_space.map_to_design_space(configuration)
        
        # Add fidelity (epoch) to the configuration
        c["epoch"] = fidelity["epoch"]        
        c['class_balanced'] = True
        c['nonlinear_classifier'] = True
        
        val_acc, results = self.get_score(c)

        acc = {list(self.objective_info.keys())[0]: float(val_acc)}
        
        # Add standard test accuracy
        acc['test_standard_acc'] = float(results['test_standard_acc'])
        
        # Calculate average of other test accuracies
        other_test_accs = [v for k, v in results.items() if k.startswith('test_') and k != 'test_standard_acc']
        if other_test_accs:
            acc['test_robust_acc'] = float(sum(other_test_accs) / len(other_test_accs))
        
        
        return acc
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"


@problem_registry.register("HPO_ERM")
class HPO_ERM(HPO_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):            
        algorithm = kwargs.pop('algorithm', 'ERM')
        architecture = kwargs.pop('architecture', 'resnet')
        model_size = kwargs.pop('model_size', 18)
        optimizer = kwargs.pop('optimizer', 'random')
        base_dir = kwargs.pop('base_dir', os.path.expanduser('~'))
        
        super(HPO_ERM, self).__init__(
            task_name=task_name, 
            budget_type=budget_type, 
            budget=budget, 
            seed=seed, 
            workload=workload, 
            algorithm=algorithm, 
            architecture=architecture, 
            model_size=model_size,
            optimizer=optimizer,
            base_dir=base_dir,
            **kwargs
        )

def test_all_combinations():
    print("Testing all combinations of architectures, algorithms, and datasets...")
    
    for architecture in HPO_base.ARCHITECTURES:
        for model_size in HPO_base.ARCHITECTURES[architecture]:
            for algorithm in HPO_base.ALGORITHMS:
                for dataset_index, dataset in enumerate(HPO_base.DATASETS):
                    print(f"Testing {architecture}-{model_size} with {algorithm} on {dataset}...")
                    try:
                        # Create an instance of HPO_base
                        hpo = HPO_base(task_name='test_combination', 
                                       budget_type='FEs', budget=100, seed=0, 
                                       workload=dataset_index, algorithm=algorithm, 
                                       architecture=architecture, model_size=model_size, optimizer='test_combination')
                        
                        # Get the configuration space
                        config_space = hpo.get_configuration_space()
                        
                        # Get the fidelity space
                        fidelity_space = hpo.get_fidelity_space()
                        
                        # Sample a random configuration
                        config = {}
                        for name, var in config_space.get_design_variables().items():
                            if isinstance(var, Integer):
                                config[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
                            elif isinstance(var, Continuous) or isinstance(var, LogContinuous):
                                config[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
                            elif isinstance(var, Categorical):
                                config[name] = np.random.choice(var.search_space_range)
                                
                        
                        
                        # Sample a random fidelity
                        fidelity = {}
                        for name, var in fidelity_space.get_fidelity_range().items():
                            if isinstance(var, Integer):
                                fidelity[name] = np.random.randint(var.search_space_range[0], var.search_space_range[1] + 1)
                            elif isinstance(var, Continuous):
                                fidelity[name] = np.random.uniform(var.search_space_range[0], var.search_space_range[1])
                            elif isinstance(var, Categorical):
                                fidelity[name] = np.random.choice(var.search_space_range)
                        
                        # Set a small epoch for quick testing
                        fidelity['epoch'] = 2
                        
                        # Run the objective function
                        result = hpo.objective_function(configuration=config, fidelity=fidelity)
                        
                        print(f"Configuration: {config}")
                        print(f"Fidelity: {fidelity}")
                        print(f"Result: {result}")
                        
                        assert list(hpo.get_objectives().keys())[0] in result, f"Result should contain '{list(hpo.get_objectives().keys())[0]}'"
                        assert 0 <= result[list(hpo.get_objectives().keys())[0]] <= 1, f"{list(hpo.get_objectives().keys())[0]} should be between 0 and 1"
                        
                        print(f"Test passed for {architecture}-{model_size} with {algorithm} on {dataset}!")
                        print("--------------------")
                    except Exception as e:
                        print(f"Error occurred during test for {architecture}-{model_size} with {algorithm} on {dataset}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        print("--------------------")

if __name__ == "__main__":
    import torch
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Run the comprehensive test
    try:
        test_all_combinations()
    except Exception as e:
        print(f"Error occurred during HPO_ERM test: {str(e)}")
        import traceback
        traceback.print_exc()



