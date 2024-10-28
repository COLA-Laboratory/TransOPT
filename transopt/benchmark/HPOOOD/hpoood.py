# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import random
import collections
import time
import json
import shutil
import hashlib
import copy


from torchvision import datasets, transforms

from typing import Dict, Union


from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.benchmark.HPOOOD.hparams_registry import random_hparams, default_hparams, get_hparams
from transopt.benchmark.HPOOOD import ooddatasets
from transopt.benchmark.HPOOOD import misc
from transopt.benchmark.HPOOOD import algorithms
from transopt.benchmark.HPOOOD.fast_data_loader import InfiniteDataLoader, FastDataLoader


def make_record(step, hparams_seed, envs):
    """envs is a list of (in_acc, out_acc, is_test_env) tuples"""
    result = {
        'args': {'test_envs': [], 'hparams_seed': hparams_seed},
        'step': step
    }
    for i, (in_acc, out_acc, is_test_env) in enumerate(envs):
        if is_test_env:
            result['args']['test_envs'].append(i)
        result[f'env{i}_in_acc'] = in_acc
        result[f'env{i}_out_acc'] = out_acc
    return result
        
        

class HPOOOD_base(NonTabularProblem):
    DATASETS = [
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
    ]

    problem_type = 'hpoood'
    num_variables = 10
    num_objectives = 1
    workloads = []
    fidelity = None
    
    ALGORITHMS = [
        'ERM',
        'Fish',
        'IRM',
        'GroupDRO',
        'Mixup',
        'MLDG',
        'CORAL',
        'MMD',
        'DANN',
        'CDANN',
        'MTL',
        'SagNet',
        'ARM',
        'VREx',
        'RSC',
        'SD',
        'ANDMask',
        'SANDMask',
        'IGA',
        'SelfReg',
        "Fishr",
        'TRM',
        'IB_ERM',
        'IB_IRM',
        'CAD',
        'CondCAD',
        'Transfer',
        'CausIRL_CORAL',
        'CausIRL_MMD',
        'EQRM',
        'RDM',
        'ADRMX',
    ]

    def __init__(
        self, task_name, budget_type, budget, seed, workload, algorithm
        ):
        self.dataset_name = HPOOOD_base.DATASETS[workload]
        self.algorithm_name = algorithm
        self.test_envs = [0,1]
        self.data_dir = '/home/cola/transopt_files/data/'
        self.output_dir = f'/home/cola/transopt_files/output/'
        self.holdout_fraction = 0.2
        self.validate_fraction = 0.1
        self.uda_holdout_fraction = 0.8
        self.task = 'domain_generalization'
        self.steps = 500
        self.checkpoint_freq = 50
        self.query = 0
        
        self.save_model_every_checkpoint = False
        
        self.skip_model_save = False
        
        self.trial_seed = seed
        
        self.model_save_dir = self.output_dir + f'models/{self.algorithm_name}_{self.dataset_name}_{seed}/'
        self.results_save_dir = self.output_dir + f'results/{self.algorithm_name}_{self.dataset_name}_{seed}/'
        
        print(f"Selected algorithm: {self.algorithm_name}, dataset: {self.dataset_name}")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)
        super(HPOOOD_base, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.hparams = default_hparams(self.algorithm_name, self.dataset_name)

        if self.dataset_name in vars(ooddatasets):
            self.dataset = vars(ooddatasets)[self.dataset_name](self.data_dir,
                self.test_envs, self.hparams)
        else:
            raise NotImplementedError
        
        in_splits = []
        val_splits = []
        out_splits = []
        uda_splits = []
        
        for env_i, env in enumerate(self.dataset):
            uda = []

            out, in_ = misc.split_dataset(env,
                int(len(env)*self.holdout_fraction),
                misc.seed_hash(self.seed, env_i))
            
            val, in_ = misc.split_dataset(in_,
                int(len(in_)*self.validate_fraction),
                misc.seed_hash(self.seed, env_i))

            if env_i in self.test_envs:
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*self.uda_holdout_fraction),
                    misc.seed_hash(self.trial_seed, env_i))

            if self.hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                val_weights = misc.make_weights_for_balanced_classes(val)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, val_weights, out_weights, uda_weights = None, None, None, None
            in_splits.append((in_, in_weights))
            val_splits.append((val, val_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))
            if self.task == "domain_adaptation" and len(uda_splits) == 0:
                raise ValueError("Not enough unlabeled samples for domain adaptation.")

        self.train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits) 
            if i not in self.test_envs]
        
        self.val_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(val_splits)
            if i not in self.test_envs]

        self.uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)]

        self.eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=self.dataset.N_WORKERS)
            for env, _ in (in_splits + val_splits + out_splits + uda_splits)]
    
     
        self.eval_weights = [None for _, weights in (in_splits + val_splits + out_splits + uda_splits)]
        self.eval_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
        self.eval_loader_names += ['env{}_val'.format(i)
            for i in range(len(val_splits))]
        self.eval_loader_names += ['env{}_out'.format(i)
            for i in range(len(out_splits))]
        self.eval_loader_names += ['env{}_uda'.format(i)
            for i in range(len(uda_splits))]
        
        self.train_minibatches_iterator = zip(*self.train_loaders)
        self.uda_minibatches_iterator = zip(*self.uda_loaders)
        self.checkpoint_vals = collections.defaultdict(lambda: [])

        self.steps_per_epoch = min([len(env)/self.hparams['batch_size'] for env,_ in in_splits])
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.trial_seed}")

        else:
            self.device = "cpu"
        



    def save_checkpoint(self, filename):
        if self.skip_model_save:
            return
        save_dict = {
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_num_domains": len(self.dataset) - len(self.test_envs),
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(self.model_save_dir, filename))

    def get_configuration_space(
        self, seed: Union[int, None] = None):
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
        variables=[Continuous('lr', [-8.0, 0.0]),
            Continuous('weight_decay', [-10.0, -5.0]),
            ]
        ss = SearchSpace(variables)
        self.hparam = ss
        return ss

    def get_fidelity_space(
        self, seed: Union[int, None] = None):
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

        # return fidel_space
        fs = FidelitySpace([])
        return fs
    def train(self, configuration: dict):
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.hparams = default_hparams(self.algorithm_name, self.dataset_name)
        self.hparams['lr'] = configuration["lr"]
        self.hparams['weight_decay'] = configuration["weight_decay"]
        self.steps = configuration['epoch']
        print(self.steps)

        n_steps = self.steps or self.dataset.N_STEPS
        
        last_results_keys = None
        
        start_step = 0
        
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            minibatches_device = [(x.to(self.device), y.to(self.device))
                for x,y in next(self.train_minibatches_iterator)]
            if self.task == "domain_adaptation":
                uda_device = [x.to(self.device)
                    for x,_ in next(self.uda_minibatches_iterator)]
            else:
                uda_device = None
            step_vals = self.algorithm.update(minibatches_device, uda_device)
            self.checkpoint_vals['step_time'].append(time.time() - step_start_time)

            for key, val in step_vals.items():
                self.checkpoint_vals[key].append(val)

            if (step % self.checkpoint_freq == 0) or (step == n_steps - 1):
                results = {
                    'step': step,
                    'epoch': step / self.steps_per_epoch,
                }

                for key, val in self.checkpoint_vals.items():
                    results[key] = np.mean(val)

                evals = zip(self.eval_loader_names, self.eval_loaders, self.eval_weights)
                for name, loader, weights in evals:
                    acc = misc.accuracy(self.algorithm, loader, weights, self.device)
                    results[name+'_acc'] = acc

                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                    colwidth=12)

                results.update({
                    'hparams': self.hparams,
                })

                start_step = step + 1

                if self.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{step}.pkl')
        
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')
        
        return results
    
    
    def get_score(self, configuration: dict):
        algorithm_class = algorithms.get_algorithm_class(self.algorithm_name)
        self.algorithm = algorithm_class(self.dataset.input_shape, self.dataset.num_classes,
            len(self.dataset) - len(self.test_envs), self.hparams)
        self.algorithm.to(self.device)
        
        self.query += 1
        results = self.train(configuration)
        
        epochs_path = os.path.join(self.results_save_dir, f"{self.query}_lr_{configuration['lr']}_weight_decay_{configuration['weight_decay']}.jsonl")
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")


        val_acc = [i[1] for i in results.items() if 'val' in i[0]]
        avg_val_acc = np.mean(val_acc)
        
        test_acc = [i[1] for i in results.items() if 'out' in i[0]]
        avg_test_acc = np.mean(test_acc)
        
        return avg_val_acc, avg_test_acc
        

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

            
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
        else:
            epoch = 500
            
        if fidelity is None:
            fidelity = {"epoch": epoch, "data_frac": 0.8}
        c = {
            "lr": np.exp2(configuration["lr"]),
            "weight_decay": np.exp2(configuration["weight_decay"]),
            "batch_size": 64,
            "epoch": fidelity["epoch"],
        }
        val_acc, test_acc = self.get_score(c)

        results = {list(self.objective_info.keys())[0]: float(1 - val_acc)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"
    
    
    
@problem_registry.register("ERMOOD")
class ERMOOD(HPOOOD_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(ERMOOD, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='ERM')

@problem_registry.register("IRMOOD")
class IRMOOD(HPOOOD_base):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(IRMOOD, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='IRM')

@problem_registry.register("ARMOOD")
class ARMOOD(HPOOOD_base):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(ARMOOD, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='ARM')

@problem_registry.register("MixupOOD")
class MixupOOD(HPOOOD_base):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(MixupOOD, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='Mixup')

@problem_registry.register("DANNOOD")
class DANNOOD(HPOOOD_base):
    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
        ):
        super(DANNOOD, self).__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed = seed, workload = workload, algorithm='DANN')
        




if __name__ == "__main__":
    p = MixupOOD(task_name='', budget_type='FEs', budget=100, seed = 0, workload = 2)
    configuration = {
        "lr": -0.3,
        "weight_decay": -5,
    }
    p.f(configuration=configuration)
    

