import logging
import os
from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pds

from transopt.benchmark.problem_base.base import ProblemBase
from transopt.utils.encoding import multitarget_encoding, target_encoding
from transopt.utils.Read import read_file

logger = logging.getLogger("TabularProblem")



class TabularProblem(ProblemBase):
    def __init__(
            self,
            task_name: str,
            task_type: str,
            budget: int,
            workload,
            path: str = None,
            seed: Union[int, np.random.RandomState, None] = None,
            space_info: Dict = None,
            **kwargs,
    ):

        super(TabularProblem, self).__init__(task_name= task_name, task_type=task_type, budget=budget,workload=workload, seed=seed, **kwargs)
        self.path = path

        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            return "URL"
        # If the string is a valid file path
        elif os.path.exists(path) or os.path.isabs(path):
            dir_path = Path(path)
            workload_path = dir_path / workload
            data = read_file(workload_path)
            unnamed_columns = [col for col in data.columns if "Unnamed" in col]
            # delete the unnamed column
            data.drop(unnamed_columns, axis=1, inplace=True)

            para_names = [value for value in data.columns]
            if space_info is None or not isinstance(space_info, dict):
                self.space_info = {}
            else:
                self.space_info = space_info

            if 'input_dim' not in self.space_info and 'num_objective' not in self.space_info:
                self.space_info['input_dim'] = len(para_names) - 1

                self.space_info['num_objective'] = len(para_names) - self.space_info['input_dim']
            elif 'input_dim' in self.space_info and 'num_objective' in self.space_info:
                pass
            else:
                if 'num_objective' in self.space_info:
                    self.space_info['input_dim'] = len(para_names) - self.space_info['num_objective']

                if 'input_dim' in self.space_info:
                        self.space_info['num_objective'] = len(para_names) - self.space_info['input_dim']

            self.input_dim = self.space_info['input_dim']
            self.num_objective = self.space_info['num_objective']
            self.encodings = {}
            for i in range(self.num_objective):
                data[f"function_value_{i+1}"] = data[para_names[self.input_dim+i]]

            if 'variables' not in self.space_info:
                self.space_info['variables'] = {}
                for i in range(self.space_info['input_dim']):
                    var_name = para_names[i]
                    max_value = data[var_name].max()
                    min_value = data[var_name].min()
                    contains_decimal = False
                    contains_str = False
                    if data[var_name][1:].nunique() > 10:
                        for item in data[var_name][1:]:
                            if isinstance(item, str):
                                contains_str = True
                            if int(item) - item != 0:
                                contains_decimal = True
                                break  # 如果找到小数，无需继续检查
                        if contains_decimal:
                            var_type =  'continuous'
                            self.space_info['variables'][var_name] = {'bounds': [min_value, max_value],
                                                                      'type': var_type}
                        elif contains_str:
                            var_type = 'categorical'
                            data[var_name] = data[var_name].astype(str)

                            self.space_info['variables'][var_name] = {'bounds': [0, len(data[var_name][1:].unique()) - 1] ,
                                                                      'type': var_type}
                            if self.num_objective > 1:
                                self.cat_mapping = multitarget_encoding(data, var_name, [f'function_value_{i+1}' for i in range(self.num_objective)])
                            else:
                                self.cat_mapping = target_encoding(data, var_name, 'function_value_1')

                        else:
                            var_type = 'integer'
                            data[var_name] = data[var_name].astype(int)
                            self.space_info['variables'][var_name] = {'bounds': [min_value, max_value],
                                                                      'type': var_type}
                    else:
                        var_type = 'categorical'
                        data[var_name] = data[var_name].astype(str)


                        if self.num_objective > 1:
                            self.cat_mapping = multitarget_encoding(data, var_name, [f'function_value_{i + 1}' for i in
                                                                           range(self.num_objective)])
                        else:
                            self.cat_mapping = target_encoding(data, var_name, 'function_value_1')
                        max_key = max(self.cat_mapping.keys())

                        # 找出最小的键
                        min_key = min(self.cat_mapping.keys())
                        self.space_info['variables'][var_name] = {'bounds': [min_key, max_key],
                                                                  'type': var_type}


            data['config'] = data.apply(lambda row: row[:self.input_dim].tolist(), axis=1)
            data["config_s"] = data["config"].astype(str)
        else:
            raise ValueError("Unknown path type, only accept url or file path")

        
        self.var_range = self.get_configuration_bound()
        self.var_type = self.get_configuration_type()
        self.unqueried_data = data
        self.queried_data = pds.DataFrame(columns=data.columns)

    def f(
            self,
            configuration: Union[Dict, None],
            fidelity: Union[Dict, None] = None,
            **kwargs,
    ) -> Dict:

        results = self.objective_function(
            configuration=configuration, fidelity=fidelity, seed=self.seed
        )

        return results

    def objective_function(
            self,
            configuration: Union[ Dict],
            fidelity: Union[Dict, None] = None,
            seed: Union[np.random.RandomState, int, None] = None,
            **kwargs,
    ) -> Dict:
        c = {}
        for k in configuration.keys():
            if self.space_info['variables'][k]['type'] == 'categorical':
                c[k] = self.cat_mapping[configuration[k]]
            else:
                c[k] = configuration[k]

        X = str([configuration[k] for idx, k in enumerate(configuration.keys())])
        data = self.unqueried_data[self.unqueried_data['config_s'] == X]

        if not data.empty:
            self.unqueried_data.drop(data.index, inplace=True)
            self.queried_data = pds.concat([self.queried_data, data], ignore_index=True)
        else:
            raise ValueError(f"Configuration {X} not exist in oracle")

        res = {}
        for i in range(self.num_objective):
            res[f"function_value_{i+1}"] = float(data['fitness'])
        res["info"] = {"fidelity": fidelity}
        return res

    def sample_dataframe(key, df, p_remove=0.):
        """Randomly sample dataframe by the removal percentage."""
        if p_remove < 0 or p_remove >= 1:
            raise ValueError(
                f'p_remove={p_remove} but p_remove must be <1 and >= 0.')
        if p_remove > 0:
            n_remain = (1 - p_remove) * len(df)
            n_remain = int(np.ceil(n_remain))
            df = df.sample(n=n_remain, replace=False, random_state=key[0])
        return df


    # def get_configuration_bound(self):
    #     configuration_bound = {}
    #     for k, v in self.configuration_space.items():
    #         if type(v) is ConfigSpace.CategoricalHyperparameter:
    #             configuration_bound[k] = [0, len(v.choices) - 1]
    #         else:
    #             configuration_bound[k] = [v.lower, v.upper]

    #     return configuration_bound

    def get_configuration_type(self):
        configuration_type = {}
        for k, v in self.configuration_space.items():
            configuration_type[k] = type(v).__name__
        return configuration_type

    def get_configuration_space(
            self, seed: Union[int, None] = None
    ) :
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
        seed = seed if seed is not None else np.random.randint(1, 100000)
        # cs = CS.ConfigurationSpace(seed=seed)
        # variables = []

        # for k,v in self.space_info['variables'].items():
        #     lower = v['bounds'][0]
        #     upper = v['bounds'][1]
        #     if 'continuous' == v['type']:
        #         variables.append(CS.UniformFloatHyperparameter(k, lower=lower, upper=upper))
        #     elif 'integer' == v['type']:
        #         variables.append(CS.UniformIntegerHyperparameter(k, lower=lower, upper=upper))
        #     elif 'categorical' == v['type']:
        #         variables.append(CS.UniformIntegerHyperparameter(k, lower=lower, upper=upper))
        #     else:
        #         raise ValueError('Unknown variable type')

        # cs.add_hyperparameters(variables)
        # return cs

    def get_fidelity_space(
            self, seed: Union[int, None] = None
    ):
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
        seed = seed if seed is not None else np.random.randint(1, 100000)
        # fidel_space = CS.ConfigurationSpace(seed=seed)

        # return fidel_space

    def get_meta_information(self) -> Dict:
        return {}

    def get_budget(self) -> int:
        """Provides the function evaluations number about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.budget

    def get_name(self) -> str:
        """Provides the task name about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_name

    def get_type(self) -> str:
        """Provides the task type about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.task_type

    def get_input_dim(self) -> int:
        """Provides the input dimension about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.input_dim

    def get_objective_num(self) -> int:
        return self.num_objective

    def lock(self):
        self.lock_flag = True

    def unlock(self):
        self.lock_flag = False

    def get_lock_state(self) -> bool:
        return self.lock_flag


    def get_dataset_size(self):
        raise NotImplementedError

    def get_var_by_idx(self, idx):
        raise NotImplementedError

    def get_idx_by_var(self, vectors):
        raise NotImplementedError

    def get_unobserved_vars(self):
        raise NotImplementedError

    def get_unobserved_idxs(self):
        raise NotImplementedError
