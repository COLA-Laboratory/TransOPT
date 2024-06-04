import time
import logging
import ConfigSpace
import numpy as np
import ConfigSpace as CS
from typing import Union, Dict, Tuple
from pathlib import Path
from csstuning.dbms.dbms_benchmark import MySQLBenchmark

from transopt.benchmark.problem_base import NonTabularProblem
from agent.registry import benchmark_register

ERROR_VALUE = 1e10

@benchmark_register("DBMS")
class DBMSTuning(NonTabularProblem):
    def __init__(
        self, task_name, budget, seed, task_id, task_type="non-tabular", workload=None, **kwargs
    ):
        self.workload = workload if workload is not None else MySQLBenchmark.AVAILABLE_WORKLOADS[0]
        self.benchmark = MySQLBenchmark(workload=self.workload)
        self.config_knob = self.benchmark.get_config_space()
        
        super().__init__(task_name=task_name, workload=self.workload, seed=seed, task_type=task_type, budget=budget)
        np.random.seed(seed)

    def objective_function(
        self,
        configuration: Union[ConfigSpace.Configuration, Dict],
        fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        c = {}
        for k, v in configuration.items():
            assert k in self.config_knob
            if self.config_knob[k]["type"] == "enum":
                c[k] = self.config_knob[k]["range"][v]
                configuration[k] = int(configuration[k])
            else:
                if k in self.min_value:
                    c[k] = int(np.floor(np.exp2(v) + self.min_value[k]))
                    configuration[k] = int(configuration[k])

        try:
            start_time = time.time()
            performance = self.benchmark.run(c)
            end_time = time.time()
            return {
                "function_value_1": -float(performance["throughput"]),
                "function_value_2": float(performance["latency"] * 10e-3),
                "latency": float(performance["latency"] * 10e-3),
                "throughput": -float(performance["throughput"]),
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }
            
        except:
            end_time = time.time()
            return {
                "function_value_1": ERROR_VALUE,
                "function_value_2": ERROR_VALUE,
                "latency": ERROR_VALUE,
                "throughput": ERROR_VALUE,
                "cost": float(end_time - start_time),
                "info": {"fidelity": fidelity},
            }
            
            
            

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)

        config = self.benchmark.get_config_space()
        hyperparameters = []
        expflag = {}
        self.min_value = {}
        for key, value in config.items():
            if value["type"] == "enum":
                hyperparameters.append(
                    CS.CategoricalHyperparameter(
                        key, choices=value["range"], default_value=str(value["default"])
                    )
                )
            elif value["type"] == "integer":
                if value["range"][1] >= 1024 or "size" in key:
                    self.min_value[key] = value["range"][0] + 1
                    lower = 0
                    upper = np.floor(np.log2(value["range"][1] - value["range"][0]))
                    expflag[key] = True
                    hyperparameters.append(
                        CS.UniformFloatHyperparameter(
                            key,
                            lower=lower,
                            upper=upper,
                            default_value=np.floor(
                                np.log2(value["default"] - value["range"][0] + 1)
                            ),
                        )
                    )
                else:
                    hyperparameters.append(
                        CS.UniformIntegerHyperparameter(
                            key,
                            lower=value["range"][0],
                            upper=value["range"][1],
                            default_value=value["default"],
                        )
                    )
            else:
                pass

        cs.add_hyperparameters(hyperparameters)

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> ConfigSpace.ConfigurationSpace:
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        return fidel_space

    def get_meta_information(self) -> Dict:
        return {"number_objective": 2}


if __name__ == "__main__":
    a = DBMSTuning("1", 121, 0, 1)
