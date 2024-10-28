import abc
import logging
import numpy as np
from typing import Union, Dict, List

from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.benchmark.problem_base.tab_problem import TabularProblem
from transopt.remote import ExperimentClient
from transopt.space.search_space import SearchSpace
logger = logging.getLogger("TransferProblem")


class TransferProblem:
    def __init__(self, seed: Union[int, np.random.RandomState, None] = None, **kwargs):
        self.seed = seed
        self.tasks = []
        self.time = []
        self.query_nums = []
        self.__id = 0

    def add_task_to_id(
        self,
        insert_id: int,
        task: Union[
            NonTabularProblem,
            TabularProblem,
        ],
    ):
        num_tasks = len(self.tasks)
        assert insert_id < num_tasks + 1

        self.tasks.insert(insert_id, task)
        self.query_nums.insert(insert_id, 0)

    def add_task(
        self,
        task: Union[
            NonTabularProblem,
            TabularProblem,
        ],
    ):
        num_tasks = len(self.tasks)
        insert_id = num_tasks
        self.add_task_to_id(insert_id, task)

    def del_task_by_id(self, del_id, name):
        pass

    def get_cur_id(self):
        return self.__id

    def get_tasks_num(self):
        return len(self.tasks)

    def get_unsolved_num(self):
        return len(self.tasks) - self.__id

    def get_rest_budget(self):
        return self.get_cur_budget() - self.get_query_num()

    def get_query_num(self):
        return self.query_nums[self.__id]

    def get_cur_budgettype(self):
        return self.tasks[self.__id].get_budget_type()

    def get_cur_budget(self):
        return self.tasks[self.__id].get_budget()

    def get_curname(self):
        return self.tasks[self.__id].get_name()

    def get_curdim(self):
        return self.tasks[self.__id].get_input_dim()

    def get_curobj_info(self):
        return self.tasks[self.__id].get_objectives()
    
    def get_cur_fidelity_info(self) -> Dict:
        return self.tasks[self.__id].fidelity_space.get_fidelity_range()

    def get_cur_searchspace_info(self) -> Dict:
        return self.tasks[self.__id].configuration_space.get_design_variables()
    
    
    def get_cur_searchspace(self) -> SearchSpace:
        return self.tasks[self.__id].configuration_space
    

    def get_curtask(self):
        return self.tasks[self.__id]
    
    
    def get_cur_seed(self):
        return self.tasks[self.__id].seed

    def get_cur_task_id(self):
        return self.tasks[self.__id].task_id

    def get_cur_workload(self):
        return self.tasks[self.__id].workload


    def sync_query_num(self, query_num: int):
        self.query_nums[self.__id] = query_num

    def roll(self):
        self.__id += 1

    def lock(self):
        self.tasks[self.__id].lock()

    def unlock(self):
        self.tasks[self.__id].unlock()

    def get_lockstate(self):
        return self.tasks[self.__id].get_lock_state()

    def get_task_type(self):
        if isinstance(self.tasks[self.__id], TabularProblem):
            return "tabular"
        elif isinstance(self.tasks[self.__id], NonTabularProblem):
            return "non-tabular"
        else:
            logger.error("Unknown task type.")
            raise NameError


    ###Methods only for tabular data###
    def get_dataset_size(self):
        assert isinstance(self.tasks[self.__id], TabularProblem)
        return self.tasks[self.__id].get_dataset_size()

    def get_var_by_idx(self, idx):
        assert isinstance(self.tasks[self.__id], TabularProblem)
        return self.tasks[self.__id].get_var_by_idx(idx)

    def get_idx_by_var(self, vectors):
        assert isinstance(self.tasks[self.__id], TabularProblem)
        return self.tasks[self.__id].get_idx_by_var(vectors)

    def get_unobserved_vars(self):
        assert isinstance(self.tasks[self.__id], TabularProblem)
        return self.tasks[self.__id].get_unobserved_vars()

    def get_unobserved_idxs(self):
        assert isinstance(self.tasks[self.__id], TabularProblem)
        return self.tasks[self.__id].get_unobserved_idxs()

    def add_query_num(self):
        if self.get_lockstate() == False:
            self.query_nums[self.__id] += 1

    def f(
        self,
        configuration: Union[
            Dict,
            List[Dict],
        ],
        fidelity: Union[
            Dict,
            None,
            List[Dict],
        ] = None,
        **kwargs,
    ):
        if isinstance(configuration, list):
            try:
                if (
                    self.get_query_num() + len(configuration) > self.get_cur_budget()
                    and self.get_lockstate() == False
                ):
                    logger.error(
                        " The current function evaluation has exceeded the user-set budget."
                    )
                    raise RuntimeError("The current function evaluation has exceeded the user-set budget.")
            except RuntimeError as e:
                return None

            if isinstance(fidelity, list):
                assert len(fidelity) == len(configuration)
            elif fidelity is None:
                fidelity = [None] * len(configuration)
            else:
                pass

            results = []
            for c_id, config in enumerate(configuration):
                result = self.tasks[self.__id].f(config, fidelity[c_id])
                self.add_query_num()

                results.append(result)
            return results
        else:
            if (
                self.get_query_num() >= self.get_cur_budget()
                and self.get_lockstate() == False
            ):
                logger.error(
                    " The current function evaluation has exceeded the user-set budget."
                )
                raise EnvironmentError

            result = self.tasks[self.__id].f(configuration, fidelity)
            self.add_query_num()
            return result

            # raise TypeError(f"Unrecognized task type.")


class RemoteTransferOptBenchmark(TransferProblem):
    def __init__(
        self, server_url, seed: Union[int, np.random.RandomState, None] = None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.client = ExperimentClient(server_url)
        self.task_params_list = []

    def add_task_to_id(
        self,
        insert_id: int,
        task: NonTabularProblem | TabularProblem,
        task_params,
    ):
        assert insert_id < len(self.tasks) + 1

        self.task_params_list.insert(insert_id, task_params)
        self.tasks.insert(insert_id, task)
        self.query_nums.insert(insert_id, 0)

    def f(
        self,
        configuration: Union[
            Dict,
            List[Union[Dict]],
        ],
        fidelity: Union[
            Dict,
            None,
            List[Union[Dict]],
        ] = None,
        idx: Union[int, None, List[int]] = None,
        **kwargs,
    ):
        space = self.get_cur_searchspace()
        bench_name = self.get_curname().split("_")[0]
        bench_params = self.task_params_list[self.get_curid()]

        if not space or not bench_name or not bench_params:
            raise ValueError("Missing or incorrect data for benchmark.")

        # Package data
        data = self._package_data(
            space, bench_name, bench_params, configuration, fidelity, idx, **kwargs
        )

        result = self._execute_experiment(data)

        return result

    def _package_data(
        self, space, bench_name, bench_params, configuration, fidelity, idx, **kwargs
    ):
        return {
            "benchmark": bench_name,
            "id": space["task_id"],
            "budget": space["budget"],
            "seed": space["seed"],
            "bench_params": bench_params,
            "fitness_params": {
                "configuration": configuration,
                "fidelity": fidelity,
                "idx": idx,
                **kwargs,
            },
        }

    def _execute_experiment(self, data):
        # Send data to server and get the result
        task_id = self.client.start_experiment(data)

        # Wait for the task to complete and get the result
        return self.client.wait_for_result(task_id)
