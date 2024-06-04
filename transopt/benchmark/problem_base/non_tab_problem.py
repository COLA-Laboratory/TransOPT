""" Base-class of configuration optimization benchmarks """
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from transopt.benchmark.problem_base.base import ProblemBase

logger = logging.getLogger("NonTabularProblem")


import abc


class NonTabularProblem(ProblemBase):
    def __init__(
        self,
        task_name: str,
        budget_type,
        budget: int,
        workload,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.budget = budget
        self.workload = workload
        self.lock_flag = False
        self.budget_type = budget_type

        super(NonTabularProblem, self).__init__(seed, **kwargs)

    def get_budget_type(self) -> str:
        """Provides the budget type about the benchmark.

        Returns
        -------
        str
            some human-readable information

        """
        return self.budget_type

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
        return self.problem_type

    def get_input_dim(self) -> int:
        """Provides the input dimension about the benchmark.

        Returns
        -------
        int
            some human-readable information

        """
        return self.num_variables

    def get_objective_num(self) -> int:
        return self.num_objectives

    def lock(self):
        self.lock_flag = True

    def unlock(self):
        self.lock_flag = False

    def get_lock_state(self) -> bool:
        return self.lock_flag
    
    @property
    @abc.abstractmethod
    def workloads(self):
        raise NotImplementedError()
    
    @property
    @abc.abstractmethod
    def fidelity(self):
        raise NotImplementedError()