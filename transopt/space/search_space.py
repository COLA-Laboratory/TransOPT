import copy

import numpy as np
import pandas as pd


class SearchSpace:
    def __init__(self, variables):
        self._variables = {var.name: var for var in variables}
        self.variables_order = [var.name for var in variables]

        # 计算并存储原始范围和类型信息
        self.original_ranges = {
            name: var.search_space_range for name, var in self._variables.items()
        }
        self.var_discrete = {
            name: var.is_discrete for name, var in self._variables.items()
        }

        self.ranges = copy.deepcopy(self.original_ranges)
    
    
    def __getitem__(self, item):
        return self._variables.get(item)


    def __contains__(self, item):
        return item in self.variables_order

    def get_design_variables(self):
        return self._variables
    
    def get_design_variable(self, name):
        return self._variables[name]

    def get_hyperparameter_names(self):
        return list(self._variables.keys())
    
    def get_hyperparameter_types(self):
        return {name:self._variables[name].type for name in self._variables}
    
    
    def map_to_design_space(self, values: np.ndarray) -> dict:
        """
        Maps the given values from the search space to the design space.

        Args:
            values (np.ndarray): The values to be mapped from the search space. Must be a 1D NumPy array.

        Returns:
            dict: A dictionary containing the mapped values in the design space.

        Raises:
            ValueError: If the `values` parameter is not a 1D NumPy array.
        """

        values_dict = {}
        for i, name in enumerate(self.variables_order):
            variable = self._variables[name]
            value = values[i]
            values_dict[name] = variable.map2design(value)
        return values_dict
    
    def map_from_design_space(self, values_dict: dict) -> np.ndarray:
        """
        Maps values from the design space to the search space.

        Args:
            values_dict (dict): A dictionary containing variable names as keys and their corresponding values.

        Returns:
            np.ndarray: An array of mapped values in the search space.
        """
        values_array = np.zeros(len(self.variables_order))
        for i, name in enumerate(self.variables_order):
            variable = self._variables[name]
            value = values_dict[name]
            values_array[i] = variable.map2search(value)
        return values_array

    def update_range(self, name, new_range: tuple):
        """
        Update the range of a variable in the search space.

        Args:
            name (str): The name of the variable.
            new_range (tuple): The new range for the variable.

        Raises:
            ValueError: If the variable is not found in the search space or if the new range is out of the original range.
        """
        if name in self._variables:
            # Check if the new range is valid
            ori_range = self.original_ranges[name]
            if new_range[0] < ori_range[0] or new_range[1] > ori_range[1]:
                raise ValueError(
                    f"New range {new_range} is out of the original range {ori_range}."
                )
                
            self.ranges[name] = new_range
        else:
            raise ValueError(f"Variable '{name}' not found in search space.")


