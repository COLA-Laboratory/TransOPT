import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple, List


@dataclass
class InputData:
    X: np.ndarray


@dataclass
class TaskData:
    X: np.ndarray
    Y: np.ndarray



# def vectors_to_ndarray(keys_order, X: List[Dict]) -> np.ndarray:
#     """Convert a list of input_vectors to a ndarray."""
#     # Converting dictionaries to lists using the order from keys_order
#     data = [[vec[key] for key in keys_order] for vec in X]

#     # Converting lists to ndarray
#     ndarray = np.array(data)

#     return ndarray

# def ndarray_to_vectors(keys_order, ndarray: np.ndarray) -> List[Dict]:
#     """Convert a ndarray to a list of dictionaries."""
#     # Converting ndarray to lists of values
#     data = ndarray.tolist()

#     # Converting lists of values to dictionaries using keys from keys_order
#     input_vectors = [{key: value for key, value in zip(keys_order, row)} for row in data]

#     return input_vectors

def output_to_ndarray(Y: List[Dict]) -> np.ndarray:
    """Extract function_value from each output and convert to ndarray."""
    # Extracting function_value from each dictionary in the list
    function_values = [[y for name, y in item.items()] for item in Y]

    # Converting list to ndarray
    ndarray = np.array(function_values)

    return ndarray

def multioutput_to_ndarray(output_value: List[Dict], num_output:int) -> np.ndarray:
    """Extract function_value from each output and convert to ndarray."""
    # Extracting function_value from each dictionary in the list
    function_values = []
    for i in range(1, num_output+1):
        function_values.append([item[f'function_value_{i}'] for item in output_value])

    # Converting list to ndarray
    ndarray = np.array(function_values)

    return ndarray


def convert_np_to_bulidin(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_to_bulidin(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_bulidin(item) for item in obj]
    else:
        return obj