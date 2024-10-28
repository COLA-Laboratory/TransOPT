import numpy as np
from typing import Union, Dict, List
from sklearn.preprocessing import power_transform
from agent.registry import normalizer_registry,normalizer_register

def get_normalizer(name):
    """Create the optimizer object."""
    normalizer = normalizer_registry.get(name)


    if normalizer is not None:
        return normalizer
    else:
        # 处理任务名称不在注册表中的情况
        print(f"Normalizer '{name}' not found in the registry.")
        raise NameError


@normalizer_register('pt')
def normalize_with_power_transform(data: Union[np.ndarray, list], mean=None, std=None):
    """
    Normalize the data using mean and standard deviation, followed by power transformation.

    Parameters:
    - data (Union[np.ndarray, list]): Input data to be normalized.
    - mean (float, optional): Mean for normalization.
    - std (float, optional): Std for normalization.

    Returns:
    - Union[np.ndarray, list]: Normalized and power transformed data.
    """

    # Handle multiple data sets (list of ndarrays)
    if type(data) is list:
        all_include = data[0]
        data_len = [0, len(data[0])]
        for Y in data[1:]:
            all_include = np.concatenate((all_include, Y), axis=0)
            data_len.append(len(all_include))
    else:  # Single data set
        all_include = data
        data_len = [0, len(data)]

    # Calculate mean and std if not provided
    if mean is None:
        mean = np.mean(all_include)
    if std is None:
        std = np.std(all_include)

    # Normalize and power transform
    all_include = power_transform((all_include - mean) / std, method='yeo-johnson')

    # Split back into multiple data sets if originally provided as a list
    if type(data) is list:
        new_data = []
        for i in range(len(data_len) - 1):
            new_data.append(all_include[data_len[i]:data_len[i + 1]])
        return new_data

    # Return the transformed data
    return all_include


def rank_normalize_with_power_transform(data: Union[np.ndarray, list]):
    """
    This function first replaces the actual values of the data with their ranks.
    After that, it standardizes and then applies a power transform (yeo-johnson) on the data.

    Args:
    - data (Union[np.ndarray, list]): The input data, either as a single ndarray or as a list of ndarrays.
    - mean (float, optional): Mean value to use for standardization. If not provided, it's computed from data.
    - std (float, optional): Standard deviation value to use for standardization. If not provided, it's computed from data.

    Returns:
    - np.ndarray or list of np.ndarray: Transformed data.
    """

    # Single ndarray input
    if isinstance(data, np.ndarray):
        # Replace the values in data with their corresponding ranks
        sorted_indices = np.argsort(data, axis=0)[:, 0]
        rank_array = np.zeros(shape=data.shape[0])
        rank_array[sorted_indices] = np.arange(1, len(data) + 1)

        # Apply standardization followed by power transformation
        return power_transform(rank_array[:, np.newaxis], method='yeo-johnson')

    # List of ndarrays input
    elif isinstance(data, list):
        new_data = []
        all_include = data[0]
        data_len = [0, len(data[0])]

        # Combine all datasets in the list for subsequent processing
        for Y in data[1:]:
            all_include = np.concatenate((all_include, Y), axis=0)
            data_len.append(len(all_include))

        # Replace the values in combined data with their corresponding ranks
        sorted_indices = np.argsort(all_include, axis=0)[:, 0]
        rank_array = np.zeros(shape=all_include.shape[0])
        rank_array[sorted_indices] = np.arange(1, len(all_include) + 1)


        # Apply standardization followed by power transformation
        all_include = power_transform((rank_array[:, np.newaxis]), method='yeo-johnson')

        # Split the transformed data back into separate datasets based on the original list
        for i in range(len(data_len) - 1):
            new_data.append(all_include[data_len[i]:data_len[i + 1]])

        return new_data

    # Raise an error for unsupported input types
    raise ValueError('Unsupported input type for normalization and power transform.')


@normalizer_register('norm')
def normalize(data:Union[List, Dict, np.ndarray], mean=None, std=None):
    """
    Normalize the data using the given mean and standard deviation or compute them from the data if not provided.

    Parameters:
    - data (ndarray): The data to be normalized.
    - mean (float, optional): If provided, use this mean for normalization. Otherwise, compute from the data.
    - std (float, optional): If provided, use this standard deviation for normalization. Otherwise, compute from the data.

    Returns:
    - ndarray: Normalized data.
    """


    # Compute mean and std from data if not provided
    if isinstance(data, np.ndarray):
        if mean is None:
            mean = np.mean(data)
        if std is None:
            std = np.std(data)
        return (data - mean) / std
    elif isinstance(data, list):
        tmp = []
        for d in data:
            if mean is None:
                mean = np.mean(d)
            if std is None:
                std = np.std(d)
            tmp.append((d - mean) / std)
        return tmp
    else:
        raise TypeError("Input data must be a numpy array or a list of numpy arrays.")
