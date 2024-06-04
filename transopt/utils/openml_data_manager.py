"""
This file includes code adapted from HPOBench (https://github.com/automl/HPOBench),
which is licensed under the Apache License 2.0. A copy of the license can be
found at http://www.apache.org/licenses/LICENSE-2.0.
"""


""" OpenMLDataManager organizing the data for the benchmarks with data from
OpenML-tasks.

DataManager organizing the download of the data.
The load function of a DataManger downloads the data given an unique OpenML
identifier. It splits the data in train, test and optional validation splits.
It can be distinguished between holdout and cross-validation data sets.

For Non-OpenML data sets please use the hpobench.util.data_manager.
"""

import os
import abc
import logging
import tarfile
import requests
import openml
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union
from zipfile import ZipFile
from oslo_concurrency import lockutils
from sklearn.model_selection import train_test_split

from transopt.utils.rng_helper import get_rng


# TODO: 考虑使用 config 模块管理
def _check_dir(path: Path):
    """ Check whether dir exists and if not create it"""
    Path(path).mkdir(exist_ok=True, parents=True)

cache_dir = os.environ.get('OPENML_CACHE_HOME', '~/.cache/transopt')
data_dir = os.environ.get('OPENML_DATA_HOME', '~/.local/share/transopt')
cache_dir = Path(cache_dir).expanduser().absolute()
data_dir = Path(data_dir).expanduser().absolute()
_check_dir(cache_dir)
_check_dir(data_dir)


def get_openml100_taskids():
    """
    Return task ids for the OpenML100 data ets
    See also here: https://www.openml.org/s/14
    Reference: https://arxiv.org/abs/1708.03731
    """
    return [
        258, 259, 261, 262, 266, 267, 271, 273, 275, 279, 283, 288, 2120,
        2121, 2125, 336, 75093, 75092, 75095, 75097, 75099, 75103, 75107,
        75106, 75109, 75108, 75112, 75129, 75128, 75135, 146574, 146575,
        146572, 146573, 146578, 146579, 146576, 146577, 75154, 146582,
        146583, 75156, 146580, 75159, 146581, 146586, 146587, 146584,
        146585, 146590, 146591, 146588, 146589, 75169, 146594, 146595,
        146592, 146593, 146598, 146599, 146596, 146597, 146602, 146603,
        146600, 146601, 75181, 146604, 146605, 75215, 75217, 75219, 75221,
        75225, 75227, 75231, 75230, 75232, 75235, 3043, 75236, 75239, 3047,
        232, 233, 236, 3053, 3054, 3055, 241, 242, 244, 245, 246, 248, 250,
        251, 252, 253, 254,
    ]


def get_openmlcc18_taskids():
    """
    Return task ids for the OpenML-CC18 data sets
    See also here: https://www.openml.org/s/99
    TODO: ADD reference
    """
    return [167149, 167150, 167151, 167152, 167153, 167154, 167155, 167156, 167157,
            167158, 167159, 167160, 167161, 167162, 167163, 167165, 167166, 167167,
            167168, 167169, 167170, 167171, 167164, 167173, 167172, 167174, 167175,
            167176, 167177, 167178, 167179, 167180, 167181, 167182, 126025, 167195,
            167194, 167190, 167191, 167192, 167193, 167187, 167188, 126026, 167189,
            167185, 167186, 167183, 167184, 167196, 167198, 126029, 167197, 126030,
            167199, 126031, 167201, 167205, 189904, 167106, 167105, 189905, 189906,
            189907, 189908, 189909, 167083, 167203, 167204, 189910, 167202, 167097,
            ]


def _load_data(task_id: int):
    """ Helper-function to load the data from the OpenML website. """
    task = openml.tasks.get_task(task_id)

    try:
        # This should throw an ValueError!
        task.get_train_test_split_indices(fold=0, repeat=1)
        raise AssertionError(f'Task {task_id} has more than one repeat. This '
                             f'benchmark can only work with a single repeat.')
    except ValueError:
        pass

    try:
        # This should throw an ValueError!
        task.get_train_test_split_indices(fold=1, repeat=0)
        raise AssertionError(f'Task {task_id} has more than one fold. This '
                             f'benchmark can only work with a single fold.')
    except ValueError:
        pass

    train_indices, test_indices = task.get_train_test_split_indices()

    X, y = task.get_X_and_y()

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # TODO replace by more efficient function which only reads in the data
    # saved in the arff file describing the attributes/features
    dataset = task.get_dataset()
    _, _, categorical_indicator, _ = dataset.get_data(target=task.target_name)
    variable_types = ['categorical' if ci else 'numerical' for ci in categorical_indicator]

    return X_train, y_train, X_test, y_test, variable_types, dataset.name

class DataManager(abc.ABC, metaclass=abc.ABCMeta):
    """ Base Class for loading and managing the data.

    Attributes
    ----------
    logger : logging.Logger

    """

    def __init__(self):
        self.logger = logging.getLogger("DataManager")

    @abc.abstractmethod
    def load(self):
        """ Loads data from data directory as defined in
        config_file.data_directory
        """
        raise NotImplementedError()

    def create_save_directory(self, save_dir: Path):
        """ Helper function. Check if data directory exists. If not, create it.

        Parameters
        ----------
        save_dir : Path
            Path to the directory. where the data should be stored
        """
        if not save_dir.is_dir():
            self.logger.debug(f'Create directory {save_dir}')
            save_dir.mkdir(parents=True, exist_ok=True)

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{cache_dir}/lock_download_file', delay=0.5)
    def _download_file_with_progressbar(self, data_url: str, data_file: Path):
        data_file = Path(data_file)

        if data_file.exists():
            self.logger.info('Data File already exists. Skip downloading.')
            return

        self.logger.info(f"Download the file from {data_url} to {data_file}")
        data_file.parent.mkdir(parents=True, exist_ok=True)

        from tqdm import tqdm
        r = requests.get(data_url, stream=True)
        with open(data_file, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              unit_divisor=1024, unit='kB', total=int(total_length / 1024) + 1):
                if chunk:
                    _ = f.write(chunk)
                    f.flush()
        self.logger.info(f"Finished downloading to {data_file}")

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{cache_dir}/lock_unzip_file', delay=0.5)
    def _untar_data(self, compressed_file: Path, save_dir: Union[Path, None] = None):
        self.logger.debug('Extract the compressed data')
        with tarfile.open(compressed_file, 'r') as fh:
            if save_dir is None:
                save_dir = compressed_file.parent
            fh.extractall(save_dir)
        self.logger.debug(f'Successfully extracted the data to {save_dir}')

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{cache_dir}/lock_unzip_file', delay=0.5)
    def _unzip_data(self, compressed_file: Path, save_dir: Union[Path, None] = None):
        self.logger.debug('Extract the compressed data')
        with ZipFile(compressed_file, 'r') as fh:
            if save_dir is None:
                save_dir = compressed_file.parent
            fh.extractall(save_dir)
        self.logger.debug(f'Successfully extracted the data to {save_dir}')

class HoldoutDataManager(DataManager):
    """  Base Class for loading and managing the Holdout data sets.

    Attributes
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_valid : np.ndarray
    y_valid : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """

    def __init__(self):
        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
    
    
class CrossvalidationDataManager(DataManager):
    """
    Base Class for loading and managing the cross-validation data sets.

    Attributes
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """

    def __init__(self):
        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


class OpenMLHoldoutDataManager(HoldoutDataManager):
    """ Base class for loading holdout data set from OpenML.

    Attributes
    ----------
    task_id : int
    rng : np.random.RandomState
    name : str
    variable_types : list
        Indicating the type of each feature in the loaded data
        (e.g. categorical, numerical)

    Parameters
    ----------
    openml_task_id : int
        Unique identifier for the task on OpenML
    rng : int, np.random.RandomState, None
        defines the random state
    """

    def __init__(self, openml_task_id: int, rng: Union[int, np.random.RandomState, None] = None):
        super(OpenMLHoldoutDataManager, self).__init__()

        self._save_to = data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_root_cache_directory(str(self._save_to))

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads dataset from OpenML in config_file.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """

        self.X_train, self.y_train, self.X_test, self.y_test, self.variable_types, self.name = _load_data(self.task_id)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train,
                                                                                  self.y_train,
                                                                                  test_size=0.33,
                                                                                  stratify=self.y_train,
                                                                                  random_state=self.rng)

        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    @staticmethod
    def replace_nans_in_cat_columns(X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray,
                                    is_categorical: Union[np.ndarray, List]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """ Helper function to replace nan values in categorical features / columns by a non-used value.
        Here: Min - 1.
        """
        _cat_data = np.concatenate([X_train, X_valid, X_test], axis=0)
        nan_index = np.isnan(_cat_data[:, is_categorical])
        categories = [np.unique(_cat_data[:, i][~nan_index[:, i]])
                      for i in range(X_train.shape[1]) if is_categorical[i]]
        replace_nans_with = np.nanmin(_cat_data[:, is_categorical], axis=0) - 1

        categories = [np.concatenate([replace_value.flatten(), cat])
                      for (replace_value, cat) in zip(replace_nans_with, categories)]

        def _find_and_replace(array, replace_nans_with):
            nan_idx = np.where(np.isnan(array))
            array[nan_idx] = np.take(replace_nans_with, nan_idx[1])
            return array

        X_train[:, is_categorical] = _find_and_replace(X_train[:, is_categorical], replace_nans_with)
        X_valid[:, is_categorical] = _find_and_replace(X_valid[:, is_categorical], replace_nans_with)
        X_test[:, is_categorical] = _find_and_replace(X_test[:, is_categorical], replace_nans_with)
        return X_train, X_valid, X_test, categories


class OpenMLCrossvalidationDataManager(CrossvalidationDataManager):
    """ Base class for loading cross-validation data set from OpenML.

    Attributes
    ----------
    task_id : int
    rng : np.random.RandomState
    name : str
    variable_types : list
        Indicating the type of each feature in the loaded data
        (e.g. categorical, numerical)

    Parameters
    ----------
    openml_task_id : int
        Unique identifier for the task on OpenML
    rng : int, np.random.RandomState, None
        defines the random state
    """

    def __init__(self, openml_task_id: int, rng: Union[int, np.random.RandomState, None] = None):
        super(OpenMLCrossvalidationDataManager, self).__init__()

        self._save_to = data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(str(self._save_to))

    def load(self):
        """
        Loads dataset from OpenML in config_file.data_directory.
        Downloads data if necessary.
        """

        X_train, y_train, X_test, y_test, variable_types, name = \
            _load_data(self.task_id)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.variable_types = variable_types
        self.name = name

        return self.X_train, self.y_train, self.X_test, self.y_test
