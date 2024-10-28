import logging
import time
import numpy as np
from scipy import sparse
from typing import Union, Tuple, Dict, List
from sklearn import pipeline
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from transopt.utils.openml_data_manager import OpenMLHoldoutDataManager
from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

logger = logging.getLogger('SVMBenchmark')


@problem_registry.register('SVM')
class SupportVectorMachine(NonTabularProblem):
    """
    Hyperparameter optimization task to optimize the regularization
    parameter C and the kernel parameter gamma of a support vector machine.
    Both hyperparameters are optimized on a log scale in [-10, 10].
    The X_test data set is only used for a final offline evaluation of
    a configuration. For that the validation and training data is
    concatenated to form the whole training data set.
    """
    task_lists = [167149, 167152, 167183, 126025, 126029, 167161, 167169,
                  167178, 167176, 167177]
    problem_type = 'hpo'
    num_variables = 2
    num_objectives = 1
    workloads = []
    fidelity = None

    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        """
        Parameters
        ----------
        task_id : int, None
        rng : np.random.RandomState, int, None
        """
        super(SupportVectorMachine, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        task_type='non-tabular'
        self.task_id = SupportVectorMachine.task_lists[workload]
        self.cache_size = 200  # Cache for the SVC in MB
        self.accuracy_scorer = make_scorer(accuracy_score)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, variable_types = \
            self.get_data()
        self.categorical_data = np.array([var_type == 'categorical' for var_type in variable_types])

        # Sort data (Categorical + numerical) so that categorical and continous are not mixed.
        categorical_idx = np.argwhere(self.categorical_data)
        continuous_idx = np.argwhere(~self.categorical_data)
        sorting = np.concatenate([categorical_idx, continuous_idx]).squeeze()
        self.categorical_data = self.categorical_data[sorting]
        self.x_train = self.x_train[:, sorting]
        self.x_valid = self.x_valid[:, sorting]
        self.x_test = self.x_test[:, sorting]

        nan_columns = np.all(np.isnan(self.x_train), axis=0)
        self.categorical_data = self.categorical_data[~nan_columns]
        self.x_train, self.x_valid, self.x_test, self.categories = \
            OpenMLHoldoutDataManager.replace_nans_in_cat_columns(self.x_train, self.x_valid, self.x_test,
                                                                 is_categorical=self.categorical_data)

        self.train_idx = np.random.choice(a=np.arange(len(self.x_train)),
                                         size=len(self.x_train),
                                         replace=False)

        # Similar to [Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets]
        # (https://arxiv.org/pdf/1605.07079.pdf),
        # use 10 time the number of classes as lower bound for the dataset fraction
        n_classes = np.unique(self.y_train).shape[0]
        self.lower_bound_train_size = (10 * n_classes) / self.x_train.shape[0]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """ Loads the data given a task or another source. """

        assert self.task_id is not None, NotImplementedError('No task-id given. Please either specify a task-id or '
                                                             'overwrite the get_data Method.')

        data_manager = OpenMLHoldoutDataManager(openml_task_id=self.task_id, rng=self.seed)
        x_train, y_train, x_val, y_val, x_test, y_test = data_manager.load()

        return x_train, y_train, x_val, y_val, x_test, y_test, data_manager.variable_types

    def shuffle_data(self, seed=None):
        """ Reshuffle the training data. If 'rng' is None, the training idx are shuffled according to the
        class-random-state"""
        random_state = seed
        random_state.shuffle(self.train_idx)

    # pylint: disable=arguments-differ
    def objective_function(
        self,
        configuration: Dict,
        fidelity: Dict = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        """
        Trains a SVM model given a hyperparameter configuration and
        evaluates the model on the validation set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM model
        fidelity: Dict, None
            Fidelity parameters for the SVM model, check get_fidelity_space(). Uses default (max) value if None.
        shuffle : bool
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                train_loss : training loss
                fidelity : used fidelities in this evaluation
        """
        start_time = time.time()

        self.seed = seed

        # if shuffle:
        #     self.shuffle_data(self.seed)

        # Transform hyperparameters to linear scale
        hp_c = np.exp(float(configuration['C']))
        hp_gamma = np.exp(float(configuration['gamma']))

        # Train support vector machine
        model = self.get_pipeline(hp_c, hp_gamma)
        model.fit(self.x_train, self.y_train)

        # Compute validation error
        train_loss = 1 - self.accuracy_scorer(model, self.x_train, self.y_train)
        val_loss = 1 - self.accuracy_scorer(model, self.x_valid, self.y_valid)

        cost = time.time() - start_time

        # return {'function_value': float(val_loss),
        #         "cost": cost,
        #         'info': {'train_loss': float(train_loss),
        #                  'fidelity': fidelity}}

        results = {list(self.objective_info.keys())[0]: float(val_loss)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results

    # pylint: disable=arguments-differ
    def objective_function_test(self, configuration: Union[Dict],
                                fidelity: Union[Dict, None] = None,
                                shuffle: bool = False,
                                seed: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a SVM model with a given configuration on both the X_train
        and validation data set and evaluates the model on the X_test data set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM Model
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        shuffle : bool
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : X_test loss
            cost : time to X_train and evaluate the model
            info : Dict
                train_valid_loss: Loss on the train+valid data set
                fidelity : used fidelities in this evaluation
        """


        self.seed = seed

        if shuffle:
            self.shuffle_data(self.seed)

        start_time = time.time()

        # Concatenate training and validation dataset
        if isinstance(self.x_train, sparse.csr.csr_matrix) or isinstance(self.x_valid, sparse.csr.csr_matrix):
            data = sparse.vstack((self.x_train, self.x_valid))
        else:
            data = np.concatenate((self.x_train, self.x_valid))
        targets = np.concatenate((self.y_train, self.y_valid))

        # Transform hyperparameters to linear scale
        hp_c = np.exp(float(configuration['C']))
        hp_gamma = np.exp(float(configuration['gamma']))

        model = self.get_pipeline(hp_c, hp_gamma)
        model.fit(data, targets)

        # Compute validation error
        train_valid_loss = 1 - self.accuracy_scorer(model, data, targets)

        # Compute test error
        test_loss = 1 - self.accuracy_scorer(model, self.x_test, self.y_test)

        cost = time.time() - start_time

        # return {'function_value': float(test_loss),
        #         "cost": cost,
        #         'info': {'train_valid_loss': float(train_valid_loss),
        #                  'fidelity': fidelity}}

        results = {list(self.objective_info.keys())[0]: float(test_loss)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 

        return results

    def get_pipeline(self, C: float, gamma: float) -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """

        model = pipeline.Pipeline([
            ('preprocess_impute',
             ColumnTransformer([
                 ("categorical", "passthrough", self.categorical_data),
                 ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
            ('preprocess_one_hot',
             ColumnTransformer([
                 ("categorical", OneHotEncoder(categories=self.categories), self.categorical_data),
                 ("continuous", MinMaxScaler(feature_range=(0, 1)), ~self.categorical_data)])),
            ('svm',
             svm.SVC(gamma=gamma, C=C, random_state=self.seed, cache_size=self.cache_size))
        ])
        return model

    def get_configuration_space(self, seed: Union[int, None] = None):
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the SVM Model

        For a detailed explanation of the hyperparameters:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """

        variables=[Continuous('C', [-10, 10]), Continuous('gamma', [-10, 10])]
        ss = SearchSpace(variables)
        return ss


    def get_fidelity_space(self, seed: Union[int, None] = None):
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the SupportVector Benchmark

        Fidelities
        ----------
        dataset_fraction: float - [0.1, 1]
            fraction of training data set to use

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        # seed = seed if seed is not None else np.random.randint(1, 100000)
        # fidel_space = CS.ConfigurationSpace(seed=seed)

        # fidel_space.add_hyperparameters([
        #     CS.UniformFloatHyperparameter("dataset_fraction", lower=0.0, upper=1.0, default_value=1.0, log=False),
        # ])

        fs = FidelitySpace([])
        return fs


    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        return {'name': 'Support Vector Machine',
                'references': ["@InProceedings{pmlr-v54-klein17a",
                               "author = {Aaron Klein and Stefan Falkner and Simon Bartels and Philipp Hennig and "
                               "Frank Hutter}, "
                               "title = {{Fast Bayesian Optimization of Machine Learning Hyperparameters on "
                               "Large Datasets}}"
                               "pages = {528--536}, year = {2017},"
                               "editor = {Aarti Singh and Jerry Zhu},"
                               "volume = {54},"
                               "series = {Proceedings of Machine Learning Research},"
                               "address = {Fort Lauderdale, FL, USA},"
                               "month = {20--22 Apr},"
                               "publisher = {PMLR},"
                               "pdf = {http://proceedings.mlr.press/v54/klein17a/klein17a.pdf}, "
                               "url = {http://proceedings.mlr.press/v54/klein17a.html}, "
                               ],
                'code': 'https://github.com/automl/HPOlib1.5/blob/container/hpolib/benchmarks/ml/svm_benchmark.py',
                'shape of train data': self.x_train.shape,
                'shape of test data': self.x_test.shape,
                'shape of valid data': self.x_valid.shape,
                'initial random seed': self.seed,
                'task_id': self.task_id
                }
    
    def get_objectives(self) -> Dict:
        return {'train_loss': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"

if __name__ == '__main__':
    task_lists = [167149, 167152, 126029, 167178, 167177]
    problem = SupportVectorMachine(task_name='svm',task_id=167149, seed=0, budget=10)
    a = problem.f({'C':0.2, 'gamma':-0.3})
    print(a)
