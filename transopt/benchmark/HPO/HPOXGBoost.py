import os
import time
import logging
import torch
import numpy as np
import xgboost as xgb
from typing import Union, Tuple, Dict, List
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder

from transopt.utils.openml_data_manager import OpenMLHoldoutDataManager
from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

from transopt.optimizer.sampler.random import RandomSampler

os.environ['OMP_NUM_THREADS'] = "1"
logger = logging.getLogger('XGBBenchmark')


@problem_registry.register('XGB')
class XGBoostBenchmark(NonTabularProblem):
    task_lists = [167149, 167152, 126029, 167178, 167177, 167153, 167154, 167155, 167156]
    problem_type = 'hpo'
    num_variables = 10
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
        n_threads  : int, None
        seed : np.random.RandomState, int, None
        """
        super(XGBoostBenchmark, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        self.task_id = XGBoostBenchmark.task_lists[workload]
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_threads = 1
        self.budget = budget
        self.accuracy_scorer = make_scorer(accuracy_score)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, variable_types = \
            self.get_data()
        self.categorical_data = np.array([var_type == 'categorical' for var_type in variable_types])

        # XGB needs sorted data. Data should be (Categorical + numerical) not mixed.
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

        # Determine the number of categories in the labels.
        # In case of binary classification ``self.num_class`` has to be 1 for xgboost.
        self.num_class = len(np.unique(np.concatenate([self.y_train, self.y_test, self.y_valid])))
        self.num_class = 1 if self.num_class == 2 else self.num_class

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
        Trains a XGBoost model given a hyperparameter configuration and
        evaluates the model on the validation set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the XGBoost model
        fidelity: Dict, None
            Fidelity parameters for the XGBoost model, check get_fidelity_space(). Uses default (max) value if None.
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
                train_loss : trainings loss
                fidelity : used fidelities in this evaluation
        """
        self.seed = seed

        # if shuffle:
        #     self.shuffle_data(self.seed)

        start = time.time()

        model = self._get_pipeline(**configuration)
        model.fit(X=self.x_train, y=self.y_train)

        train_loss = 1 - self.accuracy_scorer(model, self.x_train, self.y_train)
        val_loss = 1 - self.accuracy_scorer(model, self.x_valid, self.y_valid)
        cost = time.time() - start

        # return {'function_value': float(val_loss),
        #         'cost': cost,
        #         'info': {'train_loss': float(train_loss),
        #                  'fidelity': fidelity}
        #         }
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
        Trains a XGBoost model with a given configuration on both the train
        and validation data set and evaluates the model on the test data set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the XGBoost Model
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
            function_value : test loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        default_dataset_fraction = self.get_fidelity_space().get_hyperparameter('dataset_fraction').default_value
        if fidelity['dataset_fraction'] != default_dataset_fraction:
            raise NotImplementedError(f'Test error can not be computed for dataset_fraction <= '
                                      f'{default_dataset_fraction}')

        self.seed = seed

        if shuffle:
            self.shuffle_data(self.seed)

        start = time.time()

        # Impute potential nan values with the feature-
        data = np.concatenate((self.x_train, self.x_valid))
        targets = np.concatenate((self.y_train, self.y_valid))

        model = self._get_pipeline(**configuration)
        model.fit(X=data, y=targets)

        test_loss = 1 - self.accuracy_scorer(model, self.x_test, self.y_test)
        cost = time.time() - start

        return {'function_value': float(test_loss),
                'cost': cost,
                'info': {'fidelity': fidelity}}

    def get_configuration_space(self, seed: Union[int, None] = None):
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

        variables=[Continuous('eta', [-10.0, 0.0]),
                   Integer('max_depth', [1, 15]),
                   Continuous('min_child_weight', [0.0, 7.0]),
                   Continuous('colsample_bytree', [0.01, 1.0]),
                   Continuous('colsample_bylevel', [0.01, 1.0]),
                   Continuous('reg_lambda', [-10.0, 10.0]),
                   Continuous('reg_alpha', [-10.0, 10.0]),
                   Continuous('subsample_per_it', [0.1, 1.0]),
                   Integer('n_estimators', [1, 50]),
                   Continuous('gamma', [0.0, 1.0])]
        ss = SearchSpace(variables)
        return ss


    def get_fidelity_space(self, seed: Union[int, None] = None):
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
        # seed = seed if seed is not None else np.random.randint(1, 100000)
        # fidel_space = CS.ConfigurationSpace(seed=seed)

        # fidel_space.add_hyperparameters([
        #     CS.UniformFloatHyperparameter("dataset_fraction", lower=0.0, upper=1.0, default_value=1.0, log=False),
        #     CS.UniformIntegerHyperparameter("n_estimators", lower=1, upper=256, default_value=256, log=False)
        # ])

        # return fidel_space
        fs = FidelitySpace([])
        return fs


    def get_meta_information(self) -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'XGBoost',
                'references': ['@article{probst2019tunability,'
                               'title={Tunability: Importance of hyperparameters of machine learning algorithms.},'
                               'author={Probst, Philipp and Boulesteix, Anne-Laure and Bischl, Bernd},'
                               'journal={J. Mach. Learn. Res.},'
                               'volume={20},'
                               'number={53},'
                               'pages={1--32},'
                               'year={2019}'
                               '}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/development/hpolib/benchmarks/ml/'
                        'xgboost_benchmark_old.py',
                'shape of train data': self.x_train.shape,
                'shape of test data': self.x_test.shape,
                'shape of valid data': self.x_valid.shape,
                'initial random seed': self.seed,
                'task_id': self.task_id
                }

    def _get_pipeline(self, max_depth: int, eta: float, min_child_weight: int,
                      colsample_bytree: float, colsample_bylevel: float, reg_lambda: int, reg_alpha: int,
                      n_estimators: int, subsample_per_it: float, gamma: float) \
            -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """
        objective = 'binary:logistic' if self.num_class <= 2 else 'multi:softmax'

        if torch.cuda.is_available():
            clf = pipeline.Pipeline([
                ('preprocess_impute',
                 ColumnTransformer([
                     ("categorical", "passthrough", self.categorical_data),
                     ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
                ('preprocess_one_hot',
                 ColumnTransformer([
                     ("categorical", OneHotEncoder(categories=self.categories, sparse=False), self.categorical_data),
                     ("continuous", "passthrough", ~self.categorical_data)])),
                ('xgb',
                 xgb.XGBClassifier(
                     max_depth=max_depth,
                     learning_rate=np.exp2(eta),
                     min_child_weight=np.exp2(min_child_weight),
                     colsample_bytree=colsample_bytree,
                     colsample_bylevel=colsample_bylevel,
                     reg_alpha=np.exp2(reg_alpha),
                     reg_lambda=np.exp2(reg_lambda),
                     n_estimators=n_estimators,
                     objective=objective,
                     n_jobs=self.n_threads,
                     random_state=self.seed,
                     num_class=self.num_class,
                     subsample=subsample_per_it,
                     gamma=gamma,
                     tree_method='gpu_hist',
                     gpu_id=0
                 ))
            ])
        else:
            clf = pipeline.Pipeline([
                ('preprocess_impute',
                 ColumnTransformer([
                     ("categorical", "passthrough", self.categorical_data),
                     ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
                ('preprocess_one_hot',
                 ColumnTransformer([
                     ("categorical", OneHotEncoder(categories=self.categories), self.categorical_data),
                     ("continuous", "passthrough", ~self.categorical_data)])),
                ('xgb',
                 xgb.XGBClassifier(
                     max_depth=max_depth,
                     learning_rate=np.exp2(eta),
                     min_child_weight=np.exp2(min_child_weight),
                     colsample_bytree=colsample_bytree,
                     colsample_bylevel=colsample_bylevel,
                     reg_alpha=np.exp2(reg_alpha),
                     reg_lambda=np.exp2(reg_lambda),
                     n_estimators=n_estimators,
                     objective=objective,
                     n_jobs=self.n_threads,
                     random_state=self.seed,
                     num_class=self.num_class,
                     subsample=subsample_per_it,
                     gamma = gamma,
                     ))
                ])

        return clf

    def get_objectives(self) -> Dict:
        return {'train_loss': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"

    # def get_var_range(self):
    #     return {'eta':[-10,0], 'max_depth':[1, 15], 'min_child_weight':[0, 7], 'colsample_bytree':[0.01, 1.0], 'colsample_bylevel':[0.01, 1.0],
    #             'reg_lambda':[-10, 10], 'reg_alpha':[-10, 10], 'subsample_per_it':[0.1, 1.0], 'n_estimators':[1, 50], 'gamma':[0,1.0]}
    #
    #
    # def get_var_type(self):
    #     return {'eta':'exp2', 'max_depth':'int', 'min_child_weight':'exp2', 'colsample_bytree':'float','colsample_bylevel':'float',
    #             'reg_lambda':'exp2', 'reg_alpha':'exp2', 'subsample_per_it':'float', 'n_estimators':'int', 'gamma':'float'}




if __name__ == '__main__':
    task_lists = [167149, 167152, 126029, 167178, 167177, 167153, 167154, 167155, 167156]
    workload = 8
    problem = XGBoostBenchmark(task_name='XGB', budget=20, budget_type = 'fes', workload=workload, seed = 0)
    sampler = RandomSampler(3000, config=None)
    space = problem.configuration_space
    samples = sampler.sample(space,3000)
    
    parameters = [space.map_to_design_space(sample) for sample in samples]
    import tqdm
    for para_id in tqdm.tqdm(range(len(parameters))):
        parameters[para_id]['score'] = problem.f(parameters[para_id])['train_loss']
    import pandas as pd
    

    
    df  = pd.DataFrame(parameters)
    df.to_csv(f'XGB_{workload}.csv')

    # a = problem.f({'eta':-0.2, 'max_depth':5, 'min_child_weight':2, 'colsample_bytree':0.4, 'colsample_bylevel':0.4,
    #             'reg_lambda':0.5, 'reg_alpha':-0.2, 'subsample_per_it':0.7, 'n_estimators':20, 'gamma':0.9})
    


