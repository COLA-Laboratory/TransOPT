import numpy as np
import GPy
from typing import Dict, Union, List

from transopt.optimizer.optimizer_base import BOBase
from agent.registry import optimizer_register
from transopt.utils.Normalization import get_normalizer
from transopt.utils.serialization import ndarray_to_vectors,vectors_to_ndarray

from sklearn.ensemble import ExtraTreesRegressor



def calculate_gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities ** 2)
    return gini


def features_by_gini(data, labels):
    features_gini = []

    # 遍历每个特征
    for feature_idx in range(data.shape[1]):
        current_feature_values = data[:, feature_idx]

        # 假设的分割方式：基于每个值的基尼指数
        gini_indexes = []
        for split_value in np.unique(current_feature_values):
            left_split = labels[current_feature_values <= split_value]
            right_split = labels[current_feature_values > split_value]

            # 计算左右分割的加权基尼指数
            left_gini = calculate_gini_index(left_split)
            right_gini = calculate_gini_index(right_split)
            weighted_gini = (len(left_split) * left_gini + len(right_split) * right_gini) / len(labels)
            gini_indexes.append(weighted_gini)

        # 取这个特征下的最小基尼指数
        min_gini = min(gini_indexes) if gini_indexes else 1  # 防止某个特征下所有值相同
        features_gini.append((feature_idx, min_gini))

    return features_gini

@optimizer_register("CauMO")
class CauMO(BOBase):
    def __init__(self, config: Dict, rate_oversampling = 4, seed = 0, **kwargs):
        super(CauMO, self).__init__(config=config)

        self.init_method = "Random"
        self.verbose = config.get("verbose", True)
        self.pop_size = config.get("pop_size", 10)
        self.ini_num = self.pop_size

        self.second_space = None
        self.third_space = None

        self.model = []
        self.acf = "CauMOACF"

        self.rate_oversampling = rate_oversampling
        self.num_duplicates = int(rate_oversampling * 4.0)
        self.seed = seed

    def initial_sample(self):
        return self.random_sample(self.ini_num)

    def random_sample(self, num_samples: int) -> List[Dict]:
        """
        Initialize random samples.

        :param num_samples: Number of random samples to generate
        :return: List of dictionaries, each representing a random sample
        """
        if self.input_dim is None:
            raise ValueError(
                "Input dimension is not set. Call set_search_space() to set the input dimension."
            )

        random_samples = []
        for _ in range(num_samples):
            sample = {}
            for var_info in self.search_space.config_space:
                var_name = var_info["name"]
                var_domain = var_info["domain"]
                # Generate a random floating-point number within the specified range
                random_value = np.random.uniform(var_domain[0], var_domain[1])
                sample[var_name] = random_value
            random_samples.append(sample)

        random_samples = self.inverse_transform(random_samples)
        return random_samples

    def update_model(self, Data):
        Target_Data = Data["Target"]
        assert "X" in Target_Data

        X = Target_Data["X"]
        Y = Target_Data["Y"]
        assert Y.shape[0] == self.num_objective

        if self.normalizer is not None:
            Y_norm = np.array([self.normalizer(y) for y in Y])

        if len(self.model_list) == 0:
            self.create_model(X, Y_norm)
        else:
            # self.set_data(X, Y_norm)
            self.fit_data(X, Y_norm)
        # try:
        #     for i in range(len(self.model_list)):
        #         self.model_list[i].optimize_restarts(
        #             num_restarts=1, verbose=self.verbose, robust=True
        #         )
        # except np.linalg.linalg.LinAlgError as e:
        #     # break
        #     print("Error: np.linalg.linalg.LinAlgError")

        self.Y_Norm = None
    def create_model(self, X, Y):
        assert self.num_objective is not None

        compile_time_model = ExtraTreesRegressor(
         n_estimators=200,
         max_features='sqrt',
         bootstrap=True,
         random_state=self.seed,
         max_samples = self.rate_oversampling / self.num_duplicates,
        )
        compile_time_model.fit(X, Y[2][:, np.newaxis])

        # Kc = GPy.kern.RBF(input_dim=self.input_dim)
        # compile_time_model = GPy.models.GPRegression(X, Y[2][:, np.newaxis], kernel=Kc, normalizer=None)


        file_size_feature_rank = features_by_gini(X, Y[1])
        self.file_size_rep_feature = sorted(file_size_feature_rank, key=lambda x: x[1])[0][0]

        X_file = X.copy()
        X_file[:, self.file_size_rep_feature] = np.clip(2 * (Y[2] - (-3)) / 6 - 1, -1, 1)

        file_size_model = ExtraTreesRegressor(
         n_estimators=200,
         max_features='sqrt',
         bootstrap=True,
         random_state=self.seed,
         max_samples = self.rate_oversampling / self.num_duplicates,
        )

        file_size_model.fit(X_file, Y[1][:, np.newaxis])

        # Kf = GPy.kern.RBF(input_dim=self.input_dim)
        # file_size_model = GPy.models.GPRegression(X_file, Y[1][:, np.newaxis], kernel=Kf, normalizer=None)

        run_time_feature_rank = features_by_gini(X, Y[0])
        run_time_feature_rank = sorted(run_time_feature_rank, key=lambda x: x[1])
        self.st_run_time_rep_feature = run_time_feature_rank[0][0]
        self.nd_run_time_rep_feature = run_time_feature_rank[1][0]
        X_rtime = X.copy()
        X_rtime[:, self.st_run_time_rep_feature] = np.clip(2 * (Y[2] - (-3)) / 6 - 1, -1, 1)
        X_rtime[:, self.nd_run_time_rep_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        # Kr = GPy.kern.RBF(input_dim=self.input_dim)
        # running_time_model = GPy.models.GPRegression(X_rtime, Y[0][:, np.newaxis], kernel=Kr, normalizer=None)
        running_time_model = ExtraTreesRegressor(
         n_estimators=200,
         max_features='sqrt',
         bootstrap=True,
         random_state=self.seed,
         max_samples = self.rate_oversampling / self.num_duplicates,
        )
        running_time_model.fit(X_rtime, Y[0][:, np.newaxis])

        # compile_time_model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
        # compile_time_model['.*rbf.variance'].constrain_fixed(1.0)
        # file_size_model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
        # file_size_model['.*rbf.variance'].constrain_fixed(1.0)
        # running_time_model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
        # running_time_model['.*rbf.variance'].constrain_fixed(1.0)

        self.model_list.append(compile_time_model)
        self.model_list.append(file_size_model)
        self.model_list.append(running_time_model)

    def set_data(self, X, Y):
        self.model_list[0].set_XY(X, Y[2][:, np.newaxis])
        file_size_feature_rank = features_by_gini(X, Y[1])
        self.file_size_rep_feature = sorted(file_size_feature_rank, key=lambda x: x[1])[0][0]
        X_file = X.copy()
        X_file[:, self.file_size_rep_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        self.model_list[1].set_XY(X_file, Y[1][:, np.newaxis])

        run_time_feature_rank = features_by_gini(X, Y[0])
        run_time_feature_rank = sorted(run_time_feature_rank, key=lambda x: x[1])
        self.st_run_time_rep_feature = run_time_feature_rank[0][0]
        self.nd_run_time_rep_feature = run_time_feature_rank[1][0]
        X_rtime = X.copy()
        X_rtime[:, self.st_run_time_rep_feature] = np.clip(2 * (Y[2] - (-3)) / 6 - 1, -1, 1)
        X_rtime[:, self.nd_run_time_rep_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        self.model_list[2].set_XY(X_rtime, Y[0][:, np.newaxis])


    def fit_data(self, X, Y):
        self.model_list[0].fit(X, Y[2][:, np.newaxis])
        file_size_feature_rank = features_by_gini(X, Y[1])
        self.file_size_rep_feature = sorted(file_size_feature_rank, key=lambda x: x[1])[0][0]
        X_file = X.copy()
        X_file[:, self.file_size_rep_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        self.model_list[1].fit(X_file, Y[1][:, np.newaxis])

        run_time_feature_rank = features_by_gini(X, Y[0])
        run_time_feature_rank = sorted(run_time_feature_rank, key=lambda x: x[1])
        self.st_run_time_rep_feature = run_time_feature_rank[0][0]
        self.nd_run_time_rep_feature = run_time_feature_rank[1][0]
        X_rtime = X.copy()
        X_rtime[:, self.st_run_time_rep_feature] = np.clip(2 * (Y[2] - (-3)) / 6 - 1, -1, 1)
        X_rtime[:, self.nd_run_time_rep_feature] = np.clip(2 * (Y[1] - (-3)) / 6 - 1, -1, 1)
        self.model_list[2].fit(X_rtime, Y[0][:, np.newaxis])


    def suggest(self, n_suggestions: Union[None, int] = None) -> List[Dict]:
        if self._X.size == 0:
            suggests = self.initial_sample()
            return suggests
        elif self._X.shape[0] < self.ini_num:
            pass
        else:
            if "normalize" in self.config:
                self.normalizer = get_normalizer(self.config["normalize"])

            Data = {"Target": {"X": self._X, "Y": self._Y}}
            self.update_model(Data)
            suggested_sample, acq_value = self.evaluator.compute_batch(
                None, context_manager=None
            )
            suggested_sample = self.search_space.zip_inputs(suggested_sample)
            suggested_sample = ndarray_to_vectors(
                self._get_var_name("search"), suggested_sample
            )
            design_suggested_sample = self.inverse_transform(suggested_sample)

            return design_suggested_sample

    def observe(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
        super().observe(input_vectors, output_value)


        if "normalize" in self.config:
            self.normalizer = get_normalizer(self.config["normalize"])

        self.Y_Norm = np.array([self.normalizer(y) for y in self._Y])
    def predict(self, X, full_cov=False):

        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))

        # compile_time_mean, compile_time_var = self.model_list[0].predict(X, full_cov=full_cov)
        compile_time_mean, compile_time_var = self.raw_predict(X, self.model_list[0])
        X_file = X.copy()
        X_file[:, self.file_size_rep_feature] = np.clip(2 * (compile_time_mean[:, 0] - (-3)) / 6 - 1, -1, 1)
        file_size_mean, file_size_var = self.raw_predict(X_file, self.model_list[1])

        X_run = X.copy()
        X_run[:, self.st_run_time_rep_feature] = np.clip(2 * (compile_time_mean[:, 0] - (-3)) / 6 - 1, -1, 1)
        X_run[:, self.nd_run_time_rep_feature] = np.clip(2 * (file_size_mean[:, 0] - (-3)) / 6 - 1, -1, 1)
        run_time_mean, run_time_var = self.raw_predict(X_run, self.model_list[2])

        pred_mean = np.hstack((pred_mean, run_time_mean, file_size_mean, compile_time_mean))

        if full_cov:
            pred_var = np.hstack((pred_var, run_time_var, file_size_var, compile_time_var))
        else:
            pred_var = np.hstack((pred_var, run_time_var, file_size_var, compile_time_var))

        # pred_mean = np.append(pred_mean, run_time_mean)
        # pred_mean = np.append(pred_mean, file_size_mean)
        # pred_mean = np.append(pred_mean, compile_time_mean)
        # pred_var = np.append(pred_var, run_time_var)
        # pred_var = np.append(pred_var, file_size_var)
        # pred_var = np.append(pred_var, compile_time_var)

        return pred_mean, pred_var

    def raw_predict(self, X, model):
        _X_test = X.copy()

        mu = model.predict(_X_test)
        cov = self.raw_predict_var(_X_test, model, mu)
        return mu[:,np.newaxis], cov[:,np.newaxis]

    def raw_predict_var(self, X, trees,  predictions, min_variance=0.1):
        std = np.zeros(len(X))
        for tree in trees:
            var_tree = tree.tree_.impurity[tree.apply(X)]

            # This rounding off is done in accordance with the
            # adjustment done in section 4.3.3
            # of http://arxiv.org/pdf/1211.0906v2.pdf to account
            # for cases such as leaves with 1 sample in which there
            # is zero variance.
            var_tree[var_tree < min_variance] = min_variance
            mean_tree = tree.predict(X)
            std += var_tree + mean_tree ** 2

        std /= len(trees)
        std -= predictions ** 2.0
        std[std < 0.0] = 0.0
        std = std ** 0.5
        return std
    def model_reset(self):
        self.model_list = []
        self.kernel_list = []

    def get_fmin(self):
        "Get the minimum of the current model."
        m, v = self.predict(self._X)

        return m.min()

    def get_fmin_by_id(self, idx):
        "Get the minimum of the current model."
        m, v = self.predict_by_id(self._X, idx)

        return m.min()
