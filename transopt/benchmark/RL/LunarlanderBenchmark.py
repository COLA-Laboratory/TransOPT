import gym
import logging
import random
import numpy as np
import ConfigSpace as CS
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from gplearn.genetic import SymbolicRegressor
from typing import Union, Dict

from transopt.benchmark.problem_base import NonTabularProblem
from agent.registry import benchmark_register


logger = logging.getLogger("LunarLanderBenchmark")

# 计算两组数据的 Pearson 相关系数和 p 值


def lunar_lander_simulation(w, print_reward=False, seed=1, dimension=12):
    total_reward = 0.0
    steps = 0
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    s = env.reset(seed=seed)[0]
    while True:
        if dimension == 5:
            a = heuristic_controller5d(s, w, is_continuous=False)
        # elif dimension == 6:
        #     a = heuristic_controller6d(s, w, is_continuous=False)
        # elif dimension == 8:
        #     a = heuristic_controller8d(s, w, is_continuous=False)
        elif dimension == 10:
            a = heuristic_controller10d(s, w, is_continuous=False)
        else:
            a = heuristic_controller(s[0], w)
        s, r, done, info, _ = env.step(a)
        total_reward += r
        steps += 1
        if done:
            break
    if print_reward:
        print(f"Total reward: {total_reward}")
    return total_reward


def heuristic_controller(s, w, is_continuous=True):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * w[1]
    if angle_target > w[2]:
        angle_target = w[2]
    if angle_target < w[-2]:
        angle_target = -w[2]
    hover_target = w[3] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]
    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
    return a


def heuristic_controller5d(s, w, is_continuous=True):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * 1.0
    if angle_target > 0.4:
        angle_target = 0.4
    if angle_target < -0.4:
        angle_target = -0.4
    hover_target = w[1] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[2] - (s[5]) * w[3]
    hover_todo = (hover_target - s[1]) * w[4] - (s[3]) * 0.5
    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


#
# def heuristic_controller6d(s, w, is_continuous=True):
#     # w is the array of controller parameters of shape (1, 12)
#     angle_target = s[0] * w[0] + s[2] *  w[1]
#     if angle_target > 0.4:
#         angle_target = 0.4
#     if angle_target < -0.4:
#         angle_target = -0.4
#     hover_target = w[2] * np.abs(s[0])
#     angle_todo = (angle_target - s[4]) * w[3] - (s[5]) * w[4]
#     hover_todo = (hover_target - s[1]) * w[5] - (s[3]) * 0.5
#     if s[6] or s[7]:
#         angle_todo = 0
#         hover_todo = (
#                 -(s[3]) * 0.5
#         )  # override to reduce fall speed, that's all we need after contact
#
#     if is_continuous:
#         a = np.array([hover_todo * 20 - 1, angle_todo * 20])
#         a = np.clip(a, -1, +1)
#     else:
#         a = 0
#         if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
#             a = 2
#         elif angle_todo < -0.05:
#             a = 3
#         elif angle_todo > +0.05:
#             a = 1
#     return a
#
#
# def heuristic_controller8d(s, w, is_continuous=True):
#     # w is the array of controller parameters of shape (1, 12)
#     angle_target = s[0] * w[0] + s[2] * w[1]
#     if angle_target > w[2]:
#         angle_target = w[2]
#     if angle_target < -w[2]:
#         angle_target = -w[2]
#     hover_target = w[3] * np.abs(s[0])
#     angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
#     hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
#     if s[6] or s[7]:
#         angle_todo = 0
#         hover_todo = (
#             -(s[3]) * 0.5
#         )  # override to reduce fall speed, that's all we need after contact
#
#     if is_continuous:
#         a = np.array([hover_todo * 20 - 1, angle_todo * 20])
#         a = np.clip(a, -1, +1)
#     else:
#         a = 0
#         if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
#             a = 2
#         elif angle_todo < -0.05:
#             a = 3
#         elif angle_todo > +0.05:
#             a = 1
#     return a
#


def heuristic_controller10d(s, w, is_continuous=True):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * w[1]
    if angle_target > w[2]:
        angle_target = w[2]
    if angle_target < -w[2]:
        angle_target = -w[2]
    hover_target = w[3] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]
    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def vanilla_heuristic(s, is_continuous=False):
    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if is_continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


@benchmark_register("Lunar")
class LunarlanderBenchmark(NonTabularProblem):
    """
    DixonPrice function

    :param sd: standard deviation, to generate noisy evaluations of the function.
    """

    lunar_seeds = [2, 3, 4, 5, 10, 14, 15, 19]

    def __init__(self, task_name, task_id, budget, seed, task_type="non-tabular"):
        super(LunarlanderBenchmark, self).__init__(
            task_name=task_name, seed=seed, task_type=task_type, budget=budget
        )
        self.lunar_seed = LunarlanderBenchmark.lunar_seeds[task_id]

    def objective_function(
        self,
        configuration: Union[CS.Configuration, Dict],
        fidelity: Union[Dict, CS.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs,
    ) -> Dict:
        X = np.array([configuration[k] for idx, k in enumerate(configuration.keys())])

        y = lunar_lander_simulation(X, seed=self.lunar_seed, dimension=self.input_dim)
        return {"function_value": float(y), "info": {"fidelity": fidelity}}

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
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
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                CS.UniformFloatHyperparameter(f"x{i}", lower=0, upper=2.0)
                for i in range(10)
            ]
        )

        return cs

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
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
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([])

        return fidel_space

    def get_meta_information(self) -> Dict:
        print(1)
        return {}


if __name__ == "__main__":
    seed_list = [2, 3, 4, 5, 10, 14, 15, 19]
    result_vectors = []
    for seed in seed_list:
        # 设置随机种子
        np.random.seed(seed)
        # 执行函数 100 次并记录结果
        sample_number = 100
        dim = 10

        fixed_dims = {0: 2.0, 1: 1.8, 2: 0.01, 4: 0.01, 5: 0.01}

        # Generate random data for other dimensions
        samples_x = np.random.uniform(-1, 1, (sample_number, dim))

        # Assign fixed values to specified dimensions
        # for dim, value in fixed_dims.items():
        #     samples_x[:, dim] = value

        # samples_x= np.random.uniform(0, 2, size=(sample_number, dim))
        # samples_x = np.sort(samples_x, axis=0)
        # samples_x =  np.random.uniform(0, 2, size=(100, 10))
        bench = LunarlanderBenchmark(task_name="lunar", task_id=0, seed=0, budget=10000)
        xx = {}
        for i in range(10):
            xx[f"x{i}"] = samples_x[0][i]
        result = bench.f(xx)
        print(result)
        # 将结果转换为 100*1 的向量
        result_vector = np.array(result).reshape(-1, 1)

        # 将结果向量存储到列表中
        result_vectors.append(result_vector)

        plt.figure()
        plt.clf()
        # 绘制采样结果的分布图
        plt.hist(result, bins=30, density=True, alpha=0.7)
        # # 添加横纵轴标签和标题
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Distribution of Sampled Function, seed:{seed}")
        plt.show()
        # plt.savefig(f'seed_{seed}')

        # 训练symbolic regressor
        est_gp = SymbolicRegressor(
            population_size=5000,
            generations=20,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            parsimony_coefficient=0.01,
            random_state=0,
        )
        est_gp.fit(samples_x, result_vector)

        print("最佳程序：", est_gp._program)

    # 对每个结果向量进行相关性分析
    for i, vector1 in enumerate(result_vectors):
        for j, vector2 in enumerate(result_vectors):
            if i != j:
                correlation, p = spearmanr(vector1.flatten(), vector2.flatten())
                print(
                    f"Correlation between seed {seed_list[i]} and seed {seed_list[j]}: {correlation},p_{p}"
                )
