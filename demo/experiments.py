import logging
import os
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

from transopt.Benchmark import construct_test_suits
from optimizer.construct_optimizer import get_optimizer
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TaskDataHandler import OptTaskDataHandler


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_experiments(tasks, args):
    logger = logging.getLogger(__name__)
    kb = construct_knowledgebase(args)
    testsuits = construct_test_suits(tasks, args.seed)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)


if __name__ == "__main__":
    tasks = {
        # 'DBMS':{'budget': 11, 'time_stamp': 3},
        # 'GCC' : {'budget': 11, 'time_stamp': 3},
        # 'LLVM' : {'budget': 11, 'time_stamp': 3},
        'Ackley': {'budget': 11, 'workloads': [1,2,3], 'params':{'input_dim':1}},
        # 'MPB': {'budget': 110, 'time_stamp': 3},
        # 'Griewank': {'budget': 11, 'time_stamp': 3,  'params':{'input_dim':2}},
        # "AckleySphere": {"budget": 1000, "workloads":[1,2,3], "params": {"input_dim": 2}},
        # 'Lunar': {'budget': 110, 'time_stamp': 3},
        # 'XGB': {'budget': 110, 'time_stamp': 3},
    }

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-im", "--init_method", type=str, default="random")
    parser.add_argument("-in", "--init_number", type=int, default=7)
    parser.add_argument(
        "-p", "--exp_path", type=str, default=f"{package_dir}/../LFL_experiments"
    )
    parser.add_argument(
        "-n", "--exp_name", type=str, default="test"
    )  # 实验名称，保存在experiments中
    parser.add_argument("-s", "--seed", type=int, default=0)  # 设置随机种子，与迭代次数相关
    parser.add_argument(
        "-m", "--optimizer", type=str, default="MTBO"
    )  # 设置method:WS,MT,INC
    parser.add_argument("-v", "--verbose", type=bool, default=True)
    parser.add_argument("-norm", "--normalize", type=str, default="norm")
    parser.add_argument("-sm", "--save_mode", type=int, default=1)  # 控制是否保存模型
    parser.add_argument("-lm", "--load_mode", type=bool, default=False)  # 控制是否从头开始
    parser.add_argument(
        "-ac", "--acquisition_func", type=str, default="LCB"
    )  # 控制BO的acquisition function
    args = parser.parse_args()

    run_experiments(tasks, args)
