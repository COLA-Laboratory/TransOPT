import logging
import os
import argparse

from pathlib import Path
from transopt.ResultAnalysis.AnalysisPipeline import analysis_pipeline


def run_analysis(Exper_folder:Path, tasks, methods, seeds, args):
    logger = logging.getLogger(__name__)
    analysis_pipeline(Exper_folder, tasks=tasks, methods=methods, seeds=seeds, args=args)



if __name__ == '__main__':
    tasks = {
        "GCC": {"budget": samples_num, "workloads": workloads},
        "LLVM": {"budget": samples_num, "workloads": workloads},
    }

    available_workloads = CompilerBenchmarkBase.AVAILABLE_WORKLOADS
    split_workloads = split_into_segments(available_workloads, 10)

    if split_index >= len(split_workloads):
        raise IndexError("split index out of range")

    workloads = split_workloads[split_index]

    tasks = {
        "GCC": {"budget": samples_num, "workloads": workloads},
        "LLVM": {"budget": samples_num, "workloads": workloads},
    }
    Methods_list = {'MTBO', 'BO'}
    Seeds = [1,2,3,4,5]

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-in", "--init_number", type=int, default=0)
    parser.add_argument("-p", "--exp_path", type=str, default='../LFL_experiments')
    parser.add_argument("-n", "--exp_name", type=str, default='test')  # 实验名称，保存在experiments中
    parser.add_argument("-c", "--comparision", type=bool, default=True)
    parser.add_argument("-a", "--track", type=bool, default=True)
    parser.add_argument("-r", "--report", type=bool, default=False)


    args = parser.parse_args()
    Exp_name = args.exp_name
    Exp_folder = args.exp_path
    Exper_folder = '{}/{}'.format(Exp_folder, Exp_name)
    Exper_folder = Path(Exper_folder)
    run_analysis(Exper_folder, tasks=tasks, methods=Methods_list, seeds = Seeds, args=args)

