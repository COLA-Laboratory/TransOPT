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
        # 'cp': {'budget': 8, 'time_stamp': 2, 'params': {'input_dim': 2}},
        'Ackley': {'budget': 11, 'time_stamp': 3, 'params':{'input_dim':1}},
        # 'MPB': {'budget': 110, 'time_stamp': 3},
        # 'Griewank': {'budget': 11, 'time_stamp': 3,  'params':{'input_dim':1}},
        # 'DixonPrice': {'budget': 110, 'time_stamp': 3},
        # 'Lunar': {'budget': 110, 'time_stamp': 3},
        # 'XGB': {'budget': 110, 'time_stamp': 3},
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

