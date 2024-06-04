import numpy as np
from sklearn.metrics import mutual_info_score

from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
from transopt.utils.Normalization import normalize


def parego_analysis(Exper_folder, tasks, methods, seeds, args):
    ab = AnalysisBase(Exper_folder, tasks=tasks,methods= methods,seeds= seeds, args=args)
    ab.read_data_from_kb()
    task_names = ab.get_task_names()
    for method in methods:
        for seed in seeds:
            for task in task_names:
                a = MutualInformation(ab, task, method, seed)
    Exper_folder = Exper_folder / 'analysis'