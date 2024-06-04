

import numpy as np
import dcor
from sklearn.metrics import mutual_info_score
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
from transopt.utils.Normalization import normalize

def correlation_analysis(Exper_folder, tasks, methods, seeds, args):
    ab = AnalysisBase(Exper_folder, tasks=tasks,methods= methods,seeds= seeds, args=args)
    ab.read_data_from_kb()
    task_names = ab.get_task_names()
    for method in methods:
        for seed in seeds:
            for task in task_names:
                a = MutualInformation(ab, task, method, seed)
    Exper_folder = Exper_folder / 'analysis'

def MutualInformation(ab:AnalysisBase, dataset_name, method, seed):
    results = ab.get_results_by_order(['method', 'seed', 'task'])
    res = results[method][seed][dataset_name]
    Y = res.Y
    num_objective = Y.shape[0]

    mi = mutual_info_score(normalize(Y[0]), normalize(Y[1]))
    distance_corr = dcor.distance_correlation(normalize(Y[0]), normalize(Y[1]))

    print("Distance Correlation:", distance_corr)
    print("Mutual Information:", mi)


