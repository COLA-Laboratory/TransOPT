import numpy as np
from collections import defaultdict
from transopt.utils.sk import Rx
import scipy
import os
from multiprocessing import Process, Manager
from transopt.analysis.table_to_latex import matrix_to_latex
from transopt.analysis.compile_tex import compile_tex
from transopt.agent.services import Services


class Result():
    def __init__(self):
        self.X = None
        self.Y = None
        self.best_X = None
        self.best_Y = None


def get_results(task_names):
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    db_lock = manager.Lock()
    services = Services(task_queue, result_queue, db_lock)

    results = {}
    methods = []
    tasks = []
    for group_id, group in enumerate(task_names):
        for task_name in group:
            r = Result()
            table_info = services.data_manager.db.query_dataset_info(task_name)
            task = table_info['additional_config']['problem_name']
            method = table_info['additional_config']['Model']
            seed = table_info['additional_config']['seeds']
            if method not in methods:
                methods.append(method)
            if task not in tasks:
                tasks.append(task)
            
            all_data = services.data_manager.db.select_data(task_name)
            objectives = table_info["objectives"]
            obj = objectives[0]["name"]
            obj_data = [data[obj] for data in all_data]
            var_data = [[data[var["name"]] for var in table_info["variables"]] for data in all_data]
            r.X = np.array(var_data)
            r.Y = np.array(obj_data)
            best_id = np.argmin(r.Y)
            r.best_X = r.X[best_id]
            r.best_Y = r.Y[best_id]
            if task not in results:
                results[task] = defaultdict(dict)
            if method not in results[task]:
                results[task][method] = defaultdict(dict)
            results[task][method][seed] = r

    return results, methods, tasks



def record_mean_std(task_names, save_path, **kwargs):
    # Similar to record_mean_std function in PeerComparison.py
    res_mean = {}
    res_std = {}
    res_sig = {}
    results, methods, tasks = get_results(task_names)
    for task_name, task_r in results.items():
        result_mean = []
        result_std = []
        data = {}
        data_mean = {}
        for method, method_r in task_r.items():
            best = []
            for seed, result_obj in method_r.items():
                best.append(result_obj.best_Y)
                data[method] = best.copy()
                data_mean[method] = (np.mean(best), np.std(best))
                result_mean.append(np.mean(best))
                result_std.append(np.std(best))

        res_mean[task_name] = result_mean
        res_std[task_name] = result_std
        rst_m = {}
        sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]))
        for method in methods:
            if method == sorted_dic[0][0]:
                rst_m[method] = '-'
                continue
            s, p = scipy.stats.mannwhitneyu(data[sorted_dic[0][0]], data[method], alternative='two-sided')
            if p < 0.05:
                rst_m[method] = '+'
            else:
                rst_m[method] = '-'
        res_sig[task_name] = rst_m
    latex_code = matrix_to_latex({'mean':res_mean, 'std':res_std, 'significance':res_sig}, tasks, methods,
                                 caption='Performance comparisons of the quality of solutions obtained by different algorithms.')
    save_path = os.path.join(save_path, 'Overview')
    os.makedirs(save_path, exist_ok=True)
    tex_save_path = os.path.join(save_path, 'tex')
    os.makedirs(tex_save_path, exist_ok=True)
    table_path = os.path.join(save_path, 'Table')
    os.makedirs(table_path, exist_ok=True)
    
    with open(os.path.join(tex_save_path, 'compare_mean.tex'), 'w') as f:
        f.write(latex_code)
    try:
        compile_tex(os.path.join(tex_save_path, 'compare_mean.tex'), table_path)
    except:
        pass

    print(f"LaTeX code has been saved to {tex_save_path}")


if __name__ == "__main__":
    task_names = [['Sphere_w1_s1_1715591439', 'Sphere_w1_s1_1715592120']]
    save_path = '/home/gsfall'
    record_mean_std(task_names, save_path)