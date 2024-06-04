import numpy as np
from collections import defaultdict
from transopt.utils.sk import Rx
import scipy
from transopt.ResultAnalysis.TableToLatex import matrix_to_latex
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
from transopt.ResultAnalysis.CompileTex import compile_tex
import os

table_registry = {}

# 注册函数的装饰器
def Tabel_register(name):
    def decorator(func_or_class):
        if name in table_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        table_registry[name] = func_or_class
        return func_or_class
    return decorator

@Tabel_register('mean')
def record_mean_std(ab:AnalysisBase, save_path, **kwargs):
    # Similar to record_mean_std function in PeerComparison.py
    res_mean = {}
    res_std = {}
    res_sig = {}
    results = ab.get_results_by_order(["task", "method", "seed"])
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
        for method in ab.get_methods():
            if method == sorted_dic[0][0]:
                rst_m[method] = '-'
                continue
            s, p = scipy.stats.mannwhitneyu(data[sorted_dic[0][0]], data[method], alternative='two-sided')
            if p < 0.05:
                rst_m[method] = '+'
            else:
                rst_m[method] = '-'
        res_sig[task_name] = rst_m
    latex_code = matrix_to_latex({'mean':res_mean, 'std':res_std, 'significance':res_sig}, list(ab.get_task_names()), list(ab.get_methods()),
                                 caption='Performance comparisons of the quality of solutions obtained by different algorithms.')
    save_path = save_path / 'Overview'
    os.makedirs(save_path, exist_ok=True)
    tex_save_path = save_path / 'tex'
    os.makedirs(tex_save_path, exist_ok=True)
    table_path = save_path / 'Table'
    os.makedirs(table_path, exist_ok=True)

    with open(tex_save_path / f"compare_mean.tex", 'w') as f:
        f.write(latex_code)
    try:
        compile_tex(tex_save_path / f"compare_mean.tex" , table_path)
    except:
        pass

    print(f"LaTeX code has been saved to {tex_save_path}")

@Tabel_register('cr')
def record_convergence_rate(ab:AnalysisBase, save_path, **kwargs):
    # Similar to record_convergence function in PeerComparison.py
    res_mean = {}
    res_std = {}
    res_sig = {}

    def acc_iter(Y, anchor_value):
        for i in range(1, len(Y)):
            best_fn = np.min(Y[:i])
            if best_fn <= anchor_value:
                return i/len(Y)

        return 1
    # 遍历 data 字典，收集 best_Y 值
    results = ab.get_results_by_order(["method", "seed", "task"])
    best_Y_values = defaultdict(list)
    for method, tasks in results.items():
        for seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                best_Y = result_obj.best_Y
                if best_Y is not None:
                    best_Y_values[task_name].append(best_Y)

    # 计算并返回每个 task_name 下 best_Y 值的 3/4 分位数
    quantiles = {task_name: np.percentile(values, 75) for task_name, values in best_Y_values.items()}
    results = ab.get_results_by_order(["task", "method", "seed"])
    for task_name, task_r in results.items():
        result_mean = []
        result_std = []
        data = {}
        data_mean = {}
        for method, method_r in task_r.items():
            best = []
            for seed, result_obj in method_r.items():
                Y = result_obj.Y
                if Y is None:
                    raise ValueError(f"Y is not set for method {method}, task {task_name}")

                cr = acc_iter(Y, anchor_value=quantiles[task_name])
                best.append(cr)

            data[method] = best.copy()
            data_mean[method] = (np.mean(best), np.std(best))
            result_mean.append(np.mean(best))
            result_std.append(np.std(best))

        res_mean[task_name] = result_mean
        res_std[task_name] = result_std

        rst_m = {}
        sorted_dic = sorted(data_mean.items(), key=lambda kv: (kv[1][0]), reverse=False)
        for method in ab.get_methods():
            if method == sorted_dic[0][0]:
                rst_m[method] = '-'
                continue
            s, p = scipy.stats.mannwhitneyu(data[sorted_dic[0][0]], data[method], alternative='two-sided')
            if p < 0.05:
                rst_m[method] = '+'
            else:
                rst_m[method] = '-'
        res_sig[task_name] = rst_m
    latex_code = matrix_to_latex({'mean': res_mean, 'std': res_std, 'significance': res_sig}, list(ab.get_task_names()),
                                 list(ab.get_methods()),
                                 caption='Convergence rate comparison among different algorithms.')
    save_path = save_path / 'Overview'
    os.makedirs(save_path, exist_ok=True)
    tex_save_path = save_path / 'tex'
    os.makedirs(tex_save_path, exist_ok=True)
    table_path = save_path / 'Table'
    os.makedirs(table_path, exist_ok=True)

    with open(tex_save_path / f"compare_convergence_rate.tex", 'w') as f:
        f.write(latex_code)
    try:
        compile_tex(tex_save_path / f"compare_convergence_rate.tex", table_path)
    except:
        pass

    print(f"LaTeX code has been saved to {tex_save_path}")
