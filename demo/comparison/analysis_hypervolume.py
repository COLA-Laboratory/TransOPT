import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import json

import numpy as np
import pandas as pd
import scipy.stats

from transopt.utils.pareto import calc_hypervolume, find_pareto_front
from transopt.utils.plot import plot3D

target = "gcc"

results_path = package_path / "experiment_results"
gcc_results = results_path / "gcc_archive_new"
llvm_results = results_path / "llvm_archive"

algorithm_list = ["ParEGO", "SMSEGO", "MoeadEGO", "CauMO"]
objectives = ["execution_time", "file_size", "compilation_time"]
seed_list = [65535, 65536, 65537, 65538, 65539]


def load_and_prepare_data(file_path, objectives):
    """
    Loads JSON data and prepares a DataFrame.
    """
    # print(f"Loading data from {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        data = data.get("1", {})

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    # print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]

    # print(f"Loaded {len(df_combined)} data points, removed {len(df_input) - len(df_combined)} duplicates")
    # print()
    return df_combined

def load_data(workload, algorithm, seed):
    if target == "llvm":
        result_file = llvm_results / f"llvm_{workload}" / algorithm / f"{seed}_KB.json"
    else:
        result_file = gcc_results / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
    df = load_and_prepare_data(result_file, objectives)
    return df

def collect_all_data(workload):
    all_data = []
    for algorithm in algorithm_list:
        for seed in seed_list:
            df = load_data(workload, algorithm, seed)
            all_data.append(df[objectives].values)
    all_data = np.vstack(all_data)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0)
    return all_data, global_mean, global_std


def calculate_mean_hypervolume(
    algorithm, workload, global_stats1, global_stats2, normalization_type="min-max"
):
    """
    Calculate mean hypervolume for a given algorithm across all seeds.

    Parameters:
    global_stats1: Global mean or min of all objectives (depending on normalization_type)
    global_stats2: Global std or max of all objectives (depending on normalization_type)
    normalization_type: 'min-max' or 'mean' for different types of normalization
    """
    hypervolume_list = []
    for seed in seed_list:
        df = load_data(workload, algorithm, seed)

        if normalization_type == "mean":
            # Apply mean normalization
            normalized_df = (df[objectives] - global_stats1) / global_stats2
        elif normalization_type == "min-max":
            # Apply min-max normalization
            normalized_df = (df[objectives] - global_stats1) / (
                global_stats2 - global_stats1
            )
        else:
            raise ValueError(
                "Unsupported normalization type. Choose 'mean' or 'min-max'."
            )

        pareto_front = find_pareto_front(normalized_df.values)
        hypervolume = calc_hypervolume(pareto_front, np.ones(len(objectives)))
        # print(f"{algorithm} {seed} {hypervolume}")
        hypervolume_list.append(hypervolume)

    return np.mean(hypervolume_list)


def calculate_hypervolumes(
    algorithm, workload, global_stats1, global_stats2, normalization_type="min-max"
):
    """
    Calculate hypervolumes for a given algorithm across all seeds.

    Parameters:
    global_stats1: Global mean or min of all objectives (depending on normalization_type)
    global_stats2: Global std or max of all objectives (depending on normalization_type)
    normalization_type: 'min-max' or 'mean' for different types of normalization
    """
    hypervolume_list = []
    for seed in seed_list:
        df = load_data(workload, algorithm, seed)

        if normalization_type == "mean":
            normalized_df = (df[objectives] - global_stats1) / global_stats2
        elif normalization_type == "min-max":
            normalized_df = (df[objectives] - global_stats1) / (global_stats2 - global_stats1)
        else:
            raise ValueError("Unsupported normalization type. Choose 'mean' or 'min-max'.")

        pareto_front = find_pareto_front(normalized_df.values)
        hypervolume = calc_hypervolume(pareto_front, np.ones(len(objectives)))
        hypervolume_list.append(hypervolume)

    return hypervolume_list

def analyze_and_compare_algorithms(workload_results):
    analysis_results = {}

    for workload, algorithms in workload_results.items():
        workload_analysis = {
            'means': {},
            'std_devs': {},
            'significance': {}
        }

        # 计算每种算法的平均超体积和标准差，并找出最佳算法
        best_algorithm = None
        best_mean_hv = -float('inf')
        for algorithm, hypervolumes in algorithms.items():
            mean_hv = np.mean(hypervolumes)
            workload_analysis['means'][algorithm] = mean_hv
            workload_analysis['std_devs'][algorithm] = np.std(hypervolumes)

            if mean_hv > best_mean_hv:
                best_mean_hv = mean_hv
                best_algorithm = algorithm

        # 对每个算法进行显著性检验，只与最佳算法比较
        for algorithm, hypervolumes in algorithms.items():
            if algorithm != best_algorithm:
                stat, p_value = scipy.stats.mannwhitneyu(algorithms[best_algorithm], hypervolumes)
                comparison_key = f"{algorithm} vs {best_algorithm}"
                workload_analysis['significance'][comparison_key] = ('+' if p_value < 0.05 else '-')
                
        # # 进行算法间的显著性检验
        # algorithm_names = list(algorithms.keys())
        # for i in range(len(algorithm_names)):
        #     for j in range(i+1, len(algorithm_names)):
        #         hypervolumes1 = algorithms[algorithm_names[i]]
        #         hypervolumes2 = algorithms[algorithm_names[j]]
        #         stat, p_value = scipy.stats.mannwhitneyu(hypervolumes1, hypervolumes2)
        #         comparison_key = f"{algorithm_names[i]} vs {algorithm_names[j]}"
        #         workload_analysis['significance'][comparison_key] = ('+' if p_value < 0.05 else '-')

        analysis_results[workload] = workload_analysis

    return analysis_results

def matrix_to_latex(analysis_results, caption):
    latex_code = []

    # 添加文档类和宏包
    latex_code.extend([
        "\\documentclass{article}",
        "\\usepackage{geometry}",
        "\\geometry{a4paper, margin=1in}",
        "\\usepackage{graphicx}",
        "\\usepackage{colortbl}",
        "\\usepackage{booktabs}",
        "\\usepackage{threeparttable}",
        "\\usepackage{caption}",
        "\\usepackage{xcolor}",
        "\\pagestyle{empty}",
        "\\begin{document}",
        "\\begin{table*}[t!]",
        "    \\scriptsize",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        "    \\resizebox{1.0\\textwidth}{!}{",
        "    \\begin{tabular}{c|" + "".join(["c"] * len(analysis_results)) + "}",
        "        \\hline"
    ])

    # 确定算法列表
    algorithms = list(analysis_results[next(iter(analysis_results))]['means'].keys())

    # 添加列名（每个算法一个列）
    col_header = " & ".join([""] + [f"\\texttt{{{algorithm}}}" for algorithm in algorithms]) + " \\\\"
    latex_code.append("        " + col_header)
    latex_code.append("        \\hline")

    # 添加行
    for workload in analysis_results.keys():
        row_data = [workload]
        best_algorithm = max(analysis_results[workload]['means'], key=analysis_results[workload]['means'].get)
        for algorithm in analysis_results[workload]['means'].keys():
            mean = analysis_results[workload]['means'][algorithm]
            std_dev = analysis_results[workload]['std_devs'][algorithm]
            significance_mark = ""

            if algorithm != best_algorithm:
                for other_algorithm, sig_value in analysis_results[workload]['significance'].items():
                    if algorithm in other_algorithm and sig_value == '+':
                        significance_mark = "$^\\dagger$"
                        break

            if algorithm == best_algorithm:
                row_data.append(f"\\cellcolor[rgb]{{.682, .667, .667}}\\textbf{{{mean:.3f} (±{std_dev:.3f})}}{significance_mark}")
            else:
                row_data.append(f"{mean:.3f} (±{std_dev:.3f}){significance_mark}")

        latex_code.append("        " + " & ".join(row_data) + " \\\\")

    # 添加表注
    latex_code.extend([
        "        \\hline",
        "    \\end{tabular}",
        "    }",
        "    \\begin{tablenotes}",
        "        \\tiny",
        "        \\item $^\\dagger$ indicates that the best algorithm is significantly better than the other one according to the Wilcoxon signed-rank test at a 5\\% significance level."
        "    \\end{tablenotes}",
        "\\end{table*}%",
        "\\end{document}"
    ])
    
        # latex_code.append("        " + " & ".join(row_data)
                          
    # # 添加列名
    # col_header = " & ".join([""] + list(analysis_results.keys())) + " \\\\"
    # latex_code.append("        " + col_header)
    # latex_code.append("        \\hline")

    # # 添加行
    # for algorithm in analysis_results[next(iter(analysis_results))]['means'].keys():
    #     row_data = [f"\\texttt{{{algorithm}}}"]
    #     for workload, results in analysis_results.items():
    #         mean = results['means'][algorithm]
    #         std_dev = results['std_devs'][algorithm]
    #         significance_mark = ""

    #         for other_algorithm, sig_value in results['significance'].items():
    #             if algorithm in other_algorithm and sig_value == '+':
    #                 significance_mark = "$^\\dagger$"
    #                 break

    #         row_data.append(f"{mean:.3f} (±{std_dev:.3f}){significance_mark}")
    #     latex_code.append("        " + " & ".join(row_data) + " \\\\")
        
    return "\n".join(latex_code)



def load_workloads():
    file_path = package_path / "demo" / "comparison" / f"features_by_workload_{target}.json"
    with open(file_path, "r") as f:
        return json.load(f).keys()


if __name__ == "__main__":
    workloads = load_workloads()

    workloads = list(workloads)
    workloads.sort()
    workloads = workloads[:14]
    
    # workloads = [
    #     "cbench-automotive-qsort1",
    #     "cbench-automotive-susan-e",
    #     "cbench-network-patricia",
    #     "cbench-automotive-bitcount",
    #     "cbench-bzip2",
    #     "cbench-telecom-adpcm-d",
    #     "cbench-office-stringsearch2",
    #     "cbench-security-rijndael",
    #     "cbench-security-sha",
    # ]

    workload_results = {}
    for workload in workloads:
        print(f"Processing workload: {workload}")
        all_data, global_mean, global_std = collect_all_data(workload)
        global_max = all_data.max(axis=0)
        global_min = all_data.min(axis=0)

        algorithm_results = {}
        for algorithm in algorithm_list:
            hypervolumes = calculate_hypervolumes(
                algorithm,
                workload,
                global_min,
                global_max,
                normalization_type="min-max",
            )
            algorithm_results[algorithm] = hypervolumes
        
        # Remove the prefix from the workload name ""
        workload_short_name = workload[7:]
        workload_results[workload_short_name] = algorithm_results

    final_results = analyze_and_compare_algorithms(workload_results)
    print(final_results)

    caption = "Perfomance Comparison of Algorithms"
    latex_table = matrix_to_latex(final_results, caption)

    latex_table_path = "latex_table.tex"
    with open(latex_table_path, 'w') as file:
        file.write(latex_table)
        
    # for workload in workloads:
    #     print(workload)
    #     all_data, global_mean, global_std = collect_all_data(workload)
    #     global_max = all_data.max(axis=0)
    #     global_min = all_data.min(axis=0)

    #     hv_list = []
    #     for algorithm in algorithm_list:
    #         mean_hypervolume = calculate_mean_hypervolume(
    #             algorithm,
    #             workload,
    #             global_min,
    #             global_max,
    #             normalization_type="min-max",
    #         )
    #         hv_list.append((algorithm, mean_hypervolume))

    #     # Sort by hypervolume
    #     hv_list.sort(key=lambda x: x[1], reverse=True)

    #     print(hv_list)
    #     print()
