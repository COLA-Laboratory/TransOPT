import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import json
from pathlib import Path

import cmasher as cmr
import dcor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

target = "gcc"
results_path = package_path / "experiment_results"
gcc_comparsion_path = results_path / "gcc_archive_new"
gcc_samples_path = results_path / "gcc_samples"
llvm_comparsion_path = results_path / "llvm_archive"
llvm_samples_path = results_path / "llvm_samples"

pngs_path = package_path / "demo/importances/pngs"

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']

def load_and_prepare_data(file_path, objectives):
    """
    Loads JSON data and prepares a DataFrame.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    # print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())
    # print(f"Removed {len(df_combined) - len(df_input)} duplicates")

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]
    print(f"Loaded {len(df_combined)} data points after removing extreme values")
    return df_combined


def cal_dcor(df, objectives):
    """
    Calculate the distance correlation for each pair of objectives using the dcor library.
    """
    dcor_results = {}
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            obj1, obj2 = objectives[i], objectives[j]
            dcor_value = dcor.distance_correlation(df[obj1], df[obj2])
            dcor_results[f"{obj1}-{obj2}"] = dcor_value
    return dcor_results


def cal_spearman_corr(df, objectives):
    """
    Calculate the Spearman correlation for each pair of objectives.
    """

    corr_matrix = df[objectives].corr(method="spearman")

    spearman_results = {}
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            obj1, obj2 = objectives[i], objectives[j]
            corr_value = corr_matrix.at[obj1, obj2]
            spearman_results[f"{obj1}-{obj2}"] = corr_value

    return spearman_results


def cal_pearson_corr(df, objectives):
    """
    Calculate the Pearson correlation matrix for the given objectives and extract
    pairwise correlations from it.
    """
    corr_matrix = df[objectives].corr(method="pearson")

    pearson_results = {}
    for i in range(len(objectives)):
        for j in range(i + 1, len(objectives)):
            obj1, obj2 = objectives[i], objectives[j]
            corr_value = corr_matrix.at[obj1, obj2]
            pearson_results[f"{obj1}-{obj2}"] = corr_value

    return pearson_results


def generate_grid_plot(dcor_values_dict):
    workloads = list(dcor_values_dict.keys())
    objective_pairs = list(dcor_values_dict[workloads[0]].keys())

    dcor_matrix = np.zeros((len(workloads), len(objective_pairs)))

    for i, workload in enumerate(workloads):
        for j, pair in enumerate(objective_pairs):
            dcor_matrix[i, j] = dcor_values_dict[workload].get(pair, 0)

    plt.figure(figsize=(12, 10))  # Increase the height of the heatmap

    color_sequence = ["#edf8fb", "#ccece6", "#99d8c9", "#66c2a4", "#2ca25f", "#006d2c"]

    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", color_sequence)
    
    plt.imshow(dcor_matrix, cmap=cmr.fusion_r, interpolation="nearest")
    colorbar =plt.colorbar(shrink=0.57)  # Reduce the size of the colorbar
    
    # set font size of colorbar
    colorbar.ax.tick_params(labelsize=18)

    objective_pairs_short = ['ET-CS', 'ET-CT', 'CS-CT']

    plt.yticks(range(len(workloads)), workloads, fontsize=18)  # Adjust labels as needed
    plt.xticks(range(len(objective_pairs)), ['ET-CS', 'ET-CT', 'CS-CT'], rotation=45, fontsize=18)  # Rotation for better label visibility
    
    # plt.yticks(range(len(objective_pairs)), objective_pairs_short, fontsize=18)
    # plt.xticks(range(len(workloads)), ['1', '2', '3', '4', '5'], fontsize=18)

    plt.savefig(pngs_path / f"heatmap.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    gcc_workloads = [
        "cbench-consumer-tiff2rgba",
        "cbench-security-rijndael",
        "cbench-security-pgp",
        "cbench-automotive-qsort1",
        "cbench-automotive-susan-e",
        "cbench-consumer-jpeg-d",
        "cbench-security-sha",
        "cbench-telecom-adpcm-c",
        "cbench-telecom-adpcm-d",
        "cbench-telecom-gsm",
        "cbench-telecom-crc32",
        "cbench-consumer-tiff2bw",
        "cbench-consumer-mad",
        "cbench-network-patricia",
    ]

    objectives = ["execution_time", "file_size", "compilation_time"]

    # dcor_values_dict = {}
    # spearman_corr_dict = {}
    # pearson_corr_dict = {}
    # for workload in gcc_workloads:
    #     file_path = gcc_samples_path / f"GCC_{workload}.json"
    #     df = load_and_prepare_data(file_path, objectives)
    #     dcor_values = cal_dcor(df, objectives)
    #     spearman_corr = cal_spearman_corr(df, objectives)
    #     pearson_corr = cal_pearson_corr(df, objectives)
    #     print(f"dCor values for {workload}: {dcor_values}")
    #     print(f"Spearman correlation for {workload}: {spearman_corr}")

    #     dcor_values_dict[workload] = dcor_values
    #     spearman_corr_dict[workload] = spearman_corr
    #     pearson_corr_dict[workload] = pearson_corr

    # with open(pngs_path / "dcor_values_dict.json", "w") as f:
    #     json.dump(dcor_values_dict, f)

    # with open(pngs_path / "spearman_corr_dict.json", "w") as f:
    #     json.dump(spearman_corr_dict, f)

    # with open(pngs_path / "pearson_corr_dict.json", "w") as f:
    #     json.dump(pearson_corr_dict, f)

    # with open(pngs_path / "dcor_values_dict.json", "r") as f:
    #     dcor_values_dict = json.load(f)

    # with open(pngs_path / "spearman_corr_dict.json", "r") as f:
    #     spearman_corr_dict = json.load(f)

    # with open(pngs_path / "pearson_corr_dict.json", "r") as f:
    #     pearson_corr_dict = json.load(f)
    
    dcor_values_dict = {
        "telecom-adpcm-c": {
            "execution_time-file_size": 0.5096407431062894,
            "execution_time-compilation_time": 0.02156206023915185,
            "file_size-compilation_time": 0.028167304817522342,
        },
        "automotive-qsort1": {
            "execution_time-file_size": 0.24458686101114566,
            "execution_time-compilation_time": 0.4484640731112793,
            "file_size-compilation_time": 0.1319462835609861,
        },
        "network-patricia": {
            "execution_time-file_size": 0.3136478783871287,
            "execution_time-compilation_time": 0.11344940640932157,
            "file_size-compilation_time": 0.23628956882620056,
        },
        "telecom-gsm": {
            "execution_time-file_size": 0.3199972317712137,
            "execution_time-compilation_time": 0.19506712567511303,
            "file_size-compilation_time": 0.08086715789520826,
        },
        "consumer-tiff2rgba": {
            "execution_time-file_size": 0.19036475515437773,
            "execution_time-compilation_time": 0.18802272660380803,
            "file_size-compilation_time": 0.09256748900522595,
        },
    }

    generate_grid_plot(dcor_values_dict)
