import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import dcor
import cmasher as cmr
import json
import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))


pngs_path = package_path / "demo/importances/pngs"

mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.serif"] = ["Times New Roman"]

def generate_grid_plot_combine(dcor_values_dicts):
    # 创建一个图和三个子图（对于三个数据集）
    fig, axs = plt.subplots(1, 3, figsize=(25, 10), constrained_layout=True)
    
    for ax, dcor_values_dict in zip(axs, dcor_values_dicts,):
        workloads = list(dcor_values_dict.keys())
        objective_pairs = list(dcor_values_dict[workloads[0]].keys())
    
        dcor_matrix = np.zeros((len(workloads), len(objective_pairs)))
    
        for i, workload in enumerate(workloads):
            for j, pair in enumerate(objective_pairs):
                dcor_matrix[i, j] = dcor_values_dict[workload].get(pair, 0)
    
        im = ax.imshow(dcor_matrix, cmap=cmr.prinsenvlag_r, interpolation="nearest", vmin=-0.6, vmax=0.6)
    
        ax.set_yticks(range(len(workloads)))
        ax.set_yticklabels(workloads, fontsize=36)
        ax.set_xticks(range(len(objective_pairs)))
        ax.set_xticklabels(objective_pairs, rotation=0, fontsize=36)
    
    cbar = fig.colorbar(im, ax=axs, shrink=1, location='right')
    cbar.ax.tick_params(labelsize=36)  # 设置 color bar 字体大小
    plt.savefig(pngs_path / "combined_heatmap.pdf", format="pdf", bbox_inches="tight")
    


def generate_grid_plot(dcor_values_dict, file_name):
    workloads = list(dcor_values_dict.keys())
    objective_pairs = list(dcor_values_dict[workloads[0]].keys())

    dcor_matrix = np.zeros((len(workloads), len(objective_pairs)))

    for i, workload in enumerate(workloads):
        for j, pair in enumerate(objective_pairs):
            dcor_matrix[i, j] = dcor_values_dict[workload].get(pair, 0)

    plt.figure(figsize=(12, 10))

    plt.imshow(dcor_matrix, cmap=cmr.prinsenvlag_r, interpolation="nearest", vmin=-0.6, vmax=0.6)
    colorbar = plt.colorbar(shrink=1)
    colorbar.ax.tick_params(labelsize=18)

    plt.yticks(range(len(workloads)), workloads, fontsize=18)
    plt.xticks(range(len(objective_pairs)),
               objective_pairs, rotation=0, fontsize=18)

    plt.savefig(pngs_path / f"{file_name}_heatmap.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    gcc_dcor_values_dict = {
        "adpcm-c": {"ET-CS": 0.5096407431062894, "ET-CT": 0.02156206023915185, "CS-CT": 0.028167304817522342},
        "qsort1": {"ET-CS": 0.24458686101114566, "ET-CT": 0.4484640731112793, "CS-CT": 0.1319462835609861},
        "patricia": {"ET-CS": 0.3136478783871287, "ET-CT": 0.11344940640932157, "CS-CT": 0.23628956882620056},
        "gsm": {"ET-CS": 0.3199972317712137, "ET-CT": 0.19506712567511303, "CS-CT": 0.08086715789520826},
        "tiff2rgba": {"ET-CS": 0.19036475515437773, "ET-CT": 0.18802272660380803, "CS-CT": 0.09256748900522595},
        "susan-e": {"ET-CS": 0.1362765512460971, "ET-CT": 0.36116979864249992, "CS-CT": 0.05943189644484737},
    }
    
    mysql_dcor_values_dict = {
        "SiBench": {"T-L": 0.4, "T-CU": 0.05, "L-CU": -0.13},
        "Voter": {"T-L": 0.2, "T-CU": 0.03, "L-CU": -0.14},
        "SmallBank": {"T-L": 0.6, "T-CU": 0.24, "L-CU": -0.35},
        "Twitter": {"T-L": 0.25, "T-CU": 0.43, "L-CU": -0.02},
        "TATP": {"T-L": 0.14, "T-CU": 0.05, "L-CU": -0.13},
        "TPC-C": {"T-L": 0.23, "T-CU": 0.16, "L-CU": -0.34},
    }

    hadoop_dcor_values_dict = {
        "WordCount": {"ET-CU": 0.5, "ET-MU": 0.14, "CU-MU": 0.03},
        "KMeans": {"ET-CU": 0.6, "ET-MU": 0.05, "CU-MU": 0.02},
        "Bayes": {"ET-CU": 0.4, "ET-MU": 0.23, "CU-MU": 0.4},
        "NWeight": {"ET-CU": 0.5, "ET-MU": 0.2, "CU-MU": 0.4},
        "PageRank": {"ET-CU": 0.13, "ET-MU": 0.35, "CU-MU": 0.16},
        "TeraSort": {"ET-CU": 0.4, "ET-MU": 0.16, "CU-MU": 0.15},
    }

    # generate_grid_plot(gcc_dcor_values_dict, "gcc")
    # generate_grid_plot(mysql_dcor_values_dict, "mysql")
    # generate_grid_plot(hadoop_dcor_values_dict, "hadoop")
    
    generate_grid_plot_combine([gcc_dcor_values_dict, mysql_dcor_values_dict, hadoop_dcor_values_dict])