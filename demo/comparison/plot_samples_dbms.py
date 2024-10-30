import sys
from pathlib import Path

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from transopt.utils.pareto import calc_hypervolume, find_pareto_front
from transopt.utils.plot import plot3D

results_path = package_path / "experiment_results"
dbms_samples_path = results_path / "dbms_samples"

objectives = ["throughput", "latency"]


def load_and_prepare_data(file_path):
    """
    Loads JSON data and prepares a DataFrame.
    """
    # print(f"Loading data from {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        if "1" in data:
            data = data["1"]

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)
    # print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())

    for obj in objectives:
        if obj == "latency":
            df_combined = df_combined[df_combined[obj] > 0]  # Discard latency less than 0
        else:
            df_combined = df_combined[df_combined[obj] != 1e10]  # Original condition

    # print(f"Loaded {len(df_combined)} data points, removed {len(df_input) - len(df_combined)} duplicates")
    # print()
    return df_combined

def load_data(workload):
    result_file = dbms_samples_path / f"DBMS_{workload}.json"
    df = load_and_prepare_data(result_file)
    return df


def plot_pareto_front(workload):
    df = load_data(workload)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    _, pareto_indices = find_pareto_front(df_normalized[objectives].values, return_index=True, obj_type=['max', 'min'])
    
    # Retrieve Pareto points
    points = df_normalized.iloc[pareto_indices][objectives]
    
    plt.figure()
    plt.title(f"Pareto Front for {workload}")
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.scatter(points[objectives[0]], points[objectives[1]], c='b', marker='o')
     
    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"dbms_pf_{workload}.png"
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory
    
    
def plot_all(workload):
    df = load_data(workload)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    plt.figure()
    plt.title(f"All samples for {workload}")
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.scatter(df_normalized[objectives[0]], df_normalized[objectives[1]], c='b', marker='o')
    
    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"dbms_all_{workload}.png"
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory
    
if __name__ == "__main__":
    workloads_dbms = [
        "sibench",
        "smallbank",
        "tatp",
        "tpcc",
        "twitter",
        "voter"
    ] 
        
    for workload in workloads_dbms:
        plot_pareto_front(workload)
        plot_all(workload)