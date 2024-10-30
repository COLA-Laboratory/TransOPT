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
# import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from transopt.utils.pareto import calc_hypervolume, find_pareto_front
from transopt.utils.plot import plot3D

target = "gcc"
results_path = package_path / "experiment_results"
gcc_results_path = results_path / "gcc_comparsion"
gcc_samples_path = results_path / "gcc_samples"
llvm_results = results_path / "llvm_comparsion"
llvm_samples_path = results_path / "llvm_samples"

dbms_samples_path = results_path / "dbms_samples"

algorithm_list = ["ParEGO", "SMSEGO", "MoeadEGO", "CauMO"]
# algorithm_list = ["SMSEGO"]
# objectives = ["execution_time", "file_size", "compilation_time"]
objectives = ["latency", "throughput"]
seed_list = [65535, 65536, 65537, 65538, 65539]


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
    print(f"Loaded {len(df_combined)} data points")

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]

    print(f"Loaded {len(df_combined)} data points, removed {len(df_input) - len(df_combined)} duplicates")
    print()
    return df_combined

def load_data(workload, algorithm, seed):
    if target == "llvm":
        result_file = llvm_results / f"llvm_{workload}" / algorithm / f"{seed}_KB.json"
    else:
        result_file = gcc_results_path / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
    df = load_and_prepare_data(result_file)
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


def dynamic_plot(workload, algorithm, seed):
    """
    Dynamically plot the three objectives for a given workload and algorithm for a specific seed.
    """
    # Collect all data to understand the range
    all_data, global_mean, global_std = collect_all_data(workload)
    global_min = np.min(all_data, axis=0)
    global_max = np.max(all_data, axis=0)
   
    # Load data for the specific seed
    df = load_data(workload, algorithm, seed)
    
    # Normalize data (Min-Max normalization)
    df_normalized = (df[objectives] - global_min) / (global_max - global_min)
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Dynamic Plot for {workload} - {algorithm} - Seed {seed}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])

    # Initialize two scatter plots: one for all previous points, one for the new point
    previous_points = ax.scatter([], [], [], c='b', marker='o')  # all previous points in blue
    current_point = ax.scatter([], [], [], c='r', marker='o')  # current point in red
    
    def init():
        previous_points._offsets3d = ([], [], [])
        current_point._offsets3d = ([], [], [])
        return previous_points, current_point

    def update(frame):
        # Add all previous points up to the current frame
        previous_points._offsets3d = (df_normalized.iloc[:frame][objectives[0]].values,
                                      df_normalized.iloc[:frame][objectives[1]].values,
                                      df_normalized.iloc[:frame][objectives[2]].values)

        # Add the current point (latest one in the sequence)
        current_point._offsets3d = (df_normalized.iloc[frame:frame+1][objectives[0]].values,
                                    df_normalized.iloc[frame:frame+1][objectives[1]].values,
                                    df_normalized.iloc[frame:frame+1][objectives[2]].values)
        return previous_points, current_point
    
    frames = len(df)
    ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)
    
    # Save the plot to a file
    gif_path = package_path / "demo" / "comparison" / "gifs" / f"{target}_{algorithm}_{workload}_{seed}.gif"
    ani.save(gif_path, writer='imagemagick')
    plt.close(fig)  # Close the plot to free memory


# def dynamic_plot_html(workload, algorithm, seed):
    """
    Dynamically plot the three objectives for a given workload and algorithm for a specific seed using Plotly.
    """
    # Collect all data to understand the range
    all_data, global_mean, global_std = collect_all_data(workload)
    global_min = np.min(all_data, axis=0)
    global_max = np.max(all_data, axis=0)
   
    # Load data for the specific seed
    df = load_data(workload, algorithm, seed)
    
    # Normalize data (Min-Max normalization)
    df_normalized = (df[objectives] - global_min) / (global_max - global_min)
    df_normalized = df_normalized

    pareto_front, pareto_front_index = find_pareto_front(df_normalized.values, return_index=True)
    df_normalized = df_normalized.iloc[pareto_front_index]
    
    # Create traces for previous and current points
    trace1 = go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=5, color='blue'))
    trace2 = go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=5, color='red'))

    # Combine traces into a data list
    data = [trace1, trace2]

    # Create the layout of the plot
    layout = go.Layout(
        title=f"Dynamic Plot for {workload} - {algorithm} - Seed {seed}",
        scene=dict(
            xaxis=dict(title=objectives[0], range=[0, 1]),
            yaxis=dict(title=objectives[1], range=[0, 1]),
            zaxis=dict(title=objectives[2], range=[0, 1])
        )
    )
    
    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Create frames for the animation
    frames = []
    for t in range(len(df)):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=df_normalized.iloc[:t+1][objectives[0]].values,
                    y=df_normalized.iloc[:t+1][objectives[1]].values,
                    z=df_normalized.iloc[:t+1][objectives[2]].values,
                    mode='markers',
                    marker=dict(size=5, color='blue')
                ),
                go.Scatter3d(
                    x=df_normalized.iloc[t:t+1][objectives[0]].values,
                    y=df_normalized.iloc[t:t+1][objectives[1]].values,
                    z=df_normalized.iloc[t:t+1][objectives[2]].values,
                    mode='markers',
                    marker=dict(size=5, color='red')
                )
            ]
        )
        frames.append(frame)

    fig.frames = frames
   
    prev_frame_button = dict(
        args=[None, {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        label='Previous',
        method='animate'
    )

    next_frame_button = dict(
        args=[None, {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        label='Next',
        method='animate'
    )

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0,
            x=1.05,
            xanchor='right',
            yanchor='top',
            pad=dict(t=0, r=10),
            buttons=[prev_frame_button, next_frame_button]
        )]
    )
 
    # fig.update_layout(sliders=sliders)

    # Save the plot to HTML file
    html_path = package_path / "demo" / "comparison" / "htmls" / f"dynamic_{target}_{algorithm}_{workload}_{seed}.html"
    fig.write_html(str(html_path))


def save_individual_frames(workload, algorithm, seed):
    """
    Save each frame of the three objectives as a separate plot for a given workload, algorithm, and seed.
    """
    # Load data for the specific seed
    df = load_data(workload, algorithm, seed)

    # Ensure the directory for saving frames exists
    frames_dir = package_path / "demo" / "comparison" / "frames" / f"{algorithm}_{workload}_{seed}"
    os.makedirs(frames_dir, exist_ok=True)
    
    for idx in range(len(df)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Add data points from the DataFrame row by row
        x, y, z = df.iloc[idx][objectives[0]], df.iloc[idx][objectives[1]], df.iloc[idx][objectives[2]]
        
        # Plot and customize as needed
        ax.scatter(x, y, z, color='r')
        ax.set_title(f"Frame {idx} for {workload} - {algorithm} - Seed {seed}")
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        
        # Save the plot as a file
        frame_file = frames_dir / f"frame_{idx:04d}.png"
        plt.savefig(frame_file)
        plt.close(fig)  # Close the plot to free memory
        

def load_workloads():
    file_path = package_path / "demo" / "comparison" / f"features_by_workload_{target}.json"
    with open(file_path, "r") as f:
        return json.load(f).keys()


# def plot_pareto_front_html(workload):
    # df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    df = load_and_prepare_data(llvm_samples_path / f"LLVM_{workload}.json")
    df_normalized = (df - df.min()) / (df.max() - df.min())
    _, pareto_indices = find_pareto_front(df_normalized[objectives].values, return_index=True)
    
    # Retrieve Pareto points
    pareto_points = df_normalized.iloc[pareto_indices][objectives]
    
    # Create a 3D scatter plot using plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=pareto_points[objectives[0]],
        y=pareto_points[objectives[1]],
        z=pareto_points[objectives[2]],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',  # set color to blue
            opacity=0.8
        )
    )])

    # Update the layout
    fig.update_layout(
        title=f"Pareto Front for {workload}",
        scene=dict(
            xaxis_title=objectives[0],
            yaxis_title=objectives[1],
            zaxis_title=objectives[2]
        )
    )

    # Define the path for HTML file
    html_path = package_path / "demo" / "comparison" / "htmls"
    # Ensure the directory exists
    html_path.mkdir(parents=True, exist_ok=True)

    # Save the plot as an HTML file
    fig.write_html(str(html_path / f"{target}_pareto_front_{workload}.html"))


def plot_pareto_front(workload):
    # df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    # df = load_and_prepare_data(llvm_samples_path / f"LLVM_{workload}.json")
    df = load_data(workload, "ParEGO", 65535)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    _, pareto_indices = find_pareto_front(df_normalized[objectives].values, return_index=True)
    
    # Retrieve Pareto points
    points = df_normalized.iloc[pareto_indices][objectives]
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Pareto Front for {workload}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    
    # # Scatter plot for Pareto front
    # points = df_normalized[objectives]
    
    # Convert Series to NumPy array before plotting
    x_values = points[objectives[0]].values
    y_values = points[objectives[1]].values
    z_values = points[objectives[2]].values

    ax.scatter(x_values, y_values, z_values, c='b', marker='o')

    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"{target}_pf_{workload}.png"
    plt.savefig(file_path)
    plt.close(fig)  # Close the plot to free memory
    
    
def plot_all(workload, algorithm=""):
    # df = load_and_prepare_data(llvm_samples_path / f"LLVM_{workload}.json")
    # df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    # df = load_data(workload, algorithm, 65535)
    df = load_and_prepare_data(dbms_samples_path / f"DBMS_{workload}.json")
    df_normalized = (df - df.min()) / (df.max() - df.min())
    df_normalized = df
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"All samples for {workload}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2])
    
    # Scatter plot for Pareto front
    points = df_normalized[objectives]
    
    # Convert Series to NumPy array before plotting
    x_values = points[objectives[0]].values
    y_values = points[objectives[1]].values
    z_values = points[objectives[2]].values

    ax.scatter(x_values, y_values, z_values, c='b', marker='o')

    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"{target}_{workload}.png"
    plt.savefig(file_path)
    plt.close(fig)  # Close the plot to free memory
    
# 2D plot all
def plot_all_2d(workload, algorithm=""):
    # df = load_and_prepare_data(llvm_samples_path / f"LLVM_{workload}.json")
    # df = load_and_prepare_data(gcc_samples_path / f"GCC_{workload}.json")
    # df = load_data(workload, algorithm, 65535)
    df = load_and_prepare_data(dbms_samples_path / f"DBMS_{workload}.json")
    df_normalized = (df - df.min()) / (df.max() - df.min())
    df_normalized = df
    
    # Create a 2D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"All samples for {workload}")
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    
    # Scatter plot for Pareto front
    points = df_normalized[objectives]
    
    # Convert Series to NumPy array before plotting
    x_values = points[objectives[0]].values
    y_values = points[objectives[1]].values

    ax.scatter(x_values, y_values, c='b', marker='o')

    # Save the plot as a file
    file_path = package_path / "demo" / "comparison" / "pngs" / f"{target}_{workload}.png"
    plt.savefig(file_path)
    plt.close(fig)  # Close the plot to free memory

if __name__ == "__main__":
    # workloads = load_workloads()
 
    # workloads = [
    #     "cbench-consumer-tiff2bw",
    #     "cbench-security-rijndael",
        
    #     "cbench-security-pgp", 
    #     "polybench-cholesky",
    #     "cbench-consumer-tiff2rgba",
    #     "cbench-network-patricia",
    #     # "cbench-automotive-susan-e",
    #     # "polybench-symm",
    #     "cbench-consumer-mad",
    #     "polybench-lu"
    # ]
    
    # workloads = [
    #     "cbench-security-sha",
    #     "cbench-telecom-adpcm-c",
    #     ""
    # ]
    
    # LLVM
    workloads_improved = [
        "cbench-telecom-gsm",
        "cbench-automotive-qsort1",
        "cbench-automotive-susan-e",
        "cbench-consumer-tiff2rgba",
        "cbench-network-patricia",
        "cbench-consumer-tiff2bw",
        "cbench-consumer-jpeg-d",
        "cbench-telecom-adpcm-c",
        "cbench-security-rijndael",
        "cbench-security-sha",
    ]
        
        
    workloads_mysql = [
        "sibench",
        "smallbank",
        "voter",
        "tatp",
        "tpcc",
        "twitter",
    ]
    seed = 65535  # Example seed
    
    # Plot sampling results
    for workload in workloads_mysql:
        # for algorithm in algorithm_list:
        plot_all_2d(workload)
        # plot_pareto_front(workload)
    
    # for algorithm in algorithm_list:
    #     # dynamic_plot_html("cbench-consumer-tiff2bw", algorithm, seed)
    #     for workload in workloads:
    #         dynamic_plot_html(workload, algorithm, seed)
            # dynamic_plot(workload, algorithm, seed)
        # save_individual_frames(workload, algorithm, objectives, seed)