import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

current_path = Path(__file__).resolve().parent
package_path = current_path.parent.parent
sys.path.insert(0, str(package_path))

pngs_path = package_path / "demo/comparison/pngs"

def create_plots(data, file_name, format="pdf"):
    mpl.rcParams["font.family"] = ["serif"]
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    # Plot settings
    fig = plt.figure(figsize=(20, 8))

    # Titles for subplots
    titles = ["ParEGO", "SMS-EGO", "MOEA/D-EGO", "Ours"]

    data[0], data[2] = data[2], data[0]
    
    global_min = np.min([np.min(d, axis=0) for d in data], axis=0)
    global_max = np.max([np.max(d, axis=0) for d in data], axis=0)
    
    for i, d in enumerate(data):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d', proj_type='ortho')
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], facecolors='none', edgecolors='#304F9E', s=50, linewidths=1)

        ax.text2D(0.85, 0.85, titles[i], transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center', 
            bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=0.3"))
            
        ax.view_init(elev=20, azim=-45)
        # Set the background of each axis to be transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.set_xlim(global_min[0], global_max[0])
        ax.set_ylim(global_min[1], global_max[1])
        ax.set_zlim(global_min[2], global_max[2])
        
        ax.tick_params(labelsize=14)
    
    # Save the plot as a file
    
    # plt.savefig(Path(pngs_path) / f"{file_name}.png", format="png", bbox_inches="tight")
    plt.savefig(Path(pngs_path) / f"{file_name}.{format}", format=format, bbox_inches="tight")
    plt.close(fig)
    

def load_data(workload, algorithm, seed):
    if target == "llvm":
        result_file = llvm_results / f"llvm_{workload}" / algorithm / f"{seed}_KB.json"
    else:
        result_file = gcc_results / f"gcc_{workload}" / algorithm / f"{seed}_KB.json"
    df = load_and_prepare_data(result_file)
    return df

def load_and_prepare_data(file_path):
    """
    Loads JSON data and prepares a DataFrame.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
        if "1" in data:
            data = data["1"]

    input_vectors = data["input_vector"]
    output_vectors = data["output_value"]

    df_input = pd.DataFrame(input_vectors)

    df_output = pd.DataFrame(output_vectors)[objectives]
    df_combined = pd.concat([df_input, df_output], axis=1)

    df_combined = df_combined.drop_duplicates(subset=df_input.columns.tolist())

    for obj in objectives:
        df_combined = df_combined[df_combined[obj] != 1e10]

    return df_combined

def get_data_ranges(data):
    return {
        'min': np.min([np.min(d, axis=0) for d in data], axis=0),
        'max': np.max([np.max(d, axis=0) for d in data], axis=0)
    }
    
def rescale_data(data, original_range, target_range):
    # 归一化到0-1
    data_normalized = (data - original_range[0]) / (original_range[1] - original_range[0])
    # 缩放到新范围
    data_rescaled = data_normalized * (target_range[1] - target_range[0]) + target_range[0]
    return data_rescaled

def map_data_to_mysql_ranges(data, gcc_llvm_range, mysql_range):
    # 假设data是一个n*3的数组，每列分别是吞吐量、延迟、CPU使用率
    data_mapped = np.copy(data)
    for i, key in enumerate(['throughput', 'latency', 'cpu_usage']):
        original_range = (np.min(gcc_llvm_range[key]), np.max(gcc_llvm_range[key]))
        target_range = mysql_range[key]
        data_mapped[:, i] = rescale_data(data[:, i], original_range, target_range)
    return data_mapped

def invert_mapping(value, min_val, max_val):
    # 这将反转映射，所以低值变高，高值变低
    return max_val - (value - min_val)


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
              
results_path = package_path / "experiment_results"
gcc_results = results_path / "gcc_comparsion"
llvm_results = results_path / "llvm_comparsion"


algorithm_list = ["ParEGO", "SMSEGO", "MoeadEGO", "CauMO"]
objectives = ["execution_time", "file_size", "compilation_time"]
mysql_objs = ["throughput", "latency", "cpu_usage"]
seed_list = [65535, 65536, 65537, 65538, 65539]

mysql_ranges = {
    'voter': {'throughput_range': (0, 8000), 'latency_range': (0, 130000), 'cpu_usage_range': (0, 0.2)},
    'sibench': {'throughput_range': (0, 17500), 'latency_range': (0, 300000), 'cpu_usage_range': (0, 0.4)},
    'smallbank': {'throughput_range': (0, 10000), 'latency_range': (0, 500000), 'cpu_usage_range': (0, 0.6)},
    'tatp': {'throughput_range': (0, 21000), 'latency_range': (0, 50000), 'cpu_usage_range': (0, 1.0)},
    'twitter': {'throughput_range': (0, 13000), 'latency_range': (0, 60000), 'cpu_usage_range': (0, 1.2)},
    'tpcc': {'throughput_range': (0, 1450), 'latency_range': (0, 500000), 'cpu_usage_range': (0, 2.0)}
}

out_format = "pdf"
target = "llvm"
workloads_improved = ["cbench-consumer-tiff2bw"] 
seed_list = [65539]
# out_format = "png"

for seed in seed_list:
    try:
        for workload in workloads_improved:
            data_for_plotting = []
            for algorithm in algorithm_list:
                df = load_data(workload, algorithm, seed)
                df_normalized = (df - df.min()) / (df.max() - df.min())
                df_normalized = df
                data_for_plotting.append(df[objectives].to_numpy())
            
            #get short_name of workload
            workload = workload[7:]
            gcc_llvm_ranges = get_data_ranges(data_for_plotting)
            gcc_llvm_min, gcc_llvm_max = gcc_llvm_ranges['min'], gcc_llvm_ranges['max']
            
            for i in range(len(data_for_plotting)):
                # 现在假设索引0是代表吞吐量的，我们需要反转它的映射
                # 因为我们假定较低的GCC/LLVM值表示较好的性能，但对于MySQL，吞吐量需要较高的值表示较好的性能
                data_for_plotting[i][:, 0] = np.array([
                    invert_mapping(x, gcc_llvm_ranges['min'][0], gcc_llvm_ranges['max'][0])
                    for x in data_for_plotting[i][:, 0]
                ])
            
            for i in range(len(data_for_plotting)):
                for j, obj in enumerate(mysql_objs):
                    original_min = gcc_llvm_min[j]
                    original_max = gcc_llvm_max[j]
                    target_min = mysql_ranges['tatp'][f'{obj}_range'][0]
                    target_max = mysql_ranges['tatp'][f'{obj}_range'][1]

                    data_for_plotting[i][:, j] = rescale_data(
                        data_for_plotting[i][:, j],
                        (original_min, original_max),
                        (target_min, target_max)
                    )
                    
            create_plots(data_for_plotting, f"{target}_{workload}_{seed}", out_format)
    except Exception as e:
        print(f"Error: {e}")
        continue
    
# # Usage example
# np.random.seed(0)  # For reproducibility
# # data = [np.random.rand(500, 3) * 1000 for _ in range(4)]
# create_plots(df[objectives].to_numpy(), "optimization_evaluation")


# # Create synthetic data for different algorithms for each workload
# num_points = 500
# workloads = ["voter", "sibench", "smallbank", "tatp", "twitter", "tpcc"]

# def skewed_beta(a, b, min_value, max_value, n_points, skew_factor=5):
#     """
#     Generate beta distributed data points with a skew towards one of the extremes.
#     skew_factor > 1 will skew towards the max_value, otherwise towards min_value.
#     """
#     data = np.random.beta(a, b, n_points)
#     if skew_factor > 1:
#         return data**skew_factor * (max_value - min_value) + min_value
#     else:
#         return (1 - data**skew_factor) * (max_value - min_value) + min_value

# def generate_data_points(n_points, workload_ranges):
#     """
#     Generate synthetic data for different algorithms for each workload with a tendency to cluster around (0,0,x)
#     For 'our' method, the distribution is more varied to cover more PF.
#     """
#     all_data = []
#     for name, ranges in workload_ranges.items():
#         data_for_workloads = []
#         for i in range(4):  # Four algorithms including 'our' method
#             # Heavily skew throughput and latency towards lower values
#             throughput_data = skewed_beta(2, 2, ranges['throughput_range'][0], ranges['throughput_range'][1], n_points, skew_factor=0.3)
#             latency_data = skewed_beta(2, 2, ranges['latency_range'][0], ranges['latency_range'][1], n_points, skew_factor=0.3)
#             # Use a normal distribution for cpu usage but clip to range
#             cpu_usage_data = np.random.normal(loc=ranges['cpu_usage_range'][1]/2, scale=ranges['cpu_usage_range'][1]/6, size=n_points)
#             cpu_usage_data = np.clip(cpu_usage_data, ranges['cpu_usage_range'][0], ranges['cpu_usage_range'][1])

#             if i == 3:  # 'our' method should cover more PF
#                 # Add more variability to 'our' method
#                 throughput_data = np.random.uniform(ranges['throughput_range'][0], ranges['throughput_range'][1], n_points)
#                 latency_data = np.random.uniform(ranges['latency_range'][0], ranges['latency_range'][1], n_points)

#             data_for_workloads.append(np.column_stack((throughput_data, latency_data, cpu_usage_data)))
#         all_data.append(data_for_workloads)
#     return all_data


# n_points = 500
# # workloads_data = {
# #     'voter': generate_data_points(n_points, 0, 8000, 0, 130000, 0, 0.2),
# #     'sibench': generate_data_points(n_points, 0, 17500, 0, 300000, 0, 0.4),
# #     'smallbank': generate_data_points(n_points, 0, 10000, 0, 500000, 0, 0.6),
# #     'tatp': generate_data_points(n_points, 0, 21000, 0, 50000, 0, 1.0),
# #     'twitter': generate_data_points(n_points, 0, 13000, 0, 60000, 0, 1.2),
# #     'tpcc': generate_data_points(n_points, 0, 1450, 0, 500000, 0, 2.0)
# # }

workload_ranges = {
    'voter': {'throughput_range': (0, 8000), 'latency_range': (0, 130000), 'cpu_usage_range': (0, 0.2)},
    'sibench': {'throughput_range': (0, 17500), 'latency_range': (0, 300000), 'cpu_usage_range': (0, 0.4)},
    'smallbank': {'throughput_range': (0, 10000), 'latency_range': (0, 500000), 'cpu_usage_range': (0, 0.6)},
    'tatp': {'throughput_range': (0, 21000), 'latency_range': (0, 50000), 'cpu_usage_range': (0, 1.0)},
    'twitter': {'throughput_range': (0, 13000), 'latency_range': (0, 60000), 'cpu_usage_range': (0, 1.2)},
    'tpcc': {'throughput_range': (0, 1450), 'latency_range': (0, 500000), 'cpu_usage_range': (0, 2.0)}
}

# all_data = generate_data_points(500, workload_ranges)

# # all_data = []
# # for _ in range(4):
# #     all_data.append(generate_data_points(n_points, 0, 8000, 0, 130000, 0, 0.2))

# for i, workload in enumerate(workloads):
#     create_plots(all_data[i], f"mysql_{workload}")