import os
import json
import numpy as np
from transopt.utils.sk import Rx
from matplotlib import pyplot as plt


plot_dim = 7
file_path = f"/home/gsfall/data_files/synthetic/{plot_dim}d"

# file_path = f"/home/gsfall/data_files/SVM"

plot_tasks = ["Discus", "GriewankRosenbrock", "Rastrigin", "Rosenbrock", "Schwefel"]
# plot_tasks = ["SVM"]
rank = {}
for plot_task in plot_tasks:
    data_dict = {}
    for file_name in os.listdir(file_path):
        if file_name.endswith(".json"):
            parts = file_name.split("_")
            task = parts[0]
            method = "_".join(parts[1:]).split(".")[0]
            if task == plot_task:
                with open(os.path.join(file_path, file_name), "r") as f:
                    data = json.load(f)
                    data_dict[method] = data["m"]
    
    Rx_data = Rx.data(**data_dict)
    result = Rx.sk(Rx_data)
    for r in result:
        if r.rx in rank:
            rank[r.rx].append(r.rank)
        else:
            rank[r.rx] = [r.rank]

file_name = "rank.json"
with open(os.path.join(file_path, file_name), "w") as json_file:
    json.dump(rank, json_file)
pass