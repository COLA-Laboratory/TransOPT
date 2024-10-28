import os
import numpy as np
import json
import pandas as pd
import re

import matplotlib.pyplot as plt



out_put_dir = '/home/cola/transopt_files/output1/results'
analysis_dir = './analysis_res/'



def find_jsonl_files(directory):
    jsonl_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    
    # 按照文件名中的数字序号进行排序
    jsonl_files.sort(key=lambda x: int(re.search(r'/(\d+)_', x).group(1)))
    
    return jsonl_files

def find_dirs(directory):
    dir_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_files.append(os.path.join(root, dir))
    return dir_files

def remove_empty_directories(directory):
    # 遍历指定目录下的所有子目录
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 检查目录是否为空
            if not os.listdir(dir_path):
                # 如果目录为空，则删除
                print(f"Removing empty directory: {dir_path}")
                os.rmdir(dir_path)

# remove_empty_directories(out_put_dir)
# print(find_dirs(out_put_dir))

def plot_bins(test_data, val_data, save_file_name):
    os.makedirs(analysis_dir + 'bins/', exist_ok=True)
    
    plt.clf()
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.hist(test_data, bins=bins, alpha=0.5, label='test acc', color='blue', edgecolor='black')
    plt.hist(val_data, bins=bins, alpha=0.5, label='val acc', color='orange', edgecolor='black')

    
    plt.savefig(analysis_dir + 'bins/' + save_file_name)


def plot_traj(test_data, val_data, save_file_name):
    os.makedirs(analysis_dir + 'traj/', exist_ok=True)
    
    plt.clf()
    # test_data = np.maximum.accumulate(test_data).flatten()
    # val_data = np.maximum.accumulate(val_data).flatten()
    plt.plot(test_data, label='test acc', color='blue')
    plt.plot(val_data, label='val acc', color='orange')
    plt.legend()
    plt.savefig(analysis_dir + 'traj/' + save_file_name)
    

def plot_scatter(x, y, values, save_file_name):
    os.makedirs(analysis_dir + 'scatter/', exist_ok=True)
    
    plt.clf()
    plt.scatter(x, y, s=100, c=values, cmap='Reds', edgecolor='black')
    plt.colorbar(label='Value')


    # 设置标题和标签
    plt.title('Scatter Plot with Color Mapping')
    plt.savefig(analysis_dir + 'scatter/' + save_file_name)


def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")


NN_name = {}
datasets = {}
test_env = [0,1]
for dir_name in find_dirs(out_put_dir):
    dir_name = dir_name.split('/')[-1]
    # if 'ERM' in dir_name:
    #     continue
    # if 'IRM' in dir_name:
    #     continue
    NN_name[dir_name.split('_')[0]] = 1
    datasets[dir_name.split('_')[1]] = 1

df = pd.DataFrame(0, index=list(datasets.keys()), columns=list(NN_name.keys()))
df2 = pd.DataFrame(0, index=list(datasets.keys()), columns=list(NN_name.keys()))

for dir_name in find_dirs(out_put_dir):
    # if 'ERM' in dir_name:
    #     continue
    # if 'IRM' in dir_name:
    #     continue
    dir_name = dir_name.split('/')[-1]
    nn_name=dir_name.split('_')[0]
    dataset_name = dir_name.split('_')[1]
    best_val_acc = 0
    best_test_acc = 0
    best_test_acc2 = 0
    all_test = []
    all_valid = []
    
    location = []
    # if 'ColoredMNIST' in dir_name:
    #     continue
    for  file_name in find_jsonl_files(out_put_dir + '/' + dir_name):
        # print(file_name)
        f_name = file_name.split('/')[-1]
        weight_decay = float(f_name.split('_')[-1][:-6])
        lr = float(f_name.split('_')[-4])
        location.append([lr, weight_decay])
        with open(file_name, 'r') as f:
            try:
                results = json.load(f)
                print(results)
                val_acc = []
                test_acc = []
                for t_env in test_env:
                    for k,v in results.items():
                        if f'env{t_env}_out_acc' == k:
                            test_acc.append(v)

                for k,v in results.items():
                    pattern = r'env\d+_val_acc'
                    if re.match(pattern, k):
                        number = int(k[3])
                        if number not in test_env:
                            val_acc.append(v)

                val_acc_mean = np.mean(val_acc)
                test_acc_mean = np.mean(test_acc)
                
                all_test.append(test_acc_mean)
                all_valid.append(val_acc_mean)
                
                if test_acc_mean > best_test_acc:
                    best_test_acc = test_acc_mean
                
                if val_acc_mean > best_val_acc:
                    best_val_acc = val_acc_mean
                    best_test_acc2 = test_acc_mean

            except:
                print(f'{file_name} can not open')
                continue
    plot_bins(all_test, all_valid, f'{dir_name}.png')
    plot_traj(all_test, all_valid, f'{dir_name}_traj.png')
    locations = np.array(location)
    plot_scatter(locations[:,0], locations[:,1], all_valid, f'{dir_name}_scatter.png')
    
    df.at[dataset_name, nn_name] = best_test_acc
    df2.at[dataset_name, nn_name] = best_test_acc2
print(df)
print('------------------')
print(df2)



    # with open(file, 'r') as f:
    #     it = file.split('/')[-1].split('_')[0]
    #     print(it)
    #     results = json.load(f)
