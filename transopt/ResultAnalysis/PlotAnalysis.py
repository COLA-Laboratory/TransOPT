import numpy as np
from collections import Counter, defaultdict
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pds
import os
import seaborn as sns
from transopt.utils.sk import Rx
from pathlib import Path
import scipy
import tikzplotlib
from sklearn.cluster import DBSCAN
from transopt.ResultAnalysis.CompileTex import compile_tex
import matplotlib.gridspec as gridspec
plot_registry = {}
import re

# 注册函数的装饰器
def plot_register(name):
    def decorator(func_or_class):
        if name in plot_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        plot_registry[name] = func_or_class
        return func_or_class
    return decorator



@plot_register('sk')
def plot_sk(ab:AnalysisBase, save_path:Path):
    cr_results = {}
    results = ab.get_results_by_order(["task", "method", "seed"])

    for task_name, tasks_r in results.items():
        result = {}
        for method, method_r in tasks_r.items():
            cr_list = []
            for seed, result_obj in method_r.items():
                cr = result_obj.best_Y
                cr_list.append(cr)
            result[method] = cr_list

        a = Rx.data(**result)
        RES = Rx.sk(a)
        for r in RES:
            if r.rx in cr_results:
                cr_results[r.rx].append(r.rank)
            else:
                cr_results[r.rx] = [r.rank]

    df = pds.DataFrame(cr_results)

    sns.set_theme(style="whitegrid", font='FreeSerif')
    plt.figure(figsize=(12, 7.3))
    # plt.ylim(bottom=0.9, top=len(method_names)+0.1)
    ax = plt.gca()  # 获取坐标轴对象
    y_major_locator = MultipleLocator(1)  # 设置坐标的主要刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 应用在纵坐标上
    sns.violinplot(data=df, inner="quart")
    plt.title('Skott knott', fontsize=30, y=1.01)
    plt.xlabel('Algorithm Name', fontsize=25, labelpad=-7)
    plt.ylabel('Rank', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=10)

    save_path = Path(save_path / 'Overview')
    pdf_path = Path(save_path / 'Pictures')
    tex_path = Path(save_path / 'tex')
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    tex_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tex_path / "scott_knott.tex")

    with open(tex_path / "scott_knott.tex", 'r', encoding='utf-8') as f:
        content = f.read()

    # 添加preamble和end document
    preamble = r"\documentclass{article}" + "\n" + \
               r"\usepackage{pgfplots}" + "\n" + \
               r"\usepackage{tikz}" + "\n" + \
               r"\begin{document}" + "\n" + \
               r"\pagestyle{empty}" + "\n"
    end_document = r"\end{document}" + "\n"
    # 替换 false 为 true
    content = re.sub(r'majorticks=false', 'majorticks=true', content)
    pattern = r'axis line style={lightgray204},\n'
    content = re.sub(pattern, '', content)
    # 插入字号控制
    insert_text = r"font=\large," + "\n" + \
                  r"tick label style={font=\small}," + "\n" + \
                  r"label style={font=\normalsize}," + "\n"
    insert_position = content.find(r'tick align=outside,')
    modified_content = content[:insert_position] + insert_text + content[insert_position:]

    # 将修改后的内容写回文件
    with open(tex_path / "scott_knott.tex", 'w', encoding='utf-8') as f:
        f.write(preamble + modified_content + end_document)

    compile_tex(tex_path / "scott_knott.tex", pdf_path)
    plt.close()



@plot_register('cr')
def convergence_rate(ab:AnalysisBase, save_path:Path, **kwargs):
    cr_list = []
    cr_all = {}
    cr_results = {}
    def acc_iter(Y, anchor_value):
        for i in range(1, len(Y)):
            best_fn = np.min(Y[:i])
            if best_fn <= anchor_value:
                return i/len(Y)
        return 1

    results = ab.get_results_by_order(["method", "seed", "task"])
    best_Y_values = defaultdict(list)

    # 遍历 data 字典，收集 best_Y 值
    for method, tasks in results.items():
        for seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                best_Y = result_obj.best_Y
                if best_Y is not None:
                    best_Y_values[task_name].append(best_Y)

    # 计算并返回每个 task_name 下 best_Y 值的 3/4 分位数
    quantiles = {task_name: np.percentile(values, 75) for task_name, values in best_Y_values.items()}

    for method, tasks in results.items():
        for seed, task_seed in tasks.items():
            for task_name, result_obj in task_seed.items():
                Y = result_obj.Y
                if Y is None:
                    raise ValueError(f"Y is not set for method {method}, task {task_name}")

                cr = acc_iter(Y, anchor_value=quantiles[task_name])
                cr_list.append(cr)

        cr_all[method] = cr_list

    a = Rx.data(**cr_all)
    RES = Rx.sk(a)
    for r in RES:
        if r.rx in cr_results:
            cr_results[r.rx].append(r.rank)
        else:
            cr_results[r.rx] = [r.rank]

    cr_results = pds.DataFrame(cr_results)

    sns.set_theme(style="whitegrid", font='FreeSerif')
    plt.figure(figsize=(12, 7.3))
    # plt.ylim(bottom=0.9, top=len(method_names)+0.1)
    ax = plt.gca()  # 获取坐标轴对象
    y_major_locator = MultipleLocator(1)  # 设置坐标的主要刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 应用在纵坐标上
    sns.violinplot(data=cr_results, inner="quart")
    plt.title('Convergence Rate', fontsize=30, y=1.01)
    plt.xlabel('Algorithm Name', fontsize=25, labelpad=-7)
    plt.ylabel('Rate', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=10)

    save_path = Path(save_path / 'Overview')
    pdf_path = Path(save_path / 'Pictures')
    tex_path = Path(save_path / 'tex')
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    tex_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tex_path / "convergence_rate.tex")

    with open(tex_path / "convergence_rate.tex", 'r', encoding='utf-8') as f:
        content = f.read()

    # 添加preamble和end document
    preamble = r"\documentclass{article}" + "\n" + \
               r"\usepackage{pgfplots}" + "\n" + \
               r"\usepackage{tikz}" + "\n" + \
               r"\begin{document}" + "\n" + \
               r"\pagestyle{empty}" + "\n"
    end_document = r"\end{document}" + "\n"
    # 替换 false 为 true
    content = re.sub(r'majorticks=false', 'majorticks=true', content)
    pattern = r'axis line style={lightgray204},\n'
    content = re.sub(pattern, '', content)
    # 插入字号控制
    insert_text = r"font=\large," + "\n" + \
                  r"tick label style={font=\small}," + "\n" + \
                  r"label style={font=\normalsize}," + "\n"
    insert_position = content.find(r'tick align=outside,')
    modified_content = content[:insert_position] + insert_text + content[insert_position:]

    # 将修改后的内容写回文件
    with open(tex_path / "convergence_rate.tex", 'w', encoding='utf-8') as f:
        f.write(preamble + modified_content + end_document)

    compile_tex(tex_path / "convergence_rate.tex", pdf_path)
    plt.close()


def save_traj_data(ab, save_path):
    # 先找出所有的任务名称
    results = ab.get_results_by_order(["task", "method", "seed"])

    for task_name, tasks_r in results.items():
        # 为每个任务创建一个字典来存储数据
        task_data = {}

        for method, method_r in tasks_r.items():
            res = []

            for seed, result_obj in method_r.items():
                Y = result_obj.Y
                if Y is not None:
                    res.append(np.minimum.accumulate(Y).flatten())

            if res:
                # 计算中位数和标准差
                res_array = np.array(res)
                median = np.median(res_array, axis=0)
                std = np.std(res_array, axis=0)

                # 将数据存储到字典中
                task_data[f'{method}_mean'] = median
                task_data[f'{method}_low'] = median - std
                task_data[f'{method}_high'] = median + std

        if task_data:
            # 创建保存路径
            os.makedirs(save_path / 'traj'/ 'tex', exist_ok=True)

            # 设置文件路径
            file_path = save_path / 'traj'/ 'tex'/ f"{task_name}.dat"

            # 获取序号的起点
            start_idx = ab._init

            # 选择从 start_idx 开始的数据
            end_idx = start_idx + len(median)
            for key in task_data.keys():
                task_data[key] = task_data[key][start_idx:end_idx]

            # 将数据保存到文件
            with open(file_path, 'w') as f:
                # 写入列名
                col_names = ' '.join(['id'] + list(task_data.keys()))
                f.write(col_names + '\n')

                # 写入数据
                for i in range(len(task_data[list(task_data.keys())[0]])):
                    row_data = ' '.join([str(start_idx + i)] + [f'{x[i]:0.8f}' for x in task_data.values()])
                    f.write(row_data + '\n')

            print(f"Data saved for {task_name}")

@plot_register('traj')
def traj2latex(ab: AnalysisBase, save_path: Path):
    # 从 ab 对象中获取任务名称和方法名称
    save_traj_data(ab, save_path)
    results = ab.get_results_by_order(["task", "method", "seed"])
    methods = ab.get_methods()

    # 从 ab 对象中获取 start_idx, y_max 和 y_min
    start_idx = ab._init
    end_idx = ab._end

    # 创建保存路径
    os.makedirs(save_path / 'traj' / 'tex', exist_ok=True)


    # 设置文件路径
    for task_name, tasks_r in results.items():
        all_data = []

        for method, method_r in tasks_r.items():
            for seed, result_obj in method_r.items():
                Y = result_obj.Y
                if Y is not None:
                    min_values = np.minimum.accumulate(Y)
                    all_data.append(min_values.flatten())

        if all_data:
            all_data = np.concatenate(all_data)
            y_min = np.min(all_data) - np.std(all_data)
            y_max = np.max(all_data) + np.std(all_data)

        tex_save_path = save_path / 'traj' / 'tex' / f"{task_name}.tex"
        data_file = f"{task_name}.dat"
        # 开始写入 LaTeX 代码
        latex_code = f"""
        \\documentclass{{article}}
        \\usepackage{{pgfplots}}
        \\usepackage{{tikz}}
        \\usetikzlibrary{{intersections}}
        \\usepackage{{helvet}}
        \\usepackage[eulergreek]{{sansmath}}
        \\usepgfplotslibrary{{fillbetween}}

        \\begin{{document}}
        \\pagestyle{{empty}}


        \\pgfplotsset{{compat=1.12,every axis/.append style={{
            font = \\large,
            grid = major,
            xlabel = {{\\# of FEs}},
            ylabel = {{$f(\\mathbf{{x}}^\\ast)$}},
            thick,
            xmin={start_idx},
            xmax={end_idx},  % Adjust as needed
            ymin={y_min},
            ymax={y_max},
            line width = 1pt,
            tick style = {{line width = 0.8pt}}
        }}}}
        \pgfplotsset{{every plot/.append style={{very thin}}}}
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                title={{${task_name}$}},
                width=\\textwidth,
                height=0.5\\textwidth,
            ]"""


        for method in methods:
            # 这里需要根据你的数据文件的具体结构来调整
            latex_code += f"""
            \\addplot[color={{{ab.get_color_for_method(method)}}}, solid, line width=1pt]table [x = id, y = {method}_mean]{{{data_file}}};
            \\addlegendentry{{{method}}};
            """

        for method in methods:
            # 这里需要根据你的数据文件的具体结构来调整

            latex_code += f"""
            \\addplot[color={{{ab.get_color_for_method(method)}}}, name path={method}_L, draw=none] table[x = id, y = {method}_low] {{{data_file}}};
            \\addplot[color={{{ab.get_color_for_method(method)}}}, name path={method}_U, draw=none] table[x = id, y = {method}_high] {{{data_file}}};
            \\addplot[color={{{ab.get_color_for_method(method)}}},opacity=0.3] fill between[of={method}_U and {method}_L];
            """

        latex_code += f"""
                    \\end{{axis}}
            \\end{{tikzpicture}}
        \\end{{document}}"""

        # 将 LaTeX 代码保存到文件
        with open(tex_save_path, 'w') as f:
            f.write(latex_code)
        try:
            compile_tex(tex_save_path, save_path / 'traj')
        except:
            pass

        print(f"LaTeX code has been saved to {tex_save_path}")



@plot_register('violin')
def plot_violin(ab:AnalysisBase, save_path, **kwargs):
    data = {'Method': [], 'Performance rank': []}
    method_names = set()

    results = ab.get_results_by_order(["task", "seed", "method"])

    for task_name, task_r in results.items():
        for seed, seed_r in task_r.items():
            res = {}
            for method, result_obj in seed_r.items():
                method_names.add(method)
                Y = result_obj.Y
                if Y is not None:
                    min_values = np.min(Y)
                    res[method] = min_values
            sorted_value = sorted(res.values())
            for v_id, v in enumerate(sorted_value):
                for k, vv in res.items():
                    if v == vv:
                        data['Method'].append(k)
                        data['Performance rank'].append(v_id+1)

    sns.set_theme(style="whitegrid", font='FreeSerif')
    plt.figure(figsize=(12, 7.3))
    plt.ylim(bottom=0.9, top=len(method_names)+0.1)
    ax = plt.gca()  # 获取坐标轴对象
    y_major_locator = MultipleLocator(1)  # 设置坐标的主要刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 应用在纵坐标上
    sns.violinplot(x='Method', y='Performance rank', data=data,
                   order=list(method_names),
                   inner="box", color="silver", cut=0, linewidth=3)
    plt.title('Violin plot', fontsize=30, y=1.01)
    plt.xlabel('Algorithm Name', fontsize=25, labelpad=-7)
    plt.ylabel('Performance rank', fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=10)

    save_path = Path(save_path / 'Overview')
    pdf_path = Path(save_path / 'Pictures')
    tex_path = Path(save_path / 'tex')
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    tex_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tex_path / "violin.tex")
    with open(tex_path / "violin.tex", 'r', encoding='utf-8') as f:
        content = f.read()

    # 添加preamble和end document
    preamble = r"\documentclass{article}" + "\n" + \
               r"\usepackage{pgfplots}" + "\n" + \
               r"\usepackage{tikz}" + "\n" + \
               r"\begin{document}" + "\n" + \
               r"\pagestyle{empty}" + "\n"
    end_document = r"\end{document}" + "\n"
    # 替换 false 为 true
    content = re.sub(r'majorticks=false', 'majorticks=true', content)
    pattern = r'axis line style={lightgray204},\n'
    content = re.sub(pattern, '', content)
    # 插入字号控制
    insert_text = r"font=\large," + "\n" + \
                  r"tick label style={font=\small}," + "\n" + \
                  r"label style={font=\normalsize}," + "\n"
    insert_position = content.find(r'tick align=outside,')
    modified_content = content[:insert_position] + insert_text + content[insert_position:]

    # 将修改后的内容写回文件
    with open(tex_path / "violin.tex", 'w', encoding='utf-8') as f:
        f.write(preamble + modified_content + end_document)

    compile_tex(tex_path / "violin.tex", pdf_path)
    plt.close()


@plot_register('box')
def plot_box(ab:AnalysisBase, save_path, **kwargs):
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'median'
    methods = set()

    results = ab.get_results_by_order(["method", "task", "seed"])

    result_list = []
    for method, method_r in results.items():
        methods.add(method)
        result = []
        for task, task_r in method_r.items():
            best = []
            for seed, result_obj in task_r.items():
                if result_obj is not None:
                    Y = result_obj.Y
                    if Y is not None:
                        min_values = np.min(Y)
                        best.append(min_values)
            if mode == 'median':
                result.append(np.median(best))
            elif mode == 'mean':
                result.append(np.mean(best))
        result_list.append(result)
    result_list = np.array(result_list).T

    ranks = np.array([scipy.stats.rankdata(x, method='min') for x in result_list])
    df = pds.DataFrame(ranks, columns=methods)

    sns.set_theme(style='whitegrid', font='FreeSerif')
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    sns.boxplot(df, color='#c2d0e9')
    plt.title('Box plot', fontsize=30, y=1.03)
    plt.xlabel('Algorithm Name', fontsize=25)
    plt.ylabel('Rank', fontsize=25)
    plt.xticks(fontsize=20, rotation=10)
    plt.yticks(fontsize=20)

    save_path = Path(save_path / 'Overview')
    pdf_path = Path(save_path / 'Pictures')
    tex_path = Path(save_path / 'tex')
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    tex_path.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tex_path / "box.tex")

    with open(tex_path / "box.tex", 'r', encoding='utf-8') as f:
        content = f.read()

        # 添加preamble和end document
    preamble = r"\documentclass{article}" + "\n" + \
               r"\usepackage{pgfplots}" + "\n" + \
               r"\usepackage{tikz}" + "\n" + \
               r"\begin{document}" + "\n" + \
               r"\pagestyle{empty}" + "\n"
    end_document = r"\end{document}" + "\n"

    content = re.sub(r'majorticks=false', 'majorticks=true', content)
    pattern = r'axis line style={lightgray204},\n'
    content = re.sub(pattern, '', content)
    insert_text = r"font=\large," + "\n" + \
                  r"tick label style={font=\small}," + "\n" + \
                  r"label style={font=\normalsize}," + "\n"
    insert_position = content.find(r'tick align=outside,')
    modified_content = content[:insert_position] + insert_text + content[insert_position:]

    # 将修改后的内容写回文件
    with open(tex_path / "box.tex", 'w', encoding='utf-8') as f:
        f.write(preamble + modified_content + end_document)

    compile_tex(tex_path / "box.tex", pdf_path)

    plt.close()


@plot_register('dbscan')
def dbscan_analysis(ab: AnalysisBase, save_path, **kwargs):
    results = ab.get_results_by_order(['task', 'method', 'seed'])
    tasks_names = set()
    method_names = set()
    result_of_n_clusters = {}
    result_of_noise_points = {}
    result_of_avg_cluster_size = {}

    for task_name, task_r in results.items():
        tasks_names.add(task_name)
        result_of_n_clusters[task_name] = defaultdict(dict)
        result_of_noise_points[task_name] = defaultdict(dict)
        result_of_avg_cluster_size[task_name] = defaultdict(dict)
        for method, method_r in task_r.items():
            method_names.add(method)
            result_of_n_clusters[task_name][method] = []
            result_of_noise_points[task_name][method] = []
            result_of_avg_cluster_size[task_name][method] = []
            for seed, result_obj in method_r.items():
                if result_obj is not None:
                    Y = result_obj.Y
                    X = result_obj.X
                    if Y is not None:
                        db = DBSCAN(eps=0.5, min_samples=5)
                        # 执行聚类
                        db.fit(X)
                        # 获取聚类标签
                        labels = db.labels_
                        # 计算簇的数量（忽略噪声点，其标签为 -1）
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        cluster_sizes = Counter(labels)
                        noise_points = cluster_sizes[-1]  # 标签为 -1 的点是噪声点
                        # 计算平均簇大小（不包括噪声点）
                        if n_clusters > 0:
                            t_size = 0
                            for ids, cs in cluster_sizes.items():
                                if ids >= 0:
                                    t_size += cs
                            avg_cluster_size = t_size / n_clusters
                        else:
                            avg_cluster_size = 0

                        result_of_n_clusters[task_name][method].append(n_clusters)
                        result_of_noise_points[task_name][method].append(noise_points)
                        result_of_avg_cluster_size[task_name][method].append(avg_cluster_size)

    def modify_lex_file(tex_path, pdf_path):
        with open(tex_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # 添加preamble和end document
        preamble = r"\documentclass{article}" + "\n" + \
                   r"\usepackage{pgfplots}" + "\n" + \
                   r"\usepackage{tikz}" + "\n" + \
                   r"\begin{document}" + "\n" + \
                   r"\pagestyle{empty}" + "\n"
        end_document = r"\end{document}" + "\n"

        content = re.sub(r'majorticks=false', 'majorticks=true', content)
        pattern = r'axis line style={lightgray204},\n'
        content = re.sub(pattern, '', content)
        insert_text = r"font=\large," + "\n" + \
                      r"tick label style={font=\small}," + "\n" + \
                      r"label style={font=\normalsize}," + "\n"
        insert_position = content.find(r'tick align=outside,')
        modified_content = content[:insert_position] + insert_text + content[insert_position:]

        # 将修改后的内容写回文件
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(preamble + modified_content + end_document)

        compile_tex(tex_path, pdf_path)

    tex_path = save_path / 'dbscan' / 'tex'
    pdf_path = save_path / 'dbscan'
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path.mkdir(parents=True, exist_ok=True)
    tex_path.mkdir(parents=True, exist_ok=True)
    for task_name in tasks_names:
        df_n_clusters = pds.DataFrame(result_of_n_clusters[task_name])
        sns.set_theme(style='whitegrid', font='FreeSerif')
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        sns.boxplot(data=df_n_clusters, order=method_names)
        plt.title('Clusters', fontsize=30, y=1.03)
        plt.ylabel('number', fontsize=25)
        plt.xticks(fontsize=20, rotation=10)
        plt.yticks(fontsize=20)
        tikzplotlib.save(tex_path / f"{task_name}_n_clusters.tex")
        modify_lex_file(tex_path / f"{task_name}_n_clusters.tex", pdf_path)
        plt.close()

        df_noise_points = pds.DataFrame(result_of_noise_points[task_name])
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        sns.boxplot(data=df_noise_points, order=method_names)
        plt.title('Noise Points', fontsize=30, y=1.03)
        plt.ylabel('number', fontsize=25)
        plt.xticks(fontsize=20, rotation=10)
        plt.yticks(fontsize=20)
        tikzplotlib.save(tex_path / f"{task_name}_noise_points.tex")
        modify_lex_file(tex_path / f"{task_name}_noise_points.tex", pdf_path)
        plt.close()

        df_avg_cluster_size = pds.DataFrame(result_of_avg_cluster_size[task_name])
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        sns.boxplot(data=df_avg_cluster_size, order=method_names)
        plt.title('Avg Cluster Size', fontsize=30, y=1.03)
        plt.ylabel('number', fontsize=25)
        plt.xticks(fontsize=20, rotation=10)
        plt.yticks(fontsize=20)
        tikzplotlib.save(tex_path / f"{task_name}_avg_cluster_size.tex")
        modify_lex_file(tex_path / f"{task_name}_avg_cluster_size.tex", pdf_path)
        plt.close()


@plot_register('heatmap')
def plot_heatmap(ab:AnalysisBase, save_path, **kwargs):
    results = ab.get_results_by_order(['method', 'task', 'seed'])
    methods = ab.get_methods()
    tasks = list(ab.get_task_names())

    # Step 1: Calculate the best result for each method, task, and seed
    best_results = {method: {task: [] for task in tasks} for method in methods}
    for method in methods:
        for task in tasks:
            for seed, result_obj in results[method][task].items():
                if result_obj is not None and result_obj.Y is not None:
                    best_results[method][task].append(result_obj.best_Y)

    # Step 2: Calculate the mean of the best results for each method and task
    mean_best_results = {method: {task: np.mean(best_results[method][task]) for task in tasks} for method in methods}

    # Step 3: Create a dataframe for the heatmap
    heatmap_data = pds.DataFrame(mean_best_results).T

    # Step 4: Plot the heatmap
    n_cols = len(tasks)
    colormaps = ['Blues', 'Reds', 'Greens', 'Purples']  # Adjust as needed
    fig = plt.figure(figsize=(n_cols * 2, 5))
    gs = gridspec.GridSpec(1, n_cols, width_ratios=[1 for _ in range(n_cols)])
    for col, task in enumerate(tasks):
        ax = plt.subplot(gs[col])
        sns.heatmap(heatmap_data[[task]], annot=True, cmap=colormaps[col % len(colormaps)], cbar=False, ax=ax, **kwargs)
        if col == 0:
            ax.set_ylabel('Method')
        if col != 0:
            ax.set_yticks([])
            ax.set_yticklabels([])

    plt.suptitle("Heatmap of Methods")
    fig.text(0.5, 0.01, 'Task Name', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    save_path = Path(save_path / 'Overview')
    png_path = Path(save_path / 'Pictures')
    save_path.mkdir(parents=True, exist_ok=True)
    png_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(png_path/'heatmap.png', format='png')

