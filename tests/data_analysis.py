import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def load_data(data_folder):
    data = {}
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = json.load(f)
                x = []
                for key in ['lr', 'weight_decay', 'momentum', 'dropout_rate']:
                    pattern = rf'({key}_)([\d.e-]+)'
                    match = re.search(pattern, filename)
                    if match:
                        value = float(match.group(2))
                        if key in ['lr', 'weight_decay']:
                            x.append(np.log10(value))
                        else:
                            x.append(value)
                data[filename] = {
                    'x': x,
                    'test_standard_acc': content['test_standard_acc'],
                    'test_robust_acc': np.mean([v for k, v in content.items() if k.startswith('test_') and k != 'test_standard_acc'])
                }
    return data

def get_non_dominated_solutions(data):
    F = np.array([[1 - d['test_standard_acc'], 1 - d['test_robust_acc']] for d in data.values()])
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    non_dominated = fronts[0]
    return F[non_dominated]

def plot_non_dominated_solutions(ax, solutions, label, color):
    ax.scatter(solutions[:, 0], solutions[:, 1], label=label, color=color)
    # ax.plot(solutions[:, 0], solutions[:, 1], color=color)
# 主函数
    

def compare_nsga2_results_all(res):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define color mapping
    colors = plt.cm.rainbow(np.linspace(0, 1, len(res)))

    for (k, v), color in zip(res.items(), colors):
        data = load_data(v)
        
        # Get all data points
        all_points = np.array([[1 - d['test_standard_acc'], 1 - d['test_robust_acc']] for d in data.values()])
        
        # Get non-dominated solutions
        non_dominated_data = get_non_dominated_solutions(data)

        # Plot all points with transparency
        ax.scatter(all_points[:, 0], all_points[:, 1], label=f'{k} (all)', color=color, alpha=0.3)
        
        # Plot non-dominated solutions without transparency
        ax.scatter(non_dominated_data[:, 0], non_dominated_data[:, 1], label=f'{k} (non-dominated)', color=color, edgecolors='black')

    # Set chart properties
    ax.set_xlabel('Test Standard Accuracy')
    ax.set_ylabel('Test Robust Accuracy')
    ax.set_title('Comparison of Solutions for Different Sizes')
    ax.legend()
    ax.grid(True)

    # Invert x and y axes to show accuracy instead of error
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Save the chart
    plt.savefig('compare_all.png')
    plt.close(fig)
    

def compare_nsga2_results(res):
    # 加载两个文件夹的数据
    all_res = {}
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(res)))

    for (k, v), color in zip(res.items(), colors):
        data = load_data(v)
        
        # 获取非支配解
        non_dominated_data = get_non_dominated_solutions(data)

        # 对非支配解进行1-操作
        non_dominated_data = 1 - non_dominated_data

        # 绘制非支配解，使用不同的颜色
        plot_non_dominated_solutions(ax, non_dominated_data, f'{k}', color)

    # 设置图表属性
    ax.set_xlabel('Test Standard Accuracy')
    ax.set_ylabel('Test Robust Accuracy')
    ax.set_title('Comparison of Non-Dominated Solutions for Different Sizes')
    ax.legend()
    ax.grid(True)

    # 显示图表
    plt.savefig('compare.png')
    
def calculate_variable_importance(res):
    # 聚合所有场景的数据
    aggregated_data = {}
    for k, v in res.items():
        data = load_data(v)
        for key, value in data.items():
            if key not in aggregated_data:
                aggregated_data[key] = value

    # 定义变量名称
    variable_names = ['lr', 'weightdecay', 'momentum', 'dropout_rate']
    importance = {'test_standard_acc': {}, 'test_robust_acc': {}}

    for i, var_name in enumerate(variable_names):
        X = np.array([entry['x'][i] for entry in aggregated_data.values()])
        y_standard = np.array([entry['test_standard_acc'] for entry in aggregated_data.values()])
        y_robust = np.array([entry['test_robust_acc'] for entry in aggregated_data.values()])

        # 使用线性回归模型来评估变量重要性
        model_standard = ols(f'y ~ x', data={'x': X, 'y': y_standard}).fit()
        model_robust = ols(f'y ~ x', data={'x': X, 'y': y_robust}).fit()

        # 计算F-value和p-value作为重要性指标
        importance['test_standard_acc'][var_name] = {
            'f_value': model_standard.fvalue,
            'p_value': model_standard.f_pvalue
        }
        importance['test_robust_acc'][var_name] = {
            'f_value': model_robust.fvalue,
            'p_value': model_robust.f_pvalue
        }

    return importance

def plot_variable_importance(importance):
    vars = list(importance['test_standard_acc'].keys())
    f_values_standard = [stats['f_value'] for stats in importance['test_standard_acc'].values()]
    f_values_robust = [stats['f_value'] for stats in importance['test_robust_acc'].values()]

    x = np.arange(len(vars))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, f_values_standard, width, label='Test Standard Accuracy')
    rects2 = ax.bar(x + width/2, f_values_robust, width, label='Test Robust Accuracy')

    ax.set_ylabel('F-value')
    ax.set_title('Variable Importance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(vars, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig('variable_importance_comparison.png')
    plt.close()

def visualize_data_with_metrics(data, metric_name, output_file):
    """
    Visualize the input data with corresponding metrics.
    
    :param data: A dictionary where keys are sample names and values are dictionaries
                 containing 'x' (input variables) and metric values.
    :param metric_name: The name of the metric to visualize.
    :param output_file: The name of the file to save the plot.
    """
    X = np.array([sample['x'] for sample in data.values()])
    y = np.array([sample[metric_name] for sample in data.values()])

    # Check the dimensionality of X
    n_dims = X.shape[1]

    if n_dims <= 2:
        # If X has 2 or fewer dimensions, plot directly
        fig = plt.figure(figsize=(10, 8))
        if n_dims == 1:
            plt.scatter(X, y, c=y, cmap='viridis')
            plt.xlabel('Input Variable')
        else:  # n_dims == 2
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
            plt.xlabel('First Input Variable')
            plt.ylabel('Second Input Variable')
    else:
        # If X has more than 2 dimensions, use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

    plt.colorbar(label=metric_name)
    plt.title(f'{metric_name} vs Input Variables')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

    print(f"Plot saved as {output_file}")

# Example usage in the main function:
if __name__ == "__main__":
    # results_18 = './results_aug/non_augment'
    # results_34 = './results_size/results_34'
    results_50 = './results_size/results_50'
    # res = {'res_18':results_18, 'res_34':results_34, 'res_50':results_50}
    # compare_nsga2_results_all(res)
    
    
    # results_non_aug = './results_aug/non_augment'
    # results_aug =  './results_aug/augment'
    res = { 'without augment':results_50}
    # compare_nsga2_results_all(res)
    

    # 计算变量重要性
    importance = calculate_variable_importance(res)
    
    # 打印重要性结果
    print("Variable Importance:")
    for metric, vars in importance.items():
        print(f"\n{metric}:")
        for var, stats in vars.items():
            print(f"  {var}: F-value = {stats['f_value']:.4f}, p-value = {stats['p_value']:.4f}")

    # 可视化重要性
    plot_variable_importance(importance)

    # Load data
    results_non_aug = './results_size/results_34'
    data = load_data(results_non_aug)

    # Visualize data for standard accuracy
    visualize_data_with_metrics(data, 'test_standard_acc', 'standard_acc_visualization.png')

    # Visualize data for robust accuracy
    visualize_data_with_metrics(data, 'test_robust_acc', 'robust_acc_visualization.png')
