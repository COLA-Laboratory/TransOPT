import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def calculate_importances(X, y):
    """
    Calculates and returns parameter importances.
    """

    model = DecisionTreeRegressor()
    model.fit(X, y[:, np.newaxis])
    feature_importances = model.feature_importances_

    return feature_importances


def calculate_interaction(X, y):
    num_parameters = X.shape[1]
    h_matrix = np.zeros((num_parameters, num_parameters))

    # 训练单个变量的模型
    single_models = []
    for i in range(num_parameters):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:, [i]], y)
        single_models.append(model)

    # 两两特征组合，计算 H^2
    for (i, j) in combinations(range(num_parameters), 2):
        model_jk = RandomForestRegressor(n_estimators=50, random_state=42)
        model_jk.fit(X[:, [i, j]], y)
        f_jk = model_jk.predict(X[:, [i, j]])

        f_j = single_models[i].predict(X[:, [i]])
        f_k = single_models[j].predict(X[:, [j]])

        numerator = np.sqrt(np.sum((f_jk - f_j - f_k) ** 2))

        h_matrix[i, j] = numerator
        h_matrix[j, i] = h_matrix[i, j]

    mean = np.mean(h_matrix)
    std = np.std(h_matrix)

    normalized_matrix = (h_matrix - mean) / std
    scaled_matrix = 1 / (1 + np.exp(-normalized_matrix))
    
    return scaled_matrix


def plot_network(X, y, nodes):
    G = nx.Graph()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
    
    nodes_weight = calculate_importances(X, y)
    for node, weight in zip(nodes, nodes_weight):
        G.add_node(node, weight=weight)

    edges_weight = calculate_interaction(X, y)
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            weight = np.random.uniform(0, 1)  # 生成一个0到1之间的随机权重
            G.add_edge(nodes[i], nodes[j], weight=weight)
        
        
        
    # for i in range(5):
    #     for j in range(i + 1, 5):
    #         weight = edges_weight[i, j]
    #         G.add_edge(nodes[i], nodes[j], weight=weight)
    
    # 设置节点的位置为圆形布局
    pos = nx.circular_layout(G)

    # 创建颜色映射
    node_cmap = plt.cm.Greens
    edge_cmap = plt.cm.Blues

    # 节点的颜色根据权重映射
    node_color = [node_cmap(data['weight']) for v, data in G.nodes(data=True)]
    node_size = [data['weight'] * 2000 + 1000 for v, data in G.nodes(data=True)]
    node_alpha = [data['weight'] for v, data in G.nodes(data=True)]  # 透明度根据权重调整

    # 绘制网络图
    edges = G.edges(data=True)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=node_alpha)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=16)

    # 单独绘制每条边，设置颜色和透明度
    for u, v, data in edges:
        color = edge_cmap(data['weight'])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, alpha=data['weight'], edge_color=[color])

    fig.set_facecolor("None")
    ax.set_facecolor("#191C36")
    # ax.axis('off')

    path = os.getcwd()
    save_path = os.path.join(path, "webui/src/pictures/parameter_network.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    plt.close()
    # 显示图形
    # plt.show()



if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 5))  # 5个特征
    y = 5 * X[:, 0] * X[:, 1] + 3 * X[:, 2] + X[:, 3] + np.random.normal(0, 0.5, 100)
    plot_network(X, y, nodes=['X1', 'X2', 'X3', 'X4', 'X5'])