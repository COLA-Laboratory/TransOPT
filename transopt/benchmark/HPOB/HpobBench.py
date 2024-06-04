import copy

import numpy as np
import json

import os
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = "1"


##/mnt/data/cola/hpob-data
class HPOb():
    def __init__(self, search_space_id, data_set_id, xdim, path='./Benchmark/HPOB/hpob-data'):
        self.name = f'HPOb_{xdim}d_{data_set_id}'
        self.search_space_id = search_space_id
        self.data_set_id = data_set_id
        self.xdim = xdim
        self.query_num = 0
        self.task_type = 'Tabular'
        with open(path + "/meta-test-dataset.json", "r") as f:
            data_set = json.load(f)
            data_set = data_set[search_space_id][data_set_id]

        self.data_set = data_set
        self.RX = [[0,1] for i in range(xdim)]
        self.bounds = np.array([[-1.0] * self.xdim, [1.0] * self.xdim])

        self.unobserved_indexs = list(range(len(data_set['y'])))
        self.observed_indexs = []

        self.data_input = {index: value for index, value in enumerate(data_set['X'])}
        self.data_output ={index: value for index, value in enumerate(data_set['y'])}
        self.unobserved_input = {index: value for index, value in enumerate(data_set['X'])}
        self.unobserved_output = {index: value for index, value in enumerate(data_set['y'])}




    def transfer(self, X):
        return (X + 1) * (self.RX[:, 1] - self.RX[:, 0]) / 2 + (self.RX[:, 0])

    def normalize(self, X):
        return 2 * (X - (self.RX[:, 0])) / (self.RX[:, 1] - self.RX[:, 0]) - 1

    def data_num(self):
        return len(self.unobserved_output)

    def get_var(self, indexs):
        X = [self.unobserved_input[idx] for idx in indexs]
        return  np.array(X)

    def get_idx(self, vars):
        unob_idx = []
        vars = np.array(vars)
        for var in vars:
            for idx in self.unobserved_indexs:
                if np.all(var == self.unobserved_input[idx]):
                    unob_idx.append(idx)

        return  unob_idx

    def get_all_unobserved_var(self):
        return np.array(list(self.unobserved_input.values()))

    def get_all_unobserved_idxs(self):
        return self.unobserved_indexs

    def f(self,X, indexs):
        self.query_num += len(indexs)
        y = []
        for idx in indexs:
            y.append(self.unobserved_output[idx][0])
            del self.unobserved_output[idx]
            del self.unobserved_input[idx]
            self.unobserved_indexs.remove(idx)
            self.observed_indexs.append(idx)
        f = np.array(y)
        return f




dataset_dic = {'4796': ['3549', '3918', '9903', '23'],
               '5527': ['146064', '146065', '9914', '145804', '31', '10101'],
               '5636': ['146064', '145804', '9914', '146065', '10101', '31'],
               '5859': ['9983', '31', '37', '3902', '9977', '125923'], '5860': ['14965', '9976', '3493'],
               '5891': ['9889', '3899', '6566', '9980', '3891', '3492'], '5906': ['9971', '3918'],
               '5965': ['145836', '9914', '3903', '10101', '9889', '49', '9946'],
               '5970': ['37', '3492', '9952', '49', '34536', '14951'],
               '5971': ['10093', '3954', '43', '34536', '9970', '6566'],
               '6766': ['3903', '146064', '145953', '145804', '31', '10101'],
               '6767': ['146065', '145804', '146064', '9914', '9967', '31'],
               '6794': ['145804', '3', '146065', '10101', '9914', '31'],
               '7607': ['14965', '145976', '3896', '3913', '3903', '9946', '9967'],
               '7609': ['145854', '3903', '9967', '145853', '34537', '125923', '145878'],
               '5889': ['9971', '3918']}

def calculate_correlation(x1, y1, X2, Y2):
    # 计算x1与X2之间的欧氏距离
    distances = cdist(x1.reshape(1, -1), X2, metric='euclidean')

    # 找到X2中最近的点的索引
    closest_index = np.argmin(distances)

    # 使用pair-wise统计方法计算相关性
    correlation = np.corrcoef(y1, Y2[closest_index].flatten())[0, 1]

    return correlation



if __name__ == '__main__':
    search_space_id = '6794'
    for data_set_id in dataset_dic[search_space_id]:
        hpo = HPOb(search_space_id=search_space_id, data_set_id=data_set_id, xdim=10,path='./hpob-data')
        data_x = np.array(hpo.data_set['X'])
        data_y = hpo.data_set['y']

        # 对Y进行排序，获取排序后的索引
        sorted_indices = np.argsort(data_y, axis=0)

        # 根据排序后的索引重新排列X
        sorted_X = data_x[sorted_indices[:,0]]
        # 绘制heatmap图
        plt.figure(figsize=(8, 6))
        plt.imshow(sorted_X, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Heatmap of Sorted X')
        plt.xlabel('Features')
        plt.ylabel('Samples (Sorted by Y)')
        plt.savefig(f'heatmap of sorted X for dataset:{data_set_id}')
