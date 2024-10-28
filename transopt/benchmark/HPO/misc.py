import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
from collections import Counter
from itertools import cycle
import matplotlib.pyplot as plt




class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def accuracy(network, loader, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += torch.ones(len(x)).sum().item()
    network.train()

    return correct / total



def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)
    
    
    
class LossPlotter:
    def __init__(self):
        self.classification_losses = []  # 用于存储分类损失
        self.reconstruction_losses = []  # 用于存储重构损失
        self.epochs = []  # 用于存储训练的 epoch 数
        self.cur = 0

        # 初始化绘图
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

    def update(self, classification_loss, reconstruction_loss):
        # 更新损失和 epoch 数据
        self.cur += 1
        self.classification_losses.append(classification_loss)
        self.reconstruction_losses.append(reconstruction_loss)
        self.epochs.append(self.cur)

        # 清空当前的图像
        self.ax.clear()

        # 绘制分类损失曲线
        self.ax.plot(self.epochs, self.classification_losses, label='Classification Loss', color='blue', marker='o')
        
        # 绘制重构损失曲线
        self.ax.plot(self.epochs, self.reconstruction_losses, label='Reconstruction Loss', color='orange', marker='x')

        # 设置图表标题和标签
        self.ax.set_title('Loss Curves')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

        # 显示图例
        self.ax.legend()

        # 更新图表
        plt.draw()
        plt.pause(0.01)  # 暂停以便更新图像

    def show(self):
        # 展示最终图像并关闭交互模式
        plt.ioff()
        plt.savefig('loss_curves.png')