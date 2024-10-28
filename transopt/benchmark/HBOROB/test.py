import torch
import torch.nn as nn
import torch.nn.functional as F
from robustbench.data import load_cifar10, load_cifar10c
from robustbench.utils import clean_accuracy, load_model

import transopt.benchmark.HPO.networks
from transopt.benchmark.HPO.algorithms import ERM
from transopt.benchmark.HPO.wide_resnet import WideResNet








# 加载 CIFAR-10 数据集
x_test, y_test = load_cifar10(load_cifar10='~/transopt_files/data/')

# 转换为 Tensor
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

hparams = {
    'lr': 0.001,
    'weight_decay': 5e-4,
    'nonlinear_classifier': True
}

input_shape = (3, 32, 32)
num_classes = 10
num_domains = 1

model = ERM(input_shape, num_classes, num_domains, hparams)

from torch.utils.data import DataLoader, TensorDataset

# 使用训练数据
train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=True)

# 训练模型
for epoch in range(10):  # 训练10个epoch
    for batch in train_loader:
        minibatches = [(batch[0], batch[1])]
        model.update(minibatches)

    print(f"Epoch {epoch + 1} completed")
    
    

corruptions = ['fog']
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)

for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting',
                   'Carmon2019Unlabeled']:
    model = load_model(model_name, dataset='cifar10', threat_model='Linf')
    acc = clean_accuracy(model, x_test, y_test)
    print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')