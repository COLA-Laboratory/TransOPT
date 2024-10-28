# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models


SUPPORTED_ARCHITECTURES = {
    'resnet': [18, 34, 50, 101],
    'densenet': [121, 169, 201],
    'wideresnet': [16, 22, 28, 40],
    'alexnet': [1],
    'cnn': [1]
}

def Featurizer(input_shape, architecture, model_size, hparams):
    """Select an appropriate featurizer based on the input shape and hparams."""

    if architecture == 'densenet':
        return DenseNet(input_shape, model_size, hparams)
    elif architecture == 'resnet':
        return ResNet(input_shape, model_size, hparams)
    elif architecture == 'wideresnet':
        return WideResNet(input_shape, model_size, hparams)
    elif architecture == 'alexnet':
        return AlexNet(input_shape, hparams)
    elif architecture == 'cnn':
        return CNN(input_shape, hparams)
    else:
        raise ValueError(f"Unsupported network architecture: {architecture}")
    
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['dropout_rate'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        # x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, model_size, hparams):
        super(ResNet, self).__init__()
        if model_size == 18:
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
        elif model_size == 101:
            self.network = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
            self.n_outputs = 2048
        elif model_size == 34:
            self.network = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
        elif model_size == 50:
            self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            self.n_outputs = 2048
        else:
            raise ValueError(f"Unsupported ResNet model size: {model_size}")
        

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """WideResNet with the softmax layer removed"""
    def __init__(self, input_shape, model_size, hparams):
        super(WideResNet, self).__init__()
        
        # Define configurations for different model sizes
        configs = {
            28: (28, 10),  # WRN-28-10
            16: (16, 8),   # WRN-16-8
            40: (40, 2),   # WRN-40-2
            22: (22, 2)    # WRN-22-2
        }
        
        if model_size not in configs:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(configs.keys())}")
        
        self.depth, self.widen_factor = configs[model_size]
        self.nChannels = [16, 16*self.widen_factor, 32*self.widen_factor, 64*self.widen_factor]
        self.in_planes = 16

        assert ((self.depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (self.depth-4) // 6
        self.n_outputs = self.nChannels[3]
        self.dropout = hparams['dropout_rate']

        self.conv1 = nn.Conv2d(input_shape[0], self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(wide_basic, self.nChannels[1], n, self.dropout, stride=1)
        self.layer2 = self._wide_layer(wide_basic, self.nChannels[2], n, self.dropout, stride=2)
        self.layer3 = self._wide_layer(wide_basic, self.nChannels[3], n, self.dropout, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nChannels[3], momentum=0.9)
    

    def _wide_layer(self, block, planes, num_blocks, dropout, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out


class DenseNet(nn.Module):
    """DenseNet with the softmax layer removed"""
    def __init__(self, input_shape, model_size, hparams):
        super(DenseNet, self).__init__()
        self.model_size = model_size
        if self.model_size == 121:
            self.network = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
            self.n_outputs = 1024
        elif self.model_size == 169:
            self.network = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)
            self.n_outputs = 1664
        elif self.model_size == 201:
            self.network = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1)
            self.n_outputs = 1920
        else:
            raise ValueError("Unsupported DenseNet depth. Choose from 121, 169, or 201.")

        # Adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            self.network.features.conv0 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the last fully connected layer
        self.network.classifier = Identity()

        # self.dropout = nn.Dropout(hparams['dropout_rate'])

    def forward(self, x):
        features = self.network(x)
        return features
    
    
    

class ht_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(ht_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

class CNN(nn.Module):
    """
    Two-layer CNN with hidden dimensions determined by hparams.
    """
    def __init__(self, input_shape, hparams):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hparams['hidden_dim1'], 3, 1, padding=1)
        self.conv2 = nn.Conv2d(hparams['hidden_dim1'], hparams['hidden_dim2'], 3, 1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(hparams['hidden_dim1'])
        self.bn2 = nn.BatchNorm2d(hparams['hidden_dim2'])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.n_outputs = hparams['hidden_dim2']

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)

class AlexNet(nn.Module):
    """AlexNet with the classifier layer removed"""
    def __init__(self, input_shape, hparams):
        super(AlexNet, self).__init__()
        self.input_shape = input_shape
        self.hparams = hparams
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Calculate the correct n_outputs
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            features_output = self.features(dummy_input)
            avgpool_output = self.avgpool(features_output)
            self.n_outputs = avgpool_output.view(avgpool_output.size(0), -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



def Classifier(in_features, out_features, dropout=0.5, is_nonlinear=False):
    if is_nonlinear:
        hidden1 = max(in_features // 2, 64)  # Ensure at least 64 neurons
        hidden2 = max(hidden1 // 2, 32)      # Ensure at least 32 neurons
        return torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, out_features)
            )
    else:
        return torch.nn.Linear(in_features, out_features)

