# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
from transopt.benchmark.HPO import networks
from sklearn.linear_model import SGDClassifier
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import SGD

from transopt.benchmark.HPO.augmentation import mixup_data, mixup_criterion

ALGORITHMS = [
    'ERM',
    'GLMNet',
    'BayesianNN',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, architecture, model_size, mixup, device, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.architecture = architecture
        self.model_size = model_size
        self.device = device
        self.mixup = mixup
        if self.mixup:
            self.mixup_alpha = self.hparams.get('mixup_alpha', 0.3)

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, architecture, model_size, mixup, device, hparams):
        super(ERM, self).__init__(input_shape, num_classes, architecture, model_size,  mixup, device, hparams)
        self.featurizer = networks.Featurizer(input_shape, architecture, model_size, self.hparams)
        print(self.featurizer.n_outputs)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['dropout_rate'],
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            momentum=self.hparams['momentum']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        if self.mixup:
            all_x, all_y_a, all_y_b, lam = mixup_data(all_x, all_y, self.mixup_alpha,  self.device)
            all_x, all_y_a, all_y_b = map(torch.autograd.Variable, (all_x, all_y_a, all_y_b))

        predictions = self.predict(all_x)

        if self.mixup:
            loss = mixup_criterion(F.cross_entropy, predictions, all_y_a, all_y_b, lam)
        else:
            loss = F.cross_entropy(predictions, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.mixup:
            correct = (lam * predictions.argmax(1).eq(all_y_a).float() +
                       (1 - lam) * predictions.argmax(1).eq(all_y_b).float()).sum().item()
        else:
            correct = (predictions.argmax(1) == all_y).sum().item()

        return {'loss': loss.item(), 'correct': correct}

    def predict(self, x):
        return self.network(x)

class GLMNet(Algorithm):
    """
    Generalized Linear Model with Elastic Net Regularization (GLMNet)
    """

    def __init__(self, input_shape, num_classes, architecture, model_size, mixup, device, hparams):
        super(GLMNet, self).__init__(input_shape, num_classes, architecture, model_size,  mixup, device, hparams)
        self.featurizer = networks.Featurizer(input_shape, architecture, model_size, self.hparams)
        self.num_classes = num_classes
        
        # 使用 SGDClassifier 作为 GLMNet
        self.classifier = SGDClassifier(
            loss='log',  # 对数损失，用于分类
            penalty='elasticnet',  # 弹性网络正则化
            alpha=self.hparams['glmnet_alpha'],  # 正则化强度
            l1_ratio=self.hparams['glmnet_l1_ratio'],  # L1 正则化的比例
            learning_rate='optimal',
            max_iter=1,  # 每次更新只进行一次迭代
            warm_start=True,  # 允许增量学习
            random_state=self.hparams['random_seed']
        )
        
        self.optimizer = torch.optim.SGD(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            momentum=self.hparams['momentum']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        # 提取特征
        features = self.featurizer(all_x).detach().cpu().numpy()
        labels = all_y.cpu().numpy()
        
        # 更新 GLMNet 分类器
        self.classifier.partial_fit(features, labels, classes=np.arange(self.num_classes))
        
        # 计算损失（仅用于记录，不用于反向传播）
        loss = -self.classifier.score(features, labels)
        
        # 更新特征提取器
        self.optimizer.zero_grad()
        features = self.featurizer(all_x)
        logits = torch.tensor(self.classifier.decision_function(features.detach().cpu().numpy())).to(all_x.device)
        feature_loss = F.cross_entropy(logits, all_y)
        feature_loss.backward()
        self.optimizer.step()

        return {'loss': loss, 'feature_loss': feature_loss.item()}

    def predict(self, x):
        features = self.featurizer(x).detach().cpu().numpy()
        return torch.tensor(self.classifier.predict_proba(features)).to(x.device)

class BayesianNN(Algorithm):
    """
    Two-layer Bayesian Neural Network
    """

    def __init__(self, input_shape, num_classes, hparams):
        super(BayesianNN, self).__init__(input_shape, num_classes, None, None, hparams)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.hidden_dim1 = hparams['bayesian_hidden_dim1']
        self.hidden_dim2 = hparams['bayesian_hidden_dim2']
        self.output_dim = num_classes
        self.num_samples = hparams['bayesian_num_samples']

        # Initialize parameters
        self.w1_mu = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim1))
        self.w1_sigma = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim))
        self.w2_mu = nn.Parameter(torch.randn(self.hidden_dim2, self.output_dim))
        self.w2_sigma = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim))

        # Setup Pyro optimizer
        self.optimizer = SGD({
            "lr": hparams["step_length"],
            "weight_decay": hparams["weight_decay"],
            "momentum": hparams["momentum"]
        })
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        
        self.burn_in = hparams['burn_in']
        self.step_count = 0

    def model(self, x, y=None):
        # First layer
        w1 = pyro.sample("w1", dist.Normal(self.w1_mu, torch.exp(self.w1_sigma)).to_event(2))
        h = F.relu(x @ w1)

        # Second layer
        w2 = pyro.sample("w2", dist.Normal(self.w2_mu, torch.exp(self.w2_sigma)).to_event(2))
        logits = h @ w2

        # Observe data
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    def guide(self, x, y=None):
        # First layer
        w1 = pyro.sample("w1", dist.Normal(self.w1_mu, torch.exp(self.w1_sigma)).to_event(2))

        # Second layer
        w2 = pyro.sample("w2", dist.Normal(self.w2_mu, torch.exp(self.w2_sigma)).to_event(2))

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x.view(x.size(0), -1) for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # Perform SVI step
        loss = self.svi.step(all_x, all_y)
        
        self.step_count += 1

        return {'loss': loss}

    def predict(self, x):
        x = x.view(x.size(0), -1)
        num_samples = self.num_samples

        if self.step_count <= self.burn_in:
            # During burn-in, use point estimates
            w1 = self.w1_mu
            w2 = self.w2_mu
            h = F.relu(x @ w1)
            logits = h @ w2
            return F.softmax(logits, dim=-1)
        else:
            # After burn-in, use full Bayesian prediction
            def wrapped_model(x_data):
                pyro.sample("prediction", dist.Categorical(logits=self.model(x_data)))

            posterior = pyro.infer.Predictive(wrapped_model, guide=self.guide, num_samples=num_samples)(x)
            predictions = posterior["prediction"]
            return predictions.float().mean(0)

