from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from transopt.agent.registry import model_registry
from transopt.optimizer.model.model_base import Model


class RegressionDataset(Dataset):
    """create a dataset that complies with PyTorch """
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y


class RbfNet(nn.Module):
    def __init__(self, centers, beta):
        super(RbfNet, self).__init__()
        self.num_centers = centers.size(0)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(beta)
        self.linear = nn.Linear(self.num_centers, 1)
        nn.init.xavier_uniform_(self.linear.weight)

    def kernel_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdims=False).sqrt()))
        return C

    def forward(self, x):
        x = self.kernel_fun(x)
        x = self.linear(x)
        return x


class rbfn(object):
    def __init__(self, dataset, max_epoch=30, batch_size=5, lr=0.01, num_centers=5, show_details=False):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.num_centers = num_centers
        self.dim = dataset.inputs.shape[1]

        # create the DataLoader for training
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=1)

        # cluster
        self.centers = self.cluster()
        self.beta = self.calculate_beta()
        # create Rbf network
        self.model = RbfNet(self.centers, self.beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()
        self.avg_loss = 0
        self.show_details = show_details

    def train(self):
        self.model.train()
        for epoch in range(self.max_epoch):
            self.avg_loss = 0
            total_batch = len(self.dataset) // self.batch_size

            for i, (input, output) in enumerate(self.data_loader):
                X = Variable(input.view(-1, self.dim))
                Y = Variable(output)

                self.optimizer.zero_grad()
                Y_prediction = self.model(X)
                loss = self.loss_fun(Y_prediction, Y)
                loss.backward()
                self.optimizer.step()
                self.avg_loss += loss / total_batch
            if self.show_details:
                print("[Epoch: {:>4}] loss = {:>.9}".format(epoch + 1, self.avg_loss))
        print("[*] Training finished! Loss: {:.9f}".format(self.avg_loss))

    def predict(self, x):
        self.model.eval()
        x = torch.from_numpy(x)
        x = Variable(x)
        y = self.model(x)
        return y.data.numpy()

    def cluster(self):
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(self.dataset.inputs)
        centers = kmeans.cluster_centers_
        return torch.from_numpy(centers)

    def calculate_beta(self):
        r2 = torch.ones(1, self.num_centers)
        for i, center in enumerate(self.centers):
            distances = torch.linalg.norm(self.centers - center, axis=1)
            nearest_two_neighbors_indices = torch.argsort(distances)[:2]
            r2[0][i] = torch.sum(distances[nearest_two_neighbors_indices]**2) / 2
        beta = 1 / r2
        return beta

    def update_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=1)



@model_registry.register('RBFN')
class RBFN(Model):
    def __init__(
        self,
        max_epoch: int = 30,
        batch_size: int = 1,
        lr: float = 0.01,
        num_centers: int = 10,
        show_details: bool = False,
        normalize: bool = True,
        **options: dict
    ):
        super().__init__()
        self._max_epoch = max_epoch
        self._batch_size = batch_size
        self._lr = lr
        self._num_centers = num_centers
        self._rbfn_model = None
        self._show_details = show_details

        self._normalize = normalize
        self._x_normalizer = StandardScaler() if normalize else None
        self._y_normalizer = StandardScaler() if normalize else None

        self._options = options

    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimize: bool = True,
    ):
        self._X = np.copy(X)
        self._y = np.copy(Y)
        self._Y = np.copy(Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)

        if self._normalize:
            _X = self._x_normalizer.fit_transform(_X)
            _y = self._y_normalizer.fit_transform(_y)

        if self._rbfn_model is None:
            dataset = RegressionDataset(torch.from_numpy(_X), torch.from_numpy(_y))
            self._rbfn_model = rbfn(
                dataset=dataset,
                max_epoch=self._max_epoch,
                batch_size=self._batch_size,
                lr=self._lr,
                num_centers=self._num_centers,
                show_details=self._show_details,
            )
        else:
            dataset = RegressionDataset(torch.from_numpy(_X), torch.from_numpy(_y))
            self._rbfn_model.update_dataset(dataset)
        
        try:
            self._rbfn_model.train()
        except np.linalg.LinAlgError as e:
            print('Error: np.linalg.LinAlgError')

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[None, :]
        
        Y = self._rbfn_model.predict(X)
        return Y, None