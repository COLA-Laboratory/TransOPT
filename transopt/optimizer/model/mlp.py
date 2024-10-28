import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets.utils as dataset_utils
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from transopt.agent.registry import model_registry
from transopt.optimizer.model.model_base import Model


def compute_irm_penalty(losses, dummy):
    g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
    g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
    return (g1 * g2).sum()

class Net(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        return logits


    
@model_registry.register('MLP')
class MLP(Model):
    def __init__(self, config):
        super().__init__()
        self._model = None
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._batch_size = 16
        self._dropout_rate = 0.3
        self._best_model_state = None
        self._best_val_loss = float('inf')
    
    def meta_fit(
        self,
        source_X : List[np.ndarray],
        source_Y : List[np.ndarray],
        **kwargs,
    ):
        pass

    def fit(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        epochs : int = 50,
        optimize: bool = False,
    ):
        self._X = np.copy(X)
        self._y = np.copy(Y)
        self._Y = np.copy(Y)

        _X = np.copy(self._X)
        _y = np.copy(self._y)
        
        X_tensor = torch.tensor(_X, dtype=torch.float32)
        y_tensor = torch.tensor(_y, dtype=torch.float32).view(-1, 1)
        
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        
        patience = 5
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            if self._model is None or patience_counter >= patience:
                self._model = Net(input_dim=X_train.shape[1], dropout_rate=self._dropout_rate).to(self.device)
                self._optimizer = optim.Adam(self._model.parameters(), lr=0.0001, weight_decay=1e-5)
                patience_counter = 0

            self._model.train()
            train_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self._optimizer.zero_grad()
                output = self._model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                self._optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(loss.item())

            self._model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self._model(data)
                    loss = F.mse_loss(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(loss.item())
            
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_model_state = self._model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
        
        if self._best_model_state:
            self._model.load_state_dict(self._best_model_state)
        self.save_plots(train_losses, val_losses, X_val_tensor, y_val_tensor, 'output_plots', iter_num=_X.shape[0])


        
    def predict(
        self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        data = torch.tensor(X, dtype=torch.float32).to(self.device)
        self._model.eval()
        with torch.no_grad():
            output = self._model(data)
        output = output.to('cpu')
        output = output.numpy()
        variance = np.zeros(shape=(output.shape[0], 1))
        return output, variance
        
    def get_fmin(self):
        
        return np.min(self._y)
    
    
    def save_plots(self, train_losses, val_losses, X_val, y_val, output_dir, iter_num):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存损失曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation Loss Over Epochs')
        plt.savefig(os.path.join(output_dir, f'loss_plot_{iter_num}.png'))
        plt.close()

        # 保存预测值与真实值的对比图
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_val.to(self.device)).cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(y_val)), y_val.cpu().numpy(), label='True Values')
        plt.plot(range(len(predictions)), predictions, label='Predictions')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.legend()
        plt.title('Predictions vs True Values')
        plt.savefig(os.path.join(output_dir, f'predictions_vs_true_plot_{iter_num}.png'))
        plt.close()