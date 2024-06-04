
"""
This FSBO implementation is based on the original implementation from Hadi Samer Jomaa
for his work on "Transfer Learning for Bayesian HPOBench with End-to-End Landmark Meta-Features"
at the NeurIPS 2021 MetaLearning Workshop 

The implementation for Deep Kernel Learning is based on the original Gpytorch example: 
https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

"""

import copy
import logging
import os

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from transopt.agent.registry import model_registry

np.random.seed(1203)
RandomQueryGenerator= np.random.RandomState(413)
RandomSupportGenerator= np.random.RandomState(413)
RandomTaskGenerator = np.random.RandomState(413)



class Metric(object):
    def __init__(self,prefix='train: '):
        self.reset()
        self.message=prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.2f}"
        
    def update(self,loss,noise,mse):
        self.loss.append(np.asscalar(loss))
        self.noise.append(np.asscalar(noise))
        self.mse.append(np.asscalar(mse))
    
    def reset(self,):
        self.loss = []
        self.noise = []
        self.mse = []
    
    def report(self):
        return self.message.format(loss=np.mean(self.loss),
                            log_var=np.mean(self.noise),
                            mse=np.mean(self.mse))
    
    def get(self):
        return {"loss":np.mean(self.loss),
                "noise":np.mean(self.noise),
                "mse":np.mean(self.mse)}
    

def totorch(x,device):

    return torch.Tensor(x).to(device)    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=[32,32,32,32], dropout=0.0):
        
        super(MLP, self).__init__()
        self.nonlinearity = nn.ReLU()
        self.fc = nn.ModuleList([nn.Linear(in_features=input_size, out_features=hidden_size[0])])
        for d_out in hidden_size[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))
        self.out_features = hidden_size[-1]
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        x = self.dropout(x)
        return x

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,config,dims ):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        if(config["kernel"]=='rbf' or config["kernel"]=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None))
        elif(config["kernel"]=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=config["nu"],ard_num_dims=dims if config["ard"] else None))
        else:
            raise ValueError("[ERROR] the kernel '" + str(config["kernel"]) + "' is not supported for regression, use 'rbf' or 'spectral'.")
            
    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
@model_registry.register("DeepKernelGP")
class DeepKernelGP(nn.Module):
    def __init__(self, config = {}):
        super(DeepKernelGP, self).__init__()

        if len(config) == 0:
            self.config = {"kernel": "matern", 'ard': False, "nu": 2.5, 'hidden_size': [32,32,32,32], 'n_inner_steps': 1,
                           'test_batch_size':1, 'batch_size':1, 'seed':0, 'eval_batch_size':1000, 'verbose':True, 'loss_tol':0.0001,
                           'max_patience':16, 'lr':0.001, 'epochs':100, 'load_model': False, 'checkpoint_path': './external/model/FSBO/Seed_0_1'}
        else:
            self.config = config
        torch.manual_seed(self.config['seed'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = self.config['hidden_size']
        self.kernel_config = {"kernel": self.config['kernel'], "ard": self.config['ard'], "nu": self.config['nu']}
        self.max_patience = self.config['max_patience']
        self.lr  = self.config['lr']
        self.load_model = self.config['load_model']
        self.checkpoint = self.config['checkpoint_path']
        
        self.epochs = self.config['epochs']
        self.verbose = self.config['verbose']
        self.loss_tol = self.config['loss_tol']
        self.eval_batch_size = self.config['eval_batch_size']
        self.has_model = False


    def get_model_likelihood_mll(self, train_size):
        
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.kernel_config,dims = self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)



    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            optimize: bool = False,):

        self.X_obs, self.y_obs = totorch(X, self.device), totorch(Y, self.device).reshape(-1)
        
        if self.load_model:
            assert(self.checkpoint is not None)
            print("Model_loaded")
            self.load_checkpoint(os.path.join(self.checkpoint,"weights"))

        if self.has_model == False:
            self.input_size = X.shape[1]
            self.feature_extractor = MLP(self.input_size, hidden_size = self.hidden_size).to(self.device)
            self.get_model_likelihood_mll(1)
            self.has_model = True
        
        losses = [np.inf]
        best_loss = np.inf
        weights = copy.deepcopy(self.state_dict())
        patience=0
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.lr},
                                {'params': self.feature_extractor.parameters(), 'lr': self.lr}])
                    
        for _ in range(self.epochs):
            optimizer.zero_grad()
            z = self.feature_extractor(self.X_obs)
            self.model.set_train_data(inputs=z, targets=self.y_obs, strict=False)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                raise e
            
            if self.verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_+1,epochs=self.epochs,loss=loss.item(),noise=self.likelihood.noise.item()))                
            losses.append(loss.detach().to("cpu").item())
            if best_loss>losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())
            if np.allclose(losses[-1],losses[-2],atol=self.loss_tol):
                patience+=1
            else:
                patience=0
            if patience>self.max_patience:
                break
        self.load_state_dict(weights)
        return losses
    
    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint,map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'],strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'],strict=False)
        self.feature_extractor.load_state_dict(ckpt['net'],strict=False)
        

    def predict(self, X_pen):
        
        query_X = totorch(X_pen, self.device)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()        

        z_support = self.feature_extractor(self.X_obs).detach()
        self.model.set_train_data(inputs=z_support, targets=self.y_obs, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(query_X).detach()
            pred    = self.likelihood(self.model(z_query))

            
        mu    = pred.mean.detach().to("cpu").numpy()[: ,np.newaxis]
        stddev = pred.stddev.detach().to("cpu").numpy()[: ,np.newaxis]
        
        return mu,stddev

    def continuous_maximization( self, dim, bounds, acqf):

        result = differential_evolution(acqf, bounds=bounds, updating='immediate',workers=1, maxiter=20000, init="sobol")
        return result.x.reshape(-1,dim)


    def get_fmin(self):
        return np.min(self.y_obs.detach().to("cpu").numpy())
