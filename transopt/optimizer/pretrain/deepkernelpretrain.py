import copy
import os

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from transopt.agent.registry import pretrain_registry
from transopt.optimizer.pretrain.pretrain_base import PretrainBase

np.random.seed(1203)
RandomQueryGenerator= np.random.RandomState(413)
RandomSupportGenerator= np.random.RandomState(413)
RandomTaskGenerator = np.random.RandomState(413)



class Metric(object):
    def __init__(self,prefix='train: '):
        self.reset()
        self.message=prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.2f}"
        
    def update(self,loss,noise,mse):
        self.loss.append(loss.item())
        self.noise.append(noise.item())
        self.mse.append(mse.item())
    
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
    
    

@pretrain_registry.register("DeepKernelPretrain")
class DeepKernelPretrain(nn.Module):
    def __init__(self, config = {}):
        super(DeepKernelPretrain, self).__init__()
        ## GP parameters
        if len(config) == 0:
            self.config = {"kernel": "matern", 'ard': False, "nu": 2.5, 'hidden_size': [32,32,32,32],
                           'n_inner_steps': 1, 'test_batch_size':1, 'batch_size':1, 'seed':0, 'checkpoint_path':'./external/model/FSBO/'}
        else:
            self.config = config
            
        self.batch_size = self.config['batch_size']
        self.test_batch_size = self.config['test_batch_size']
        self.n_inner_steps = self.config['n_inner_steps']
        self.checkpoint_path = self.config['checkpoint_path']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = [32,32,32,32]
        self.kernel_config = {"kernel": self.config['kernel'], 'ard': self.config['ard'], "nu": self.config['nu']}
        self.Seed = self.config['seed']

        self.train_metrics = Metric()
        self.valid_metrics = Metric(prefix="valid: ")
        self.mse        = nn.MSELoss()
        self.curr_valid_loss = np.inf
        os.makedirs(self.checkpoint_path,exist_ok=True)

        print(self)

    def set_data(self, metadata, metadata_info= None):
        
        train_data = {}
        for dataset_name, data in metadata.items():
            objectives = metadata_info[dataset_name]["objectives"]
            obj = objectives[0]["name"]

            obj_data = [d[obj] for d in data]
            var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
            self.input_size = metadata_info[dataset_name]['num_variables']
            train_data[dataset_name] = {'X':np.array(var_data), 'y':np.array(obj_data)[:, np.newaxis]}
            
        self.train_data = train_data
        self.feature_extractor =  MLP(self.input_size, hidden_size = self.hidden_size).to(self.device)
        self.get_tasks()

    def get_tasks(self,):
        self.tasks = list(self.train_data.keys())


    def get_model_likelihood_mll(self, train_size):
        
        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x = train_x, train_y = train_y, likelihood = likelihood, config = self.kernel_config, dims = self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)


    def epoch_end(self):
        RandomTaskGenerator.shuffle(self.tasks)


    def meta_train(self, epochs = 50000, lr = 0.0001):
        self.get_model_likelihood_mll(self.batch_size)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)
        

        for epoch in range(epochs):
            self.train_loop(epoch, optimizer, scheduler)
        self.save_checkpoint(self.checkpoint_path + f'Seed_{self.Seed}_{len(self.tasks)}')
    def train_loop(self, epoch, optimizer, scheduler=None):
        self.epoch_end()
        assert(self.training)
        for task in self.tasks:
            inputs, labels = self.get_batch(task)
            for _ in range(self.n_inner_steps):
                optimizer.zero_grad()
                z = self.feature_extractor(inputs)
                self.model.set_train_data(inputs=z, targets=labels, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
                mse = self.mse(predictions.mean, labels)
                self.train_metrics.update(loss,self.model.likelihood.noise,mse)
        if scheduler:
            scheduler.step()
        
        training_results = self.train_metrics.get()
            
        validation_results = self.valid_metrics.get()
        # for k,v in validation_results.items():
        #     self.valid_summary_writer.add_scalar(k, v, epoch)
        self.feature_extractor.train()
        self.likelihood.train()
        self.model.train()
        
        if validation_results["loss"] < self.curr_valid_loss:
            self.save_checkpoint(os.path.join(self.checkpoint_path,"weights"))
            self.curr_valid_loss = validation_results["loss"]
        self.valid_metrics.reset()       
        self.train_metrics.reset()
            
    def test_loop(self, task, train): 
        (x_support, y_support),(x_query,y_query) = self.get_support_and_queries(task,train)
        z_support = self.feature_extractor(x_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()        
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            loss = -self.mll(pred, y_query)
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query)

        return mse,loss

    def get_batch(self,task):

        Lambda,response =     np.array(self.train_data[task]["X"]), MinMaxScaler().fit_transform(np.array(self.train_data[task]["y"])).reshape(-1,)

        card, dim = Lambda.shape
        
        support_ids = RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size= min(self.batch_size, card))

        
        inputs,labels = Lambda[support_ids], response[support_ids]
        inputs,labels = totorch(inputs,device=self.device), totorch(labels.reshape(-1,),device=self.device)
        return inputs, labels
        
    def get_support_and_queries(self,task, train=False):
        

        hpo_data = self.valid_data if not train else self.train_data
        Lambda,response =     np.array(hpo_data[task]["X"]), MinMaxScaler().fit_transform(np.array(hpo_data[task]["y"])).reshape(-1,)
        card, dim = Lambda.shape

        support_ids = RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size=min(self.batch_size, card))
        diff_set = np.setdiff1d(np.arange(card),support_ids)
        query_ids = RandomQueryGenerator.choice(diff_set,replace=False,size=min(self.batch_size, len(diff_set)))
        
        support_x,support_y = Lambda[support_ids], response[support_ids]
        query_x,query_y = Lambda[query_ids], response[query_ids]
        
        return (totorch(support_x,self.device),totorch(support_y.reshape(-1,),self.device)),\
    (totorch(query_x,self.device),totorch(query_y.reshape(-1,),self.device))
        
    def save_checkpoint(self, checkpoint):

        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
        
