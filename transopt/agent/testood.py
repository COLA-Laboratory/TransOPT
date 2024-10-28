import logging
import time
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.optimizer.sampler.random import RandomSampler
from transopt.space.fidelity_space import FidelitySpace
from transopt.space.search_space import SearchSpace
from transopt.space.variable import *
from transopt.utils.openml_data_manager import OpenMLHoldoutDataManager
from transopt.datamanager.database import Database
from services import Services

from transopt.benchmark.HPO.HPOCNN import *

import os
import sys
import unittest
from pathlib import Path

def plot_acc_scatter(train_acc, test_acc):
    # Create a scatter plot
    plt.scatter(train_acc, test_acc, label='Accuracy Points')
    
    # Plot the diagonal line
    min_val = min(min(train_acc), min(test_acc))
    max_val = max(max(train_acc), max(test_acc))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Diagonal Line')
    
    # Add labels and title
    plt.xlabel('Train Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    
    # Show the plot
    plt.savefig('./train vs test accuracy.png')





current_dir = Path(__file__).resolve().parent
package_dir = current_dir.parent
sys.path.insert(0, str(package_dir))

def setUp():
    db = Database("database.db")
    table_name = "test_table"
        
def list_pth_files(directory):
    # Create an empty list to store .pth file paths
    pth_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file ends with .pth
            if file.endswith('.pth'):
                # Create the full path to the file
                file_path = os.path.join(root, file)
                # Add the file path to the list
                pth_files.append(file_path)

    return pth_files

if __name__ == "__main__":

    services = Services(None, None, None)
    task_name = []
    parameters = []
    # tables = services.get_experiment_datasets()
    # for table in tables:
    #     print(table[1]['data_number'])
    #     if table[1]['data_number'] == 100:
    #         task_name = table[0]
    #         print(task_name)

    #         all_data = services.data_manager.db.select_data(task_name)
    #         table_info = services.data_manager.db.query_dataset_info(task_name)
                    
    #         objectives = table_info["objectives"]
    #         ranges = [tuple(var['range']) for var in table_info["variables"]]
    #         initial_number = table_info["additional_config"]["initial_number"]
    #         obj = objectives[0]["name"]
    #         obj_type = objectives[0]["type"]

    #         obj_data = [data[obj] for data in all_data]
    #         max_id = np.argmax(obj_data)
            
    #         var_data = [[data[var["name"]] for var in table_info["variables"]] for data in all_data]
    #         variables = [var["name"] for var in table_info["variables"]]
    #         ret = {}
    #         traj = services.construct_trajectory_data(task_name, obj_data, obj_type="maximize")
    #         best_var = var_data[max_id]
    #         lr = np.exp2(best_var[0])
    #         momentum = best_var[1]
    #         weight_decay = np.exp2(best_var[2])
    #         parameters.append((lr, momentum, weight_decay))
    
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(1)
    else:
        device = torch.device("cpu")
        
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.Compose(
            [
                BGRed(),
                
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.Compose(
            [
                BGRed(),
                
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
    
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False
    )
    epochs = 30
    batch_size = 64

    # lr = 0.0017607943222948076
    # momentum = 0.6997583600209312
    # weight_decay = 0.004643925899318933
    
    # lr = parameters[0][0]
    # momentum = parameters[0][1]
    # weight_decay = parameters[0][2]
    # print(lr, momentum, weight_decay)
    
    directory = './temp_model/CNN_101/'  # Replace with the path to your directory
    pth_files = list_pth_files(directory)
    train_acc = []
    test_acc = []

    net = Learner(target_classes=10).to(device)
    for model_name in pth_files:
        print(model_name)
        net.load_state_dict(torch.load(f'{model_name}'))
    # criterion = nn.NLLLoss()
    # optimizer = optim.SGD(
    #     net.parameters(),
    #     lr=lr,
    #     momentum=momentum,
    #     weight_decay = weight_decay,
    # )
    # start_time = time.time()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            
            accuracy = correct / total
            print("Training Accuracy: %.2f %%" % (100 * accuracy))
            train_acc.append(accuracy * 100)

            # print("Epoch %d, Loss: %.3f" % (e + 1, running_loss / len(trainloader)))

        correct = 0
        total = 0
        import os



        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        end_time = time.time()
        test_acc.append(accuracy * 100)
        print("Test Accuracy: %.2f %%" % (100 * accuracy))
        
    plot_acc_scatter(train_acc, test_acc)


            

            
        

    # model_save_path = './temp_model/model.pth'
    # torch.save(net.state_dict(), model_save_path)