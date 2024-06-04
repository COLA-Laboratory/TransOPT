import time
import logging
import numpy as np
import ConfigSpace as CS
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict
from torchvision import datasets, transforms

from transopt.space.variable import *
from transopt.agent.registry import problem_registry
from transopt.benchmark.problem_base.non_tab_problem import NonTabularProblem
from transopt.space.search_space import SearchSpace
from transopt.space.fidelity_space import FidelitySpace

logger = logging.getLogger("ConfigOptBenchmark")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_block, kernel_size=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        block = BasicBlock
        num_blocks = [num_block] * 4

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@problem_registry.register("Res")
class HPOResNet(NonTabularProblem):
    DATASET_NAME = ["svhn", "cifar10", "cifar100"]
    problem_type = 'hpo'
    num_variables = 10
    num_objectives = 1
    workloads = []
    fidelity = None

    def __init__(
        self, task_name, budget_type, budget, seed, workload, **kwargs
    ):
        super(HPOResNet, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset_name = HPOResNet.DATASET_NAME[workload]

    def get_configuration_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        variables=[Continuous('lr', [-10.0, 0.0]),
                   Continuous('momentum', [0.0, 1.0]),
                   Continuous('weight_decay', [-10.0, -5.0]),
                   Integer('number_block', [1, 5]),]
        ss = SearchSpace(variables)
        return ss

    def get_fidelity_space(
        self, seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        # seed = seed if seed is not None else np.random.randint(1, 100000)
        # fidel_space = CS.ConfigurationSpace(seed=seed)

        # fidel_space.add_hyperparameters(
        #     [
        #         CS.UniformFloatHyperparameter(
        #             "dataset_fraction", lower=0.0, upper=1.0, log=False
        #         ),
        #         CS.UniformIntegerHyperparameter(
        #             "epochs", lower=1, upper=300, log=False
        #         ),
        #     ]
        # )

        # return fidel_space
        fs = FidelitySpace([])
        return fs

    def get_meta_information(self) -> Dict:
        print(1)
        return {}

    def load_data(self, batch_size=64):
        dataset_name = self.dataset_name
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if dataset_name == "svhn":
            trainset = datasets.SVHN(
                root="./data", split="train", download=True, transform=transform
            )
            testset = datasets.SVHN(
                root="./data", split="test", download=True, transform=transform
            )
        elif dataset_name == "cifar10":
            trainset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "cifar100":
            trainset = datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform
            )
            testset = datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        if dataset_name == "svhn" or dataset_name == "cifar10":
            n_outputs = 10
        elif dataset_name == "cifar100":
            n_outputs = 100

        return trainloader, testloader, n_outputs

    def get_score(self, configuration: dict):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(1)
        else:
            device = torch.device("cpu")
        epochs = 30
        batch_size = 64

        lr = configuration["lr"]
        momentum = configuration["momentum"]

        trainloader, testloader, n_output = self.load_data(batch_size=batch_size)

        net = ResNet(
            num_block=int(configuration["number_block"]),
            kernel_size=3,
            num_classes=n_output,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=configuration["weight_decay"],
        )
        start_time = time.time()
        for e in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("Epoch %d, Loss: %.3f" % (e + 1, running_loss / len(trainloader)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        end_time = time.time()
        print("Accuracy: %.2f %%" % (100 * accuracy))

        return accuracy, end_time - start_time

    def objective_function(
        self,
        configuration: Union[CS.Configuration, Dict],
        fidelity: Union[Dict, CS.Configuration, None] = None,
        seed: Union[np.random.RandomState, int, None] = None,
        **kwargs
    ) -> Dict:
        if fidelity is None:
            fidelity = {"epoch": 30, "data_frac": 0.8}
        c = {
            "momentum": configuration["momentum"],
            "lr": np.exp2(configuration["lr"]),
            "weight_decay": np.exp2(configuration["weight_decay"]),
            "number_block": configuration["number_block"],
            "batch_size": 64,
            "epoch": fidelity["epoch"],
            "data_frac": fidelity["data_frac"],
        }
        acc, time = self.get_score(c)
        # return {
        #     "function_value": float(1 - acc),
        #     "cost": time,
        #     "info": {"fidelity": fidelity},
        # }

        results = {list(self.objective_info.keys())[0]: float(1 - acc)}
        for fd_name in self.fidelity_space.fidelity_names:
            results[fd_name] = fidelity[fd_name] 
        return results
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"


if __name__ == "__main__":
    mlp = HPOResNet(
        task_name="Res", budget=40, seed=0, task_type="non-tabular", dataset_name="svhn"
    )
    configuration = {
        "momentum": -0.1,
        "lr": -0.3,
        "weight_decay": 0.9,
        "number_block": 0.3,
    }
    mlp.f(configuration=configuration)
