import copy
import numpy as np
from typing import Dict, Hashable, Union, Sequence, Tuple, List

from transopt.optimizer.model.model_base import Model
from transopt.agent.registry import model_registry

@model_registry.register("NeuralProcess")
class NeuralProcess(Model):
    def __init__(self):
        super().__init__()
        
    