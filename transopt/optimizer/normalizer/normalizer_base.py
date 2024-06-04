from abc import abstractmethod, ABC
from typing import Dict, Hashable
import numpy as np

class NormalizerBase(ABC):
    def __init__(self, config):
        self.config = config
    @abstractmethod
    def fit(self, X, Y):
        raise NotImplementedError
    @abstractmethod 
    def transform(self, X = None, Y = None):
        raise NotImplementedError
    @abstractmethod
    def inverse_transform(self, X = None, Y = None):

        raise NotImplementedError 
    