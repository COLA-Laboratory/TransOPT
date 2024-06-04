
from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import normalizer_registry
from transopt.optimizer.normalizer.normalizer_base import NormalizerBase


@normalizer_registry.register("Standard")
class Standard_normalizer(NormalizerBase):
    def __init__(self, config, metadata =  None, metadata_info = None):
        self.y_normalizer = StandardScaler()
        self.x_normalizer = StandardScaler()
        

    def fit(self, X, Y):
        self.x_normalizer.fit(X)
        self.y_normalizer.fit(Y)
            
    def transform(self, X = None, Y = None):
        if X:
            X = self.x_normalizer.transform(X)
        if Y:
            Y = self.y_normalizer.transform(Y)
        return X, Y

    def inverse_transform(self, X = None, Y = None):
        if X:
            X = self.x_normalizer.inverse_transform(X)
        if Y:
            Y = self.y_normalizer.inverse_transform(Y)
        return X, Y