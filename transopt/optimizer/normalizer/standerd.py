import numpy as np

from sklearn.preprocessing import StandardScaler

from transopt.agent.registry import normalizer_registry
from transopt.optimizer.normalizer.normalizer_base import NormalizerBase



# class XScaler:
#     def __init__(self, ranges):
#         self.ranges = np.array(ranges)
#         self.min = self.ranges[:, 0]
#         self.max = self.ranges[:, 1]
    
    
#     def transform(self, values):
#         values = np.array(values)
#         scaled_values = 2 * (values - self.min) / (self.max - self.min) - 1
#         return scaled_values
    
#     def inverse_transform(self, scaled_values):
#         scaled_values = np.array(scaled_values)
#         values = (scaled_values + 1) / 2 * (self.max - self.min) + self.min
#         return values
    
    
@normalizer_registry.register("Standard")
class Standard_normalizer(NormalizerBase):
    def __init__(self, config, metadata =  None, metadata_info = None):
        self.y_normalizer = StandardScaler()
        super(Standard_normalizer, self).__init__(config)
        

    def fit(self, X, Y):
        self.y_normalizer.fit(Y)
            
    def transform(self, X = None, Y = None):
        # if X is not None:
        #     X = self.x_normalizer.transform(X)
        if Y is not None:
            Y = self.y_normalizer.transform(Y)
        return X, Y

    def inverse_transform(self, X = None, Y = None):
        # if X is not None:
        #     X = self.x_normalizer.inverse_transform(X)
        if Y is not None:
            Y = self.y_normalizer.inverse_transform(Y)
        return X, Y