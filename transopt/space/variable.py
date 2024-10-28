import math
import numpy as np

class Variable:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_

    @property
    def search_space_range(self):
        raise NotImplementedError

    def map2design(self, value):
        # To design space
        raise NotImplementedError

    def map2search(self, value):
        # To search space
        raise NotImplementedError


class Continuous(Variable):
    def __init__(self, name, range_):
        super().__init__(name, "continuous")
        self.range = range_
        
        self.is_discrete = False

    @property
    def search_space_range(self):
        return self.range

    def map2design(self, value):
        return float(value)  # Ensure it remains a float
    
    def map2search(self, value):
        return value


class Categorical(Variable):
    def __init__(self, name, categories):
        super().__init__(name, "categorical")
        self.categories = categories
        self.range = (1, len(self.categories))
        
        self.is_discrete = True

    @property
    def search_space_range(self):
        return (1, len(self.categories))

    def map2design(self, value):
        return self.categories[round(value) - 1]

    def map2search(self, value):
        return self.categories.index(value) + 1
    


class Integer(Variable):
    def __init__(self, name, range_):
        super().__init__(name, "integer")
        self.range = range_

        self.is_discrete = True

    @property
    def search_space_range(self):
        return self.range

    def map2design(self, value):
        # Ensure the mapped value is an integer
        return int(round(value)) 

    def map2search(self, value):
        return round(value)

class LargeInteger(Variable):
    def __init__(self, name, range_):
        super().__init__(name, "large_integer")
        self.range = range_
        self.is_discrete = True

    @property
    def search_space_range(self):
        # Convert large range to a manageable float range
        lower = 0
        upper = 1
        return lower, upper

    def map2design(self, value):
        # Map float value [0, 1] to the large integer range
        return min(int(self.range[0] + value * (self.range[1] - self.range[0])), self.range[1])

    def map2search(self, value):
        # Map large integer value to a float value in [0, 1]
        return (value - self.range[0]) / (self.range[1] - self.range[0])

class ExponentialInteger(Variable):
    def __init__(self, name, range_):
        super().__init__(name, "exp2")
        # Adjust the range to ensure it is in the form of [2^x, 2^y] and satisfies 2^63 - 1
        lower_bound = 2 ** math.floor(math.log2(range_[0]))
        upper_bound = min(2 ** math.ceil(math.log2(range_[1])), 2 ** 63)
        self.range = (lower_bound, upper_bound)
        self.is_discrete = True

    @property
    def search_space_range(self):
        lower = math.log2(self.range[0])
        upper = math.log2(self.range[1])
        return lower, upper

    def map2design(self, value):
        return int(2 ** value)

    def map2search(self, value):
        value = max(value, self.range[0])  # Ensure value is within valid range
        return math.log2(value)
    
class LogContinuous(Variable):
    def __init__(self, name, range_):
        super().__init__(name, "log_continuous")
        self.range = range_
        
        self.is_discrete = False

    @property
    def search_space_range(self):
        return self.range

    def map2design(self, value):
        return 10**value

    def map2search(self, value):
        return math.log10(value)

