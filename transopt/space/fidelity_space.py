import copy

import numpy as np
import pandas as pd


class FidelitySpace:
    def __init__(self, fidelity_variables):
        self.ranges = {var.name: var for var in fidelity_variables}
    
    @property
    def fidelity_names(self):
        return self.ranges.keys()
    
    
    def get_fidelity_range(self):
        return self.ranges
    
    