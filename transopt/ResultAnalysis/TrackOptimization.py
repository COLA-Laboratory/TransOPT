import numpy as np
from collections import Counter, defaultdict
from transopt.ResultAnalysis.AnalysisBase import AnalysisBase


track_registry = {}

# 注册函数的装饰器
def track_register(name):
    def decorator(func_or_class):
        if name in track_registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        track_registry[name] = func_or_class
        return func_or_class
    return decorator




