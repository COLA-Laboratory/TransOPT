

from transopt.datamanager.manager import DataManager
from abc import ABC, abstractmethod

class SelectorBase:
    def __init__(self, config):
        self.data_manager = DataManager()
        


    @abstractmethod
    def fetch_data(self, tasks_info):
        raise NotImplementedError
    
    