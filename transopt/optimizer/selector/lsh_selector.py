from transopt.optimizer.selector.selector_base import SelectorBase 
from transopt.agent.registry import selector_registry

@selector_registry.register('LSH')
class LSHSelector(SelectorBase):
    def __init__(self, config):
        
        super(LSHSelector, self).__init__(config)
        
    def fetch_data(self, tasks_info):
        task_name = tasks_info['additional_config']['problem_name']
        variable_names = [var['name'] for var in tasks_info["variables"]] 
        num_variables = len(variable_names)
        num_objectives = len(tasks_info["objectives"])
        name_str = " ".join(variable_names)
        datasets_list = self.data_manager.search_similar_datasets(task_name, {'variable_names':name_str, 'num_variables':num_variables, 'num_objectives':num_objectives})
        metadata = {}
        metadata_info = {}
        for dataset_name in datasets_list:
                metadata[dataset_name] = self.data_manager.db.select_data(dataset_name)
                metadata_info[dataset_name] = self.data_manager.db.query_dataset_info(dataset_name)
        return metadata, metadata_info
