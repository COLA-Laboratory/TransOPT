from transopt.agent.registry import selector_registry
from transopt.optimizer.selector.selector_base import SelectorBase


@selector_registry.register("Fuzzy")
class FuzzySelector(SelectorBase):
    def __init__(self, config):
        super(FuzzySelector, self).__init__(config)

    def fetch_data(self, tasks_info):
        task_name = tasks_info["additional_config"]["problem_name"]        
        variable_names = [var['name'] for var in tasks_info["variables"]] 
        dimensions = len(variable_names)
        objectives = len(tasks_info["objectives"])
        
        conditions = {
            "task_name": task_name,
            "dimensions": dimensions,
            "objectives": objectives,
        }

        datasets_list = self.data_manager.db.search_tables_by_metadata(conditions)
        metadata = {
            dataset_name: self.data_manager.db.select_data(dataset_name)
            for dataset_name in datasets_list
        }
        metadata_info = {
            dataset_name: self.data_manager.db.query_dataset_info(dataset_name)
            for dataset_name in datasets_list
        }

        return metadata, metadata_info
