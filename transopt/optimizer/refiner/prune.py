
from transopt.optimizer.refiner.refiner_base import RefinerBase
from transopt.agent.registry import space_refiner_registry

@space_refiner_registry.register("Prune")
class Prune(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)
            
    def refine(self, search_space, metadata=None):
        
        raise NotImplementedError("Sample method should be implemented by subclasses.")
    
    def check_metadata_avaliable(self, metadata):
        if metadata is None:
            return False
        return True 