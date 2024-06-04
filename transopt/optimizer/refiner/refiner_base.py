


class RefinerBase:
    def __init__(self, config) -> None:
        self.config = config
        
    def refine(self, search_space, metadata=None):
        
        raise NotImplementedError("Sample method should be implemented by subclasses.")
    
    def check_metadata_avaliable(self, metadata):
        if metadata is None:
            return False
        return True