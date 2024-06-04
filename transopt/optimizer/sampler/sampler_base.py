
class Sampler:
    def __init__(self, n_samples, config) -> None:
        self.config = config
        self.n_samples = n_samples
        
    def sample(self, search_space, metadata=None):
        raise NotImplementedError("Sample method should be implemented by subclasses.")
    
    def change_n_samples(self, n_samples):
        self.n_samples = n_samples
    
    def check_metadata_avaliable(self, metadata):
        if metadata is None:
            return False
        return True