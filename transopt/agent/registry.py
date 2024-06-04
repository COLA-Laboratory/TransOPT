class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None, cls=None, **kwargs):
        if cls is None:
            def wrapper(cls):
                return self.register(name, cls, **kwargs)
            return wrapper
        
        if name is None:
            name = cls.__name__

        if name in self._registry:
            raise ValueError(f"Error: '{name}' is already registered.")
        
        self._registry[name] = {'cls': cls, **kwargs}
        return cls

    def get(self, name):
        return self._registry[name]['cls']

    def list_names(self):
        return list(self._registry.keys())

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return item in self._registry

space_refiner_registry = Registry()
sampler_registry = Registry()
pretrain_registry = Registry()
model_registry = Registry()
acf_registry = Registry()
problem_registry = Registry()
statistic_registry = Registry()
selector_registry = Registry()
normalizer_registry = Registry()
