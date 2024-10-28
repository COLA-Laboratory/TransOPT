
class Config:
    DEBUG = True
    OPENAI_API_KEY = "sk-1XGNThXZQVYh6EI25b44Bb74940d4eEdBdDa81723e00C794"
    OPENAI_URL = "https://aihubmix.com/v1"




class RunningConfig:
    _instance = None
    _init = False  # 用于保证初始化代码只运行一次

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RunningConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    
    def __init__(self):
        self.tasks = None
        self.optimizer = {'SpaceRefiner':None, 'Sampler':None, 'ACF':None, 'Pretrain':None, 'Model':None, 'Normalizer':None}
        self.metadata = {'SpaceRefiner':[], 'Sampler':[], 'ACF':[], 'Pretrain':[], 'Model':[], 'Normalizer':[]}
        
        
    def set_tasks(self, tasks):
        self.tasks = tasks
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        if 'SamplerInitNum' not in self.optimizer:
            self.optimizer['SamplerInitNum'] = 11
        self.optimizer['SamplerInitNum'] =  int(self.optimizer['SamplerInitNum'])

    def set_metadata(self, metadata):
        self.metadata[metadata['object']] = metadata['datasets']

    