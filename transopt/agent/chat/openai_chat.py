import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from transopt.agent.config import RunningConfig
from transopt.agent.registry import *
from transopt.benchmark.instantiate_problems import InstantiateProblems
from transopt.datamanager.manager import DataManager
from transopt.optimizer.construct_optimizer import ConstructOptimizer
from transopt.utils.log import logger


def dict_to_string(dictionary):
    return json.dumps(dictionary, ensure_ascii=False, indent=4)


class Message(BaseModel):
    """Model for LLM messages"""

    role: str  # The role of the message author (system, user, assistant, or function).
    content: Optional[Union[str, List[Dict]]] = None  # The message content.
    tool_call_id: Optional[str] = None  # ID for the tool call response
    name: Optional[str] = None  # Name of the tool or function, if applicable
    metrics: Dict[str, Any] = {}  # Metrics for the message.
    

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return json.dumps(self.content)
        return ""

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude_none=True, exclude={"metrics"})
        # Manually add the content field if it is None
        if self.content is None:
            _dict["content"] = None
        return _dict

    def log(self, level: Optional[str] = None):
        """Log the message to the console."""
        _logger = getattr(logger, level or "debug")
        
        _logger(f"============== {self.role} ==============")
        message_detail = f"Content: {self.get_content_string()}"
        if self.tool_call_id:
            message_detail += f", Tool Call ID: {self.tool_call_id}"
        if self.name:
            message_detail += f", Name: {self.name}"
        _logger(message_detail)


class OpenAIChat:
    history: List[Message]

    def __init__(
        self,
        api_key,
        model="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        client_kwargs: Optional[Dict[str, Any]] = None,
        data_manager: Optional[DataManager] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key 
        self.client_kwargs = client_kwargs or {}

        self.prompt = self._get_prompt()
        self.is_first_msg = True
        
        self.history = []

        self.data_manager = DataManager() if data_manager is None else data_manager
        self.running_config = RunningConfig()

    def _get_prompt(self):
        """Reads a prompt from a file."""
        current_dir = Path(__file__).parent
        file_path = current_dir / "prompt"
        with open(file_path, "r") as file:
            return file.read()
        
    @property
    def client(self):
        """Lazy initialization of the OpenAI client."""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url,
            **self.client_kwargs
        )

    def invoke_model(self, messages: List[Dict]) -> ChatCompletion:
        self.history.extend(messages)
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_all_datasets",
                    "description": "Show all available datasets in our system",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_info",
                    "description": "Show detailed information of dataset according to the dataset name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "The name of the dataset",
                            },
                        },
                        "required": ["dataset_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_problems",
                    "description": "Show all optimization problems that our system supoorts",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "get_optimization_techniques",
                    "description": "Show all optimization techniques supported in  our system,",
                    "parameters": {},
                },
            },
                        
            {
                "type": "function",
                "function": {
                    "name": "set_optimization_problem",
                    "description": "Define or set an optimization problem based on user inputs for 'problem name', 'workload' and 'budget'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "problem_name": {
                                "type": "string",
                                "description": "The name of the optimization problem",
                            },
                            "workload": {
                                "type": "integer",
                                "description": "The number of workload",
                            },
                            "budget": {
                                "type": "integer",
                                "description": "The number of budget to do function evaluations",
                            },
                        },
                        "required": ["problem_name", "workload", "budget"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_model",
                    "description": "Set the model used as surrogate model in the  Bayesian optimization, The input model name should be one of the available models.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Model": {
                                "type": "string",
                                "description": "The model name",
                            },
                        },
                        "required": ["Model"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_sampler",
                    "description": "Set the sampler for the optimization process as user input. The input sampler name should be one of the available samplers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Sampler": {
                                "type": "string",
                                "description": "The name of Sampler",
                            },
                        },
                        "required": ["Sampler"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_pretrain",
                    "description": "Set the Pretrain methods. The input of users should include one of the available pretrain methods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Pretrain": {
                                "type": "string",
                                "description": "The name of Pretrain method",
                            },
                        },
                        "required": ["Pretrain"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_normalizer",
                    "description": "Set the normalization method to nomalize function evaluation and parameters. It requires one of the available normalization methods as input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Normalizer": {
                                "type": "string",
                                "description": "The name of Normalization method",
                            },
                        },
                        "required": ["Normalizer"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_metadata",
                    "description": "Set the metadata using a dataset stored in our system and specify a module to utilize this metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Normalizer": {
                                "type": "string",
                                "description": "The name of Normalization method",
                            },
                        },
                        "required": ["module_name", "dataset_name"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "run_optimization",
                    "description": "Set the normalization method to nomalize function evaluation and parameters. It requires one of the available normalization methods as input.",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "show_configuration",
                    "description": "Display all configurations set by the user so far, including the optimizer configuration, metadata configuration, and optimization problems",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "install_package",
                    "description": "Install a Python package using pip",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "package_name": {
                                "type": "string",
                                "description": "The name of the package to install",
                            },
                        },
                        "required": ["package_name"],
                    },
                },
            },      
        ]
                
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Process tool calls if there are any
        if tool_calls:
            self.history.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = self.call_manager_function(function_name, **function_args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                }
                self.history.append(tool_message)
                
            # Refresh the model with the function response and get a new response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
            )
        
        self.history.append(response.choices[0].message) 
        logger.debug(f"Response: {response.choices[0].message.content}")
        return response

    def get_response(self, user_input) -> str:
        logger.debug("---------- OpenAI Response Start ----------")
        user_message = {"role": "user", "content": user_input}
        logger.debug(f"User: {user_input}")
        messages = [user_message]

        if self.is_first_msg:
            system_message = {"role": "system", "content": self.prompt}
            messages.insert(0, system_message)
            self.is_first_msg = False
        else:
            system_message = {"role": "system", "content": "Don't tell me which function to use, just call it. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous"}
            messages.insert(0, system_message)
            

        response = self.invoke_model(messages)
        logger.debug(f"Assistant: {response.choices[0].message.content}")
        logger.debug("---------- OpenAI Response End ----------")
        return response.choices[0].message.content 
    
    def call_manager_function(self, function_name, **kwargs):
        available_functions = {
            "get_all_datasets": self.data_manager.get_all_datasets,
            "get_all_problems": self.get_all_problems,
            "get_optimization_techniques": self.get_optimization_techniques,
            "get_dataset_info": lambda: self.data_manager.get_dataset_info(kwargs['dataset_name']),
            "set_optimization_problem": lambda: self.set_optimization_problem(kwargs['problem_name'], kwargs['workload'], kwargs['budget']),
            'set_space_refiner': lambda: self.set_space_refiner(kwargs['refiner']),
            'set_sampler': lambda: self.set_sampler(kwargs['Sampler']),
            'set_pretrain': lambda: self.set_pretrain(kwargs['Pretrain']),
            'set_model': lambda: self.set_model(kwargs['Model']),
            'set_normalizer': lambda: self.set_normalizer(kwargs['Normalizer']),
            'set_metadata': lambda: self.set_metadata(kwargs['module_name'], kwargs['dataset_name']),
            'run_optimization': self.run_optimization,
            'show_configuration': self.show_configuration,
            "install_package": lambda: self.install_package(kwargs['package_name']),
        }
        function_to_call = available_functions[function_name]
        return json.dumps({"result": function_to_call()})
    
    def _initialize_modules(self):
        import transopt.benchmark.synthetic
        # import transopt.benchmark.CPD
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.refiner
        import transopt.optimizer.sampler

    def get_all_problems(self):
        tasks_info = []

        # tasks information
        task_names = problem_registry.list_names()
        for name in task_names:
            if problem_registry[name].problem_type == "synthetic":
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": "True",
                    'num_vars': [],
                    "num_objs": [1],
                    "workloads": [],
                    "fidelity": [],
                }
            else:
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                fidelity = problem_registry[name].fidelity
                workloads = problem_registry[name].workloads
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": False,
                    "num_vars": [num_var],
                    "num_objs": [num_obj],
                    "workloads": [workloads],
                    "fidelity": [fidelity],
                }
            tasks_info.append(task_info)
        return tasks_info
    
    def get_optimization_techniques(self):
        basic_info = {}

        selector_info = []
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = []
        refiner_info = []
        normalizer_info = []
        
        # tasks information
        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append(name)
        basic_info["Sampler"] = ','.join(sampler_info)

        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append(name)
        basic_info["SpaceRefiner"] = ','.join(refiner_info)

        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append(name)
        basic_info["Pretrain"] = ','.join(pretrain_info)

        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append(name)
        basic_info["Model"] = ','.join(model_info)

        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append(name)
        basic_info["ACF"] = ','.join(acf_info)

        selector_names = selector_registry.list_names()
        for name in selector_names:
            selector_info.append(name)
        basic_info["DataSelector"] = ','.join(selector_info)
        
        normalizer_names = selector_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append(name)
        basic_info["Normalizer"] = ','.join(normalizer_info)
        
        
        return basic_info
    
    def set_optimization_problem(self, problem_name, workload, budget):        
        problem_info = {}
        if problem_name in problem_registry:
            problem_info[problem_name] = {
                'budget': budget,
                'workload': workload,
                'budget_type': 'Num_FEs',
                "params": {},
            }

        self.running_config.set_tasks(problem_info)
        return "Succeed"
    
    def set_space_refiner(self, refiner):
        self.running_config.optimizer['SpaceRefiner'] = refiner
        return f"Succeed to set the space refiner {refiner}"

    def set_sampler(self, Sampler):
        self.running_config.optimizer['Sampler'] = Sampler
        return f"Succeed to set the sampler {Sampler}"
    
    
    def set_pretrain(self, Pretrain):
        self.running_config.optimizer['Pretrain'] = Pretrain
        return f"Succeed to set the pretrain {Pretrain}"
    
    def set_model(self, Model):
        self.running_config.optimizer['Model'] = Model
        return f"Succeed to set the model {Model}"
    
    def set_normalizer(self, Normalizer):
        self.running_config.optimizer['Normalizer'] = Normalizer
        return f"Succeed to set the normalizer {Normalizer}"
    
    def set_metadata(self, module_name, dataset_name):
        self.running_config.metadata[module_name] = dataset_name
        return f"Succeed to set the metadata {dataset_name} for {module_name}"
    
    def run_optimization(self):
        task_set = InstantiateProblems(self.running_config.tasks, 0)
        optimizer = ConstructOptimizer(self.running_config.optimizer, 0)
        
        try:
            while (task_set.get_unsolved_num()):
                iteration = 0
                search_space = task_set.get_cur_searchspace()
                dataset_info, dataset_name = self.construct_dataset_info(task_set, self.running_config, seed=0)
                
                self.data_manager.db.create_table(dataset_name, dataset_info, overwrite=True)
                optimizer.link_task(task_name=task_set.get_curname(), search_sapce=search_space)
                
                metadata, metadata_info = self.get_metadata('SpaceRefiner')
                optimizer.search_space_refine(metadata, metadata_info)
                
                metadata, metadata_info = self.get_metadata('Sampler')
                samples = optimizer.sample_initial_set(metadata, metadata_info)
                
                parameters = [search_space.map_to_design_space(sample) for sample in samples]
                observations = task_set.f(parameters)
                self.save_data(dataset_name, parameters, observations, iteration)
                
                optimizer.observe(samples, observations)
                
                #Pretrain
                metadata, metadata_info = self.get_metadata('Model')
                optimizer.meta_fit(metadata, metadata_info)
        
                while (task_set.get_rest_budget()):
                    optimizer.fit()
                    suggested_samples = optimizer.suggest()
                    parameters = [search_space.map_to_design_space(sample) for sample in suggested_samples]
                    observations = task_set.f(parameters)
                    self.save_data(dataset_name, parameters, observations, iteration)
                    
                    optimizer.observe(suggested_samples, observations)
                    iteration += 1
                    
                    print("Seed: ", 0, "Task: ", task_set.get_curname(), "Iteration: ", iteration)
                    # if self.verbose:
                    #     self.visualization(testsuits, suggested_sample)
                task_set.roll()
        except Exception as e:
            raise e
    def show_configuration(self):
        conf = {'Optimization problem': self.running_config.tasks, 'Optimizer': self.running_config.optimizer, 'Metadata': self.running_config.metadata}
        return dict_to_string(conf)
    
    def install_package(self, package_name: str) -> str:
        """Install a Python package using pip."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"Package '{package_name}' installed successfully."
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install package '{package_name}': {e}")
            return f"Failed to install package '{package_name}'. Error: {str(e)}"