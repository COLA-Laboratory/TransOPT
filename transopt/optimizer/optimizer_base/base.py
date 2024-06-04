import abc
from typing import List, Dict, Union

class OptimizerBase(abc.ABC, metaclass=abc.ABCMeta):
    """Abstract base class for the optimizers in the benchmark. This creates a common API across all packages.
    """

    # Every implementation package needs to specify this static variable, e.g., "primary_import=opentuner"
    primary_import = None

    def __init__(self, config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        self.config = config
        # self.verbose = config['verbose']
        # self.optimizer_name = config['optimizer_name']
        # self.exp_path = config['save_path']



    @abc.abstractmethod
    def suggest(self, n_suggestions:Union[None, int] = None)->List[Dict]:
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        pass

    @abc.abstractmethod
    def observe(self, input_vectors: Union[List[Dict], Dict], output_value: Union[List[Dict], Dict]) -> None:
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        pass
