import math
import numpy as np
from pymoo.core.problem import Problem
from GPyOpt import Design_space
from pymoo.algorithms.soo.nonconvex.ga import GA
from transopt.agent.registry import acf_registry
from transopt.optimizer.acquisition_function.acf_base import AcquisitionBase


@acf_registry.register('GA-PreSelect')
class GAPreSelect(AcquisitionBase):
    analytical_gradient_prediction = False

    def __init__(self, config):
        super(GAPreSelect, self).__init__()
        config_dict = {}
        if config != "":
            if ',' in config:
                key_value_pairs = config.split(',')
            else:
                key_value_pairs = [config]
            for pair in key_value_pairs:
                key, value = pair.split(':')
                config_dict[key.strip()] = value.strip()
        if 'k' in config_dict:
            self.k = int(config_dict['k'])
        else:
            self.k = 2
        if 'n' in config_dict:
            self.pop_size = 4 + math.floor(3 * math.log(int(config_dict['n'])))
        else:
            self.pop_size = 10
        self.model = None
        self.ea = None
        self.problem = None

    def link_space(self, space):
        opt_space = []
        for var_name in space.variables_order:
            var_dic = {
                'name': var_name,
                'type': 'continuous',
                'domain': space[var_name].search_space_range,
            }
            if space[var_name].type == 'categorical' or 'integer':
                var_dic['type'] = 'discrete'

            opt_space.append(var_dic.copy())
            
        self.space = Design_space(opt_space)

        if self.ea is None:
            self.problem = EAProblem(self.space.config_space, self.model.predict)
            self.ea = GA(self.pop_size)
            self.ea.setup(self.problem, verbose=False)
        else:
            self.problem = EAProblem(self.space.config_space, self.model.predict)

    def optimize(self, duplicate_manager=None):
        pop = self.ea.ask()
        self.ea.evaluator.eval(self.problem, pop)
        pop_X = np.array([p.X for p in pop])
        pop_F = np.array([p.F for p in pop])
        total_pop_X = pop_X
        total_pop_F = pop_F
        for i in range(self.k - 1):
            pop = self.ea.ask()
            self.ea.evaluator.eval(self.problem, pop)
            pop_X = np.array([p.X for p in pop])
            pop_F = np.array([p.F for p in pop])
            total_pop_X = np.concatenate((total_pop_X, pop_X))
            total_pop_F = np.concatenate((total_pop_F, pop_F))
        top_k_idx = sorted(range(len(total_pop_F)), key=lambda i: total_pop_F[i])[:self.pop_size]
        elites = total_pop_X[top_k_idx]
        elites_F = total_pop_F[top_k_idx]
        return elites, elites_F

    def _compute_acq(self, x):
        raise NotImplementedError()

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError()


class EAProblem(Problem):
    def __init__(self, space, predict):
        input_dim = len(space)
        xl = []
        xu = []
        for var_info in space:
            var_domain = var_info['domain']
            xl.append(var_domain[0])
            xu.append(var_domain[1])
        xl = np.array(xl)
        xu = np.array(xu)
        self.predict = predict
        super().__init__(n_var=input_dim, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"], _ = self.predict(x)