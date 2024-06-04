import numpy as np
from scipy.stats import qmc

from transopt.optimizer.sampler.sampler_base import Sampler
from transopt.agent.registry import sampler_registry
from transopt.space.search_space import SearchSpace


@sampler_registry.register("lhs")
class LatinHypercubeSampler(Sampler):
    def sample(self, search_space:SearchSpace, metadata = None):
        d = len(search_space.variables_order)
        sampler = qmc.LatinHypercube(d=d)
        sample_points = sampler.random(n=self.n_samples)
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]: 
                continuous_vals = qmc.scale(
                    sample_points[:, i][np.newaxis], var_range[0], var_range[1]
                )
                sample_points[:, i] = np.round(continuous_vals).astype(int)
            else:  # 连续变量处理
                sample_points[:, i] = qmc.scale(sample_points[:, i][np.newaxis], var_range[0], var_range[1])
        return sample_points
