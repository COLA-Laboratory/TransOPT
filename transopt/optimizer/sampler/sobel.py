import numpy as np
from scipy.stats import qmc

from transopt.optimizer.sampler.sampler_base import Sampler
from transopt.agent.registry import sampler_registry

@sampler_registry.register("sobol")
class SobolSampler(Sampler):
    def sample(self, search_space, metadata = None):
        d = len(search_space.variables_order)
        sampler = qmc.Sobol(d=d, scramble=True)
        sample_points = sampler.random(n=self.n_samples)
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:
                # 对离散变量进行处理
                continuous_vals = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
                sample_points[:, i] = np.round(continuous_vals).astype(int)
            else:
                sample_points[:, i] = qmc.scale(
                    sample_points[:, i], var_range[0], var_range[1]
                )
        return sample_points
