import numpy as np
from sampler.sampler_base import Sampler
from agent.registry import sampler_registry

# @sampler_registry.register("grid")
class GridSampler(Sampler):
    def generate_grid_for_variable(self, var_range, is_discrete, steps):
        if is_discrete:
            if (var_range[1] - var_range[0] + 1) <= steps:
                return np.arange(var_range[0], var_range[1] + 1)
            else:
                return np.linspace(
                    var_range[0], var_range[1], num=steps, endpoint=True
                ).round()
        else:
            return np.linspace(var_range[0], var_range[1], num=steps)

    def sample(self, search_space, steps=5, metadata=None):
        grids = []
        for name in search_space.variables_order:
            var_range = search_space.ranges[name]
            is_discrete = search_space.var_discrete[name]
            grid = self.generate_grid_for_variable(var_range, is_discrete, steps)
            grids.append(grid)

        mesh = np.meshgrid(*grids, indexing="ij")
        sample_points = np.stack(mesh, axis=-1).reshape(
            -1, len(search_space.variables_order)
        )
        return sample_points
