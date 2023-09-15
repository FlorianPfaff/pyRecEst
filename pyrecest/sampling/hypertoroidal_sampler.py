from beartype import beartype
from .abstract_sampler import AbstractSampler
from pyrecest.distributions import CircularUniformDistribution
import numpy as np


class CircularUniformSampler(AbstractSampler):
    @beartype
    def sample_stochastic(self, n_samples: int):
        return CircularUniformDistribution().sample(n_samples)
    
    @beartype
    def get_grid(self, grid_density_parameter: int) -> np.ndarray:
        """
        Returns an equidistant grid of points on the circle [0,2*pi).
        """
        points = np.linspace(0, 2 * np.pi, grid_density_parameter, endpoint=False)
        # Set it to the middle of the interval instead of the start
        points += (2 * np.pi / grid_density_parameter) / 2
        return points