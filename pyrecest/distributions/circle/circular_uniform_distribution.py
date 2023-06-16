import numpy as np

from ..hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularUniformDistribution(
    HypertoroidalUniformDistribution, AbstractCircularDistribution
):
    """
    Circular uniform distribution
    """

    def __init__(self):
        HypertoroidalUniformDistribution.__init__(self, 1)
        AbstractCircularDistribution.__init__(self)

    def get_manifold_size(self):
        return AbstractCircularDistribution.get_manifold_size(self)

    def shift(self, _):
        return CircularUniformDistribution()

    def cdf(self, xa, starting_point=0):
        """
        Evaluate cumulative distribution function

        Parameters
        ----------
        xa : (1, n)
            points where the cdf should be evaluated
        starting_point : scalar
            point where the cdf is zero (starting point can be
            [0, 2pi) on the circle, default 0

        Returns
        -------
        val : (1, n)
            cdf evaluated at columns of xa
        """

        val = (xa - starting_point) / (2 * np.pi)
        val[val < 0] = val[val < 0] + 1

        return val
