from math import pi
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import cos, empty, int32, int64, linalg, random, sin, sqrt, stack

from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalUniformDistribution(
    AbstractHypersphericalDistribution, AbstractHypersphereSubsetUniformDistribution
):
    def __init__(self, dim: Union[int, int32, int64]):
        AbstractHypersphericalDistribution.__init__(self, dim)
        AbstractHypersphereSubsetUniformDistribution.__init__(self, dim)

    def pdf(self, xs):
        return AbstractHypersphereSubsetUniformDistribution.pdf(self, xs)

    def sample(self, n: Union[int, int32, int64]):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"

        if self.dim == 2:
            s = empty(
                (
                    n,
                    self.dim + 1,
                )
            )
            phi = 2.0 * pi * random.uniform(size=n)
            sz = random.uniform(size=n) * 2.0 - 1.0
            r = sqrt(1 - sz**2)
            s = stack([r * cos(phi), r * sin(phi), sz], axis=1)
        else:
            samples_unnorm = random.normal(size=(n, self.dim + 1))
            s = samples_unnorm / linalg.norm(samples_unnorm, axis=1).reshape(-1, 1)
        return s

    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
