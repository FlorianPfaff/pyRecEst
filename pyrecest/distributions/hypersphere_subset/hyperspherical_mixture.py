import numpy as np

from ..abstract_mixture import AbstractMixture
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalMixture(AbstractMixture, AbstractHypersphericalDistribution):
    def __init__(self, dists, w):
        AbstractHypersphericalDistribution.__init__(self, dim=dists[0].dim)
        assert all(
            isinstance(dist, AbstractHypersphericalDistribution) for dist in dists
        ), "dists must be a list of hyperspherical distributions"
        # if all(isinstance(dist, AbstractSphericalHarmonicDistribution) for dist in dists):
        #    print('Warning: Creating a mixture of Spherical Harmonics may not be necessary.')

        self.dists = dists
        self.w = w
        AbstractMixture.__init__(self, dists, w)

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1, "Dimension mismatch"

        p = np.zeros(xs.shape[0])
        for dist, weight in zip(self.dists, self.w):
            p += weight * dist.pdf(xs)

        return p
