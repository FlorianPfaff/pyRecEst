import warnings

from ..abstract_mixture import AbstractMixture
from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution


class LinearMixture(AbstractMixture, AbstractLinearDistribution):
    def __init__(self, dists, w):
        assert all(
            isinstance(dist, AbstractLinearDistribution) for dist in dists
        ), "dists must be a list of linear distributions"
        if all(isinstance(dist, GaussianDistribution) for dist in dists):
            warnings.warn(
                "For mixtures of Gaussians, consider using GaussianMixture.",
                UserWarning,
            )
        AbstractLinearDistribution.__init__(self, dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean
        for dist in self.dists:
            dist.set_mean(dist.mean + mean_offset)

    @property
    def input_dim(self):
        return AbstractLinearDistribution.input_dim.fget(self)
