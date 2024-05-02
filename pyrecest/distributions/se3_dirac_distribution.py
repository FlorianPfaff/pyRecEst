# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ones

from .abstract_se3_distribution import AbstractSE3Distribution
from .cart_prod.lin_hypersphere_cart_prod_dirac_distribution import (
    LinHypersphereCartProdDiracDistribution,
)
from .hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)
from geomstats.learning.frechet_mean import FrechetMean


class SE3DiracDistribution(
    LinHypersphereCartProdDiracDistribution, AbstractSE3Distribution
):
    def __init__(self, d, w=None):
        AbstractSE3Distribution.__init__(self)
        LinHypersphereCartProdDiracDistribution.__init__(self, bound_dim=3, d=d, w=w)

    def marginalize_linear(self):
        dist = HyperhemisphericalDiracDistribution(self.d[:, :4], self.w)
        return dist

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return FrechetMean(space=self.manifold).fit(self.d, weights=self.w)

    @staticmethod
    def from_distribution(distribution, n_particles):
        assert isinstance(
            distribution, AbstractSE3Distribution
        ), "dist must be an instance of AbstractSE3Distribution"
        assert (
            isinstance(n_particles, int) and n_particles > 0
        ), "n_particles must be a positive integer"

        ddist = SE3DiracDistribution(
            distribution.sample(n_particles),
            1 / n_particles * ones(n_particles),
        )
        return ddist
