import numpy as np

from ..abstract_mixture import AbstractMixture
from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution
from .linear_dirac_distribution import LinearDiracDistribution


class GaussianMixture(AbstractMixture, AbstractLinearDistribution):
    def __init__(self, dists, w=None):
        assert all(
            isinstance(dist, GaussianDistribution) for dist in dists
        ), "dists must be a list of Gaussian distributions"
        AbstractLinearDistribution.__init__(self, dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    def mean(self):
        gauss_array = self.dists
        return np.dot(np.array([g.mu for g in gauss_array]), self.w)

    def to_gaussian(self):
        gauss_array = self.dists
        mu, C = self.mixture_parameters_to_gaussian_parameters(
            np.array([g.mu for g in gauss_array]),
            np.stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return GaussianDistribution(mu, C)

    def covariance(self):
        gauss_array = self.dists
        _, C = self.mixture_parameters_to_gaussian_parameters(
            np.array([g.mu for g in gauss_array]),
            np.stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return C

    @staticmethod
    def mixture_parameters_to_gaussian_parameters(
        means, covariance_matrices, weights=None
    ):
        if weights is None:
            weights = np.ones(means.shape[1]) / means.shape[1]

        C_from_cov = np.sum(covariance_matrices * weights.reshape(1, 1, -1), axis=2)
        mu, C_from_means = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            means, weights
        )
        C = C_from_cov + C_from_means

        return mu, C
