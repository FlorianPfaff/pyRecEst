import matplotlib.pyplot as plt
import numpy as np

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_linear_distribution import AbstractLinearDistribution


class LinearDiracDistribution(AbstractDiracDistribution, AbstractLinearDistribution):
    def __init__(self, d, w=None):
        dim = d.shape[1] if d.ndim > 1 else 1
        AbstractLinearDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, d, w)

    def mean(self):
        return np.average(self.d, weights=self.w, axis=0)

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean
        self.d += np.reshape(mean_offset, (1, -1))

    def covariance(self):
        _, C = LinearDiracDistribution.weighted_samples_to_mean_and_cov(self.d, self.w)
        return C

    def plot(self, *args, **kwargs):
        if self.dim == 1:
            plt.stem(self.d, self.w, *args, **kwargs)
        elif self.dim == 2:
            plt.scatter(
                self.d[0, :], self.d[1, :], self.w / max(self.w) * 100, *args, **kwargs
            )
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.d[0, :], self.d[1, :], self.w / max(self.w) * 100, *args, **kwargs
            )
        else:
            raise ValueError("Plotting not supported for this dimension")

    @staticmethod
    def from_distribution(distribution, n_samples):
        samples = distribution.sample(n_samples)
        return LinearDiracDistribution(samples, np.ones(n_samples) / n_samples)

    @staticmethod
    def weighted_samples_to_mean_and_cov(samples, weights=None):
        if weights is None:
            weights = np.ones(samples.shape[1]) / samples.shape[1]

        mean = np.average(samples, weights=weights, axis=0)
        deviation = samples - mean
        covariance = np.cov(deviation.T, aweights=weights, bias=True)

        return mean, covariance
