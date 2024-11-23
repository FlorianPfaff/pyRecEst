import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import cov, ones, reshape
# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_linear_distribution import AbstractLinearDistribution


class LinearDiracDistribution(AbstractDiracDistribution, AbstractLinearDistribution):
    def __init__(self, d, w=None):
        dim = d.shape[1] if d.ndim > 1 else 1
        AbstractLinearDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, d, w)

    def mean(self):
        # Like np.average(self.d, weights=self.w, axis=0) but for all backends
        return self.w @ self.d

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean
        self.d += reshape(mean_offset, (1, -1))

    def covariance(self):
        _, C = LinearDiracDistribution.weighted_samples_to_mean_and_cov(self.d, self.w)
        return C

    def plot(self, *args, **kwargs):
        if pyrecest.backend.__name__ == "pyrecest.numpy":
            sample_locs = self.d
            sample_weights = self.w
        elif pyrecest.backend.__name__ == "pyrecest.pytorch":
            sample_locs = self.d.numpy()
            sample_weights = self.w.numpy()
        else:
            raise ValueError("Plotting not supported for this backend")

        if self.dim == 1:
            plt.stem(sample_locs.squeeze(), sample_weights, *args, **kwargs)
        elif self.dim == 2:
            plt.scatter(
                sample_locs[:, 0], sample_locs[:, 1], sample_weights / max(sample_weights) * 100, *args, **kwargs
            )
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # You can adjust 's' for marker size as needed
            ax.scatter(
                sample_locs[:, 0], sample_locs[:, 1], sample_locs[:, 2],
                s=(sample_weights / max(sample_weights) * 100),
                *args, **kwargs
            )
        else:
            raise ValueError("Plotting not supported for this dimension")

        plt.show()

    @staticmethod
    def from_distribution(distribution, n_particles):
        samples = distribution.sample(n_particles)
        return LinearDiracDistribution(samples, ones(n_particles) / n_particles)

    @staticmethod
    def weighted_samples_to_mean_and_cov(samples, weights=None):
        if weights is None:
            weights = ones(samples.shape[1]) / samples.shape[1]

        mean = weights @ samples
        deviation = samples - mean
        covariance = cov(deviation.T, aweights=weights, bias=True)

        return mean, covariance
