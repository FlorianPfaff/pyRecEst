import numpy as np

from .abstract_circular_distribution import AbstractCircularDistribution


class WrappedCauchyDistribution(AbstractCircularDistribution):
    def __init__(self, mu, gamma):
        AbstractCircularDistribution.__init__(self)
        self.mu = np.mod(mu, 2 * np.pi)
        assert gamma > 0
        self.gamma = gamma

    def pdf(self, xs):
        assert xs.ndim == 1
        xs = np.mod(xs - self.mu, 2 * np.pi)
        return (
            1
            / (2 * np.pi)
            * np.sinh(self.gamma)
            / (np.cosh(self.gamma) - np.cos(xs - self.mu))
        )

    def cdf(self, xs):
        def coth(x):
            return 1 / np.tanh(x)

        assert xs.ndim == 1
        return np.arctan(coth(self.gamma / 2) * np.tan((xs - self.mu) / 2)) / np.pi

    def trigonometric_moment(self, n):
        m = np.exp(1j * n * self.mu - abs(n) * self.gamma)
        return m
