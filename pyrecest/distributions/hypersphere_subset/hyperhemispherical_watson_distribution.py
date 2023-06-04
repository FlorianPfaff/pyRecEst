import numbers

import numpy as np
from beartype import beartype

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .watson_distribution import WatsonDistribution


class HyperhemisphericalWatsonDistribution(AbstractHyperhemisphericalDistribution):
    @beartype
    def __init__(self, mu: np.ndarray, kappa: np.number | numbers.Real):
        assert mu[-1] >= 0
        self.dist_full_sphere = WatsonDistribution(mu, kappa)
        AbstractHyperhemisphericalDistribution.__init__(
            self, dim=self.dist_full_sphere.dim
        )

    def pdf(self, xs):
        return 2 * self.dist_full_sphere.pdf(xs)

    @beartype
    def set_mode(self, mu: np.ndarray) -> "HyperhemisphericalWatsonDistribution":
        w = self
        w.mu = mu
        return w

    @beartype
    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        s_full = self.dist_full_sphere.sample(n)
        s = s_full * (-1) ** (s_full[-1] < 0)  # Mirror to upper hemisphere
        return s

    @property
    def mu(self) -> np.ndarray:
        return self.dist_full_sphere.mu

    @mu.setter
    @beartype
    def mu(self, mu: np.ndarray):
        self.dist_full_sphere.mu = mu

    @property
    def kappa(self) -> float:
        return self.dist_full_sphere.kappa

    @kappa.setter
    @beartype
    def kappa(self, kappa: float):
        self.dist_full_sphere.kappa = kappa

    def mode(self) -> np.ndarray:
        return self.mu

    @beartype
    def shift(self, shift_by) -> "HyperhemisphericalWatsonDistribution":
        assert np.allclose(
            self.mu, np.append(np.zeros(self.dim - 1), 1)
        ), "There is no true shifting for the hyperhemisphere. This is a function for compatibility and only works when mu is [0,0,...,1]."
        dist_shifted = self
        dist_shifted.mu = shift_by
        return dist_shifted
