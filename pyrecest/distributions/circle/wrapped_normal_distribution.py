import numbers

import numpy as np
from beartype import beartype
from scipy.special import erf  # pylint: disable=no-name-in-module

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from .abstract_circular_distribution import AbstractCircularDistribution
from .von_mises_distribution import VonMisesDistribution


class WrappedNormalDistribution(
    AbstractCircularDistribution, HypertoroidalWrappedNormalDistribution
):
    """
    This class implements the wrapped normal distribution.
    """

    MAX_SIGMA_BEFORE_UNIFORM = 10

    @beartype
    def __init__(self, mu: np.number | numbers.Real, sigma: np.number | numbers.Real):
        """
        Initialize a wrapped normal distribution with mean mu and standard deviation sigma.
        """
        AbstractCircularDistribution.__init__(self)
        HypertoroidalWrappedNormalDistribution.__init__(
            self, mu, np.atleast_2d(sigma**2)
        )

    @property
    def sigma(self):
        return np.sqrt(self.C)

    @beartype
    def pdf(self, xs: np.ndarray | np.number | numbers.Real):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be >0, but received {self.sigma}.")

        xs = np.asarray(xs)
        if xs.ndim == 0:
            xs = np.array([xs])
        n_inputs = xs.size
        result = np.zeros(n_inputs)

        # check if sigma is large and return uniform distribution in this case
        if self.sigma > self.MAX_SIGMA_BEFORE_UNIFORM:
            result[:] = 1.0 / (2 * np.pi)
            return result

        x = np.mod(xs, 2 * np.pi)
        x[x < 0] += 2 * np.pi
        x -= self.mu

        max_iterations = 1000

        tmp = -1.0 / (2 * self.sigma**2)
        nc = 1 / np.sqrt(2 * np.pi) / self.sigma

        for i in range(n_inputs):
            old_result = 0
            result[i] = np.exp(x[i] * x[i] * tmp)

            for k in range(1, max_iterations + 1):
                xp = x[i] + 2 * np.pi * k
                xm = x[i] - 2 * np.pi * k
                tp = xp * xp * tmp
                tm = xm * xm * tmp
                old_result = result[i]
                result[i] += np.exp(tp) + np.exp(tm)

                if result[i] == old_result:
                    break

            result[i] *= nc

        return result.squeeze()

    @beartype
    def cdf(
        self,
        xs: np.ndarray,
        startingPoint: float = 0,
        n_wraps: int | np.int32 | np.int64 = 10,
    ) -> np.ndarray:
        startingPoint = np.mod(startingPoint, 2 * np.pi)
        xs = np.mod(xs, 2 * np.pi)

        def ncdf(from_, to):
            return (
                1
                / 2
                * (
                    erf((self.mu - from_) / (np.sqrt(2) * self.sigma))
                    - erf((self.mu - to) / (np.sqrt(2) * self.sigma))
                )
            )

        val = ncdf(startingPoint, xs)
        for i in range(1, n_wraps + 1):
            val = (
                val
                + ncdf(startingPoint + 2 * np.pi * i, xs + 2 * np.pi * i)
                + ncdf(startingPoint - 2 * np.pi * i, xs - 2 * np.pi * i)
            )
        # Val should be negative when x < startingPoint
        val = np.where(xs < startingPoint, 1 + val, val)
        return np.squeeze(val)

    @beartype
    def trigonometric_moment(
        self, n: int | np.int32 | np.int64
    ) -> complex | np.ndarray:
        return np.exp(1j * n * self.mu - n**2 * self.sigma**2 / 2)

    def multiply(self, wn2: "WrappedNormalDistribution") -> "WrappedNormalDistribution":
        return self.multiply_vm(wn2)

    def multiply_vm(self, wn2):
        vm1 = self.to_vm()
        vm2 = wn2.to_vm()
        vm = vm1.multiply(vm2)
        wn = vm.to_wn()
        return wn

    def convolve(self, wn2: "WrappedNormalDistribution") -> "WrappedNormalDistribution":
        mu_ = np.mod(self.mu + wn2.mu, 2 * np.pi)
        sigma_ = np.sqrt(self.sigma**2 + wn2.sigma**2)
        wn = WrappedNormalDistribution(mu_, sigma_)
        return wn

    @beartype
    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        return np.mod(self.mu + self.sigma * np.random.randn(1, n), 2 * np.pi)

    @beartype
    def shift(self, shift_angles: float):
        assert np.isscalar(shift_angles)
        return WrappedNormalDistribution(self.mu + shift_angles, self.sigma)

    def to_vm(self) -> VonMisesDistribution:
        # Convert to Von Mises distribution
        kappa = self.sigma_to_kappa(self.sigma)
        return VonMisesDistribution(self.mu, kappa)

    @staticmethod
    def from_moment(m: complex) -> "WrappedNormalDistribution":
        mu_ = np.mod(np.angle(m), 2 * np.pi)
        sigma_ = np.sqrt(-2 * np.log(np.abs(m)))
        return WrappedNormalDistribution(mu_, sigma_)

    @staticmethod
    def sigma_to_kappa(sigma):
        # Approximate conversion from sigma to kappa for a Von Mises distribution
        return 1 / sigma**2
