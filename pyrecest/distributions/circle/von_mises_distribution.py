import numbers

import numpy as np
from beartype import beartype
from scipy.optimize import fsolve
from scipy.special import iv
from scipy.stats import vonmises

from .abstract_circular_distribution import AbstractCircularDistribution


class VonMisesDistribution(AbstractCircularDistribution):
    def __init__(
        self,
        mu: np.number | numbers.Real,
        kappa,
        norm_const: float | None = None,
    ):
        assert kappa >= 0.0
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self._norm_const = norm_const

    def get_params(self):
        return self.mu, self.kappa

    @property
    @beartype
    def norm_const(self) -> np.number:
        if self._norm_const is None:
            self._norm_const = 2 * np.pi * iv(0, self.kappa)
        return self._norm_const

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray | np.number:
        p = np.exp(self.kappa * np.cos(xs - self.mu)) / self.norm_const
        return p

    @staticmethod
    def besselratio(
        nu: np.number | numbers.Real, kappa: np.number | numbers.Real
    ) -> np.number | numbers.Real:
        return iv(nu + 1, kappa) / iv(nu, kappa)

    def cdf(self, xs, starting_point=0):
        """
        Evaluate cumulative distribution function

        Parameters:
        xs : (n)
            points where the cdf should be evaluated
        starting_point : scalar, optional, default: 0
            point where the cdf is zero (starting point can be
            [0, 2pi) on the circle, default is 0)

        Returns:
        val : (n)
            cdf evaluated at columns of xs
        """
        assert xs.ndim <= 1

        r = np.zeros_like(xs)

        def to_minus_pi_to_pi_range(
            angle: np.number | numbers.Real | np.ndarray,
        ) -> np.number | numbers.Real | np.ndarray:
            return np.mod(angle + np.pi, 2 * np.pi) - np.pi

        r = vonmises.cdf(
            to_minus_pi_to_pi_range(xs),
            kappa=self.kappa,
            loc=to_minus_pi_to_pi_range(self.mu),
        ) - vonmises.cdf(
            to_minus_pi_to_pi_range(starting_point),
            kappa=self.kappa,
            loc=to_minus_pi_to_pi_range(self.mu),
        )

        r = np.where(
            to_minus_pi_to_pi_range(xs) < to_minus_pi_to_pi_range(starting_point),
            1 + r,
            r,
        )
        return r

    @staticmethod
    @beartype
    def besselratio_inverse(
        v: np.number | numbers.Real, x: np.number | numbers.Real
    ) -> np.number | numbers.Real:
        def f(t: float) -> float:
            return VonMisesDistribution.besselratio(v, t) - x

        start = 1.0
        (kappa,) = fsolve(f, start)
        return kappa

    @beartype
    def multiply(self, vm2: "VonMisesDistribution") -> "VonMisesDistribution":
        C = self.kappa * np.cos(self.mu) + vm2.kappa * np.cos(vm2.mu)
        S = self.kappa * np.sin(self.mu) + vm2.kappa * np.sin(vm2.mu)
        mu_ = np.mod(np.arctan2(S, C), 2 * np.pi)
        kappa_ = np.sqrt(C**2 + S**2)
        return VonMisesDistribution(mu_, kappa_)

    @beartype
    def convolve(self, vm2: "VonMisesDistribution") -> "VonMisesDistribution":
        mu_ = np.mod(self.mu + vm2.mu, 2 * np.pi)
        t = VonMisesDistribution.besselratio(
            0, self.kappa
        ) * VonMisesDistribution.besselratio(0, vm2.kappa)
        kappa_ = VonMisesDistribution.besselratio_inverse(0, t)
        return VonMisesDistribution(mu_, kappa_)

    def entropy(self):
        result = -self.kappa * VonMisesDistribution.besselratio(0, self.kappa) + np.log(
            2 * np.pi * iv(0, self.kappa)
        )
        return result

    @staticmethod
    def from_moment(m):
        """
        Obtain a VM distribution from a given first trigonometric moment.

        Parameters:
            m (scalar): First trigonometric moment (complex number).

        Returns:
            vm (VMDistribution): VM distribution obtained by moment matching.
        """
        mu_ = np.mod(np.arctan2(np.imag(m), np.real(m)), 2 * np.pi)
        kappa_ = VonMisesDistribution.besselratio_inverse(0, np.abs(m))
        vm = VonMisesDistribution(mu_, kappa_)
        return vm

    def __str__(self) -> str:
        return f"VonMisesDistribution: mu = {self.mu}, kappa = {self.kappa}"
