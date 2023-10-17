from typing import Union
from pyrecest.backend import reshape
from pyrecest.backend import mod
from pyrecest.backend import meshgrid
from pyrecest.backend import exp
from pyrecest.backend import array
from pyrecest.backend import arange
from pyrecest.backend import allclose
from pyrecest.backend import all
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros
import copy

import numpy as np
from scipy.stats import multivariate_normal

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalWrappedNormalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self, mu: np.ndarray, C: np.ndarray):
        """
        Initialize HypertoroidalWrappedNormalDistribution.

        :param mu: Mean vector.
        :param C: Covariance matrix.
        :raises AssertionError: If C_ is not square, not symmetric, not positive definite, or its dimension does not match with mu_.
        """
        AbstractHypertoroidalDistribution.__init__(self, np.size(mu))
        # First check is for 1-D case
        assert np.size(C) == 1 or C.shape[0] == C.shape[1], "C must be dim x dim"
        assert np.size(C) == 1 or allclose(C, C.T, atol=1e-8), "C must be symmetric"
        assert (
            np.size(C) == 1 and C > 0 or all(np.linalg.eigvals(C) > 0)
        ), "C must be positive definite"
        assert (
            np.size(C) == np.size(mu) or np.size(mu) == C.shape[1]
        ), "mu must be dim x 1"

        self.mu = mod(mu, 2 * np.pi)
        self.C = C

    def pdf(self, xs: np.ndarray, m: Union[int, int32, int64] = 3) -> np.ndarray:
        """
        Compute the PDF at given points.

        :param xs: Points to evaluate the PDF at.
        :param m: Controls the number of terms in the Fourier series approximation.
        :return: PDF values at xs.
        """
        xs = reshape(xs, (-1, self.dim))

        # Generate all combinations of offsets for each dimension
        offsets = [arange(-m, m + 1) * 2 * np.pi for _ in range(self.dim)]
        offset_combinations = array(meshgrid(*offsets)).T.reshape(-1, self.dim)

        # Calculate the PDF values by considering all combinations of offsets
        pdf_values = zeros(xs.shape[0])
        for offset in offset_combinations:
            shifted_xa = xs + offset[np.newaxis, :]
            pdf_values += multivariate_normal.pdf(
                shifted_xa, mean=self.mu.flatten(), cov=self.C
            )

        return pdf_values

    def shift(self, shift_by) -> "HypertoroidalWrappedNormalDistribution":
        """
        Shift distribution by the given angles

        :param shift_by: Angles to shift by.
        :raises AssertionError: If shape of shift_by does not match the dimension of the distribution.
        :return: Shifted distribution.
        """
        assert shift_by.shape == (self.dim,)

        hd = self
        hd.mu = mod(self.mu + shift_by, 2 * np.pi)
        return hd

    def sample(self, n):
        if n <= 0 or not (
            isinstance(n, int)
            or (np.isscalar(n) and np.issubdtype(type(n), np.integer))
        ):
            raise ValueError("n must be a positive integer")

        s = np.random.multivariate_normal(self.mu, self.C, n)
        s = mod(s, 2 * np.pi)  # wrap the samples
        return s

    def convolve(self, other: "HypertoroidalWrappedNormalDistribution"):
        assert self.dim == other.dim, "Dimensions of the two distributions must match"
        mu_ = (self.mu + other.mu) % (2 * np.pi)
        C_ = self.C + other.C
        dist_result = self.__class__(mu_, C_)
        return dist_result

    def set_mode(self, m):
        """
        Set the mode of the distribution.

        Parameters:
        m (numpy array): The new mode.

        Returns:
        HypertoroidalWNDistribution: A new instance of the distribution with the updated mode.
        """
        dist = copy.deepcopy(self)
        dist.mu = m
        return dist

    def trigonometric_moment(self, n):
        """
        Calculate the trigonometric moment of the HypertoroidalWNDistribution.

        :param self: HypertoroidalWNDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        assert isinstance(n, int), "n must be an integer"

        m = exp(
            [1j * n * self.mu[i] - n**2 * self.C[i, i] / 2 for i in range(self.dim)]
        )

        return m

    def mode(self):
        # Determines the mode of the distribution, i.e., the point
        # where the pdf is largest.
        #
        # Returns:
        #   m (vector)
        #       the mode
        return self.mu