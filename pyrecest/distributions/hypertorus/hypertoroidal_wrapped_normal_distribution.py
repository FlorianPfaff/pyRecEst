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
        assert C.shape[0] == C.shape[1], "C must be dim x dim"
        assert np.allclose(C, C.T, atol=1e-8), "C must be symmetric"
        assert np.all(np.linalg.eigvals(C) > 0), "C must be positive definite"
        assert np.size(mu) == C.shape[1], "mu must be dim x 1"

        self.mu = np.mod(mu, 2 * np.pi)
        self.C = C

    def pdf(self, xs: np.ndarray, m: int | np.int32 | np.int64 = 3) -> np.ndarray:
        """
        Compute the PDF at given points.

        :param xs: Points to evaluate the PDF at.
        :param m: Controls the number of terms in the Fourier series approximation.
        :return: PDF values at xs.
        """
        xs = np.reshape(xs, (-1, self.dim))

        # Generate all combinations of offsets for each dimension
        offsets = [np.arange(-m, m + 1) * 2 * np.pi for _ in range(self.dim)]
        offset_combinations = np.array(np.meshgrid(*offsets)).T.reshape(-1, self.dim)

        # Calculate the PDF values by considering all combinations of offsets
        pdf_values = np.zeros(xs.shape[0])
        for offset in offset_combinations:
            shifted_xa = xs + offset[np.newaxis, :]
            pdf_values += multivariate_normal.pdf(
                shifted_xa, mean=self.mu.flatten(), cov=self.C
            )

        return pdf_values

    def shift(
        self, shift_angles: np.ndarray
    ) -> "HypertoroidalWrappedNormalDistribution":
        """
        Shift distribution by the given angles

        :param shift_angles: Angles to shift by.
        :raises AssertionError: If shape of shift_angles does not match the dimension of the distribution.
        :return: Shifted distribution.
        """
        assert shift_angles.shape == (self.dim,)

        hd = self
        hd.mu = np.mod(self.mu + shift_angles, 2 * np.pi)
        return hd

    def sample(self, n):
        if n <= 0 or not (
            isinstance(n, int)
            or (np.isscalar(n) and np.issubdtype(type(n), np.integer))
        ):
            raise ValueError("n must be a positive integer")

        s = np.random.multivariate_normal(self.mu, self.C, n)
        s = np.mod(s, 2 * np.pi)  # wrap the samples
        return s

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

        m = np.exp(
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
