import numpy as np

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalUniformDistribution(
    AbstractUniformDistribution, AbstractHypertoroidalDistribution
):
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        """
        Returns the Probability Density Function evaluated at xs

        :param xs: Values at which to evaluate the PDF
        :returns: PDF evaluated at xs
        """
        return 1 / self.get_manifold_size() * np.ones(xs.size // self.dim)

    def trigonometric_moment(self, n: int | np.int32 | np.int64) -> np.ndarray:
        """
        Returns the n-th trigonometric moment

        :param n: Moment order
        :returns: n-th trigonometric moment
        """
        if n == 0:
            return np.ones(self.dim)

        return np.zeros(self.dim)

    def entropy(self) -> float:
        """
        Returns the entropy of the distribution

        :returns: Entropy
        """
        return self.dim * np.log(2 * np.pi)

    def mean_direction(self):
        """
        Returns the mean of the circular uniform distribution.
        Since it doesn't have a unique mean, this function always raises a ValueError.

        :raises ValueError: Circular uniform distribution does not have a unique mean
        """
        raise ValueError(
            "Hypertoroidal uniform distributions do not have a unique mean"
        )

    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        """
        Returns a sample of size n from the distribution

        :param n: Sample size
        :returns: Sample of size n
        """
        return 2 * np.pi * np.random.rand(self.dim, n)

    def shift(self, shift_angles: np.ndarray) -> "HypertoroidalUniformDistribution":
        """
        Shifts the distribution by shift_angles.
        Since this is a uniform distribution, the shift does not change the distribution.

        :param shift_angles: Angles to shift by
        :returns: Shifted distribution
        """
        assert shift_angles.shape == (self.dim,)
        return self

    def integrate(
        self, integration_boundaries: tuple[np.ndarray, np.ndarray] | None = None
    ) -> float:
        """
        Returns the integral of the distribution over the specified boundaries

        :param integration_boundaries: Optional boundaries for integration.
            If None, uses the entire distribution support.
        :returns: Integral over the specified boundaries
        """
        if integration_boundaries is None:
            left = np.zeros((self.dim,))
            right = 2 * np.pi * np.ones((self.dim,))
        assert left.shape == (self.dim,)
        assert right.shape == (self.dim,)

        volume = np.prod(right - left)
        return 1 / (2 * np.pi) ** self.dim * volume
