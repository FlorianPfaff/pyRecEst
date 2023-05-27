from abc import abstractmethod

from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


class AbstractPeriodicDistribution(AbstractBoundedDomainDistribution):
    """Abstract class for a distributions on periodic manifolds."""

    def __init__(self, dim):
        super().__init__(dim=dim)

    def mean(self):
        """
        Convenient access to mean_direction to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype: np.ndarray
        """
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self):
        """
        Abstract method to compute the mean direction of the distribution.

        Returns
        -------
        mean_direction: np.ndarray
            The mean direction of the distribution.
        """
