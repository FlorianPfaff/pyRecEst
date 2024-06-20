import copy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ndim, reshape, zeros

from ..abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .abstract_linear_distribution import AbstractLinearDistribution
from beartype import beartype

class CustomLinearDistribution(
    AbstractLinearDistribution, AbstractCustomNonPeriodicDistribution
):
    """
    Linear distribution with custom pdf.
    """

    def __init__(self, f, dim, scale_by=1, shift_by=None):
        """
        Constructor, it is the user's responsibility to ensure that f is a valid
        linear density.

        Parameters:
        f_ (function handle)
            pdf of the distribution
        dim_ (scalar)
            dimension of the Euclidean space
        """
        AbstractLinearDistribution.__init__(self, dim=dim)
        AbstractCustomNonPeriodicDistribution.__init__(self, f, scale_by=scale_by)
        if shift_by is not None:
            self.shift_by = shift_by
        else:
            self.shift_by = zeros(dim)
            
    @property
    def dim(self) -> int:
        """Get dimension of the manifold."""
        return self._dim

    @dim.setter
    @beartype
    def dim(self, value: int):
        """Set dimension of the manifold. Must be a positive integer or None."""
        if value <= 0:
            raise ValueError("dim must be a positive integer or None.")

        self._dim = value

    def shift(self, shift_by):
        assert self.dim == 1 or self.dim == shift_by.shape[0] and shift_by.ndim == 1
        cd = copy.deepcopy(self)
        cd.shift_by = self.shift_by + shift_by
        return cd

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean
        self.shift_by += mean_offset
        if self._mean_numerical is not None:
            self._mean_numerical += mean_offset

    def pdf(self, xs):
        assert self.dim == 1 and xs.ndim <= 1 or xs.shape[-1] == self.dim
        p = self.scale_by * self.f(
            # To ensure 2-d for broadcasting
            reshape(xs, (-1, self.dim))
            - reshape(self.shift_by, (1, -1))
        )
        assert ndim(p) <= 1
        return p

    @staticmethod
    def from_distribution(distribution):
        """
        Creates a CustomLinearDistribution from some other distribution

        Parameters:
        dist (AbstractLinearDistribution)
            distribution to convert

        Returns:
        chd (CustomLinearDistribution)
            CustomLinearDistribution with identical pdf
        """
        chd = CustomLinearDistribution(distribution.pdf, distribution.dim)
        return chd

    def integrate(self, left=None, right=None):
        return AbstractLinearDistribution.integrate(self, left, right)
