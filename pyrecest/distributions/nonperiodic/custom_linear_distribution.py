import copy

import numpy as np

from ..abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .abstract_linear_distribution import AbstractLinearDistribution


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
            self.shift_by = np.zeros(dim)

    def shift(self, shift_vector):
        assert self.dim == np.size(shift_vector) and shift_vector.ndim <= 1
        cd = copy.deepcopy(self)
        cd.shift_by = self.shift_by + shift_vector
        return cd

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean
        self.shift_by *= mean_offset

    def pdf(self, xs):
        assert np.size(xs) % self.input_dim == 0
        n_inputs = np.size(xs) // self.input_dim
        p = self.scale_by * self.f(xs - np.atleast_2d(self.shift_by))
        assert p.ndim <= 1 and np.size(p) == n_inputs
        return p

    @staticmethod
    def from_distribution(dist):
        """
        Creates a CustomLinearDistribution from some other distribution

        Parameters:
        dist (AbstractLinearDistribution)
            distribution to convert

        Returns:
        chd (CustomLinearDistribution)
            CustomLinearDistribution with identical pdf
        """
        chd = CustomLinearDistribution(dist.pdf, dist.dim)
        return chd

    def integrate(self, left=None, right=None):
        return AbstractLinearDistribution.integrate(self, left, right)
