from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import mod, zeros

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from beartype import beartype


class CustomHypertoroidalDistribution(
    AbstractHypertoroidalDistribution, AbstractCustomDistribution
):
    def __init__(self, f, dim, scale_by=1, shift_by=None):
        # Constructor, it is the user's responsibility to ensure that f is a valid
        # hypertoroidal density and takes arguments of the same form as
        # .pdf, i.e., it needs to be vectorized.
        #
        # Parameters:
        #   f (function handle)
        #       pdf of the distribution
        #   dim (scalar)
        #       dimension of the hypertorus
        AbstractCustomDistribution.__init__(self, f, scale_by=scale_by)
        AbstractHypertoroidalDistribution.__init__(self, dim)
        if shift_by is None:
            self.shift_by = zeros(dim)
        else:
            self.shift_by = shift_by
            
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

    def pdf(self, xs):
        return AbstractCustomDistribution.pdf(self, mod(xs + self.shift_by, 2 * pi))

    def to_custom_circular(self):
        # Convert to a custom circular distribution (only in 1D case)
        #
        # Returns:
        #   ccd (CustomCircularDistribution)
        #       CustomCircularDistribution with same parameters
        from ..circle.custom_circular_distribution import CustomCircularDistribution  # to avoid circular import
        assert self.dim == 1
        ccd = CustomCircularDistribution(self.f)
        return ccd

    def to_custom_toroidal(self):
        # Convert to a custom toroidal distribution (only in 2D case)
        #
        # Returns:
        #   ctd (CustomToroidalDistribution)
        #       CustomToroidalDistribution with same parameters
        from .custom_toroidal_distribution import CustomToroidalDistribution

        assert self.dim == 2
        ctd = CustomToroidalDistribution(self.f)
        return ctd
