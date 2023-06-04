import collections
import copy

import numpy as np
from beartype import beartype

from ..abstract_mixture import AbstractMixture
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalMixture(AbstractMixture, AbstractHypertoroidalDistribution):
    @beartype
    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractHypertoroidalDistribution],
        w: np.ndarray | None = None,
    ):
        """
        Constructor

        :param dists: list of hypertoroidal distributions
        :param w: list of weights
        """
        AbstractHypertoroidalDistribution.__init__(self, dim=dists[0].dim)
        AbstractMixture.__init__(self, dists, w)
        # To tell mypy the type of the variable explicitly
        self.dists: list[AbstractHypertoroidalDistribution] = self.dists

    def trigonometric_moment(self, n: int | np.int32 | np.int64) -> np.ndarray:
        """
        Calculate n-th trigonometric moment

        :param n: number of moment
        :returns: n-th trigonometric moment (complex number)
        """
        m = np.zeros(self.dim, dtype=np.complex128)
        for i in range(len(self.dists)):
            # Calculate moments using moments of each component
            m += self.w[i] * self.dists[i].trigonometric_moment(n)
        return m

    def shift(self, shift_by):
        """
        Shifts the distribution by shift_angles.

        :param shift_angles: angles to shift by
        :returns: shifted distribution
        """
        assert np.size(shift_by) == self.dim
        hd_shifted = copy.deepcopy(self)
        hd_shifted.dists = [dist.shift(shift_by) for dist in hd_shifted.dists]
        return hd_shifted

    def to_circular_mixture(self):
        """
        Convert to a circular mixture (only in 1D case)

        :returns: CircularMixture with same parameters
        """
        assert self.dim == 1
        from ..circle.circular_mixture import CircularMixture

        return CircularMixture(self.dists, self.w)

    def to_toroidal_mixture(self):
        """
        Convert to a toroidal mixture (only in 2D case)

        :returns: ToroidalMixture with same parameters
        """
        assert self.dim == 2
        from toroidal_mixture import ToroidalMixture

        return ToroidalMixture(self.dists, self.w)

    @property
    def input_dim(self):
        return AbstractHypertoroidalDistribution.input_dim.fget(self)
