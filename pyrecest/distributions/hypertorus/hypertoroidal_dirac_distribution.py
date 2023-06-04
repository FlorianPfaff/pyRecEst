import copy
from collections.abc import Callable

import numpy as np
from beartype import beartype

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalDiracDistribution(
    AbstractDiracDistribution, AbstractHypertoroidalDistribution
):
    @beartype
    def __init__(
        self, d: np.ndarray, w: np.ndarray | None = None, dim: int | None = None
    ):
        """Can set dim manually to tell apart number of samples vs dimension for 1-D arrays."""
        if dim is None:
            if d.ndim > 1:
                dim = d.shape[-1]
            elif w is not None:
                dim = np.size(d) // np.size(w)
            else:
                raise ValueError("Cannot determine dimension.")

        AbstractHypertoroidalDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, np.mod(d, 2 * np.pi), w=w)

    def plot(self, *args, **kwargs):
        raise NotImplementedError("Plotting is not implemented")

    def mean_direction(self) -> np.ndarray:
        """
        Calculate the mean direction of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :return: Mean direction
        """
        a = self.trigonometric_moment(1)
        m = np.mod(np.arctan2(np.imag(a), np.real(a)), 2 * np.pi)
        return m

    @beartype
    def trigonometric_moment(self, n: int | np.int32 | np.int64) -> np.ndarray:
        """
        Calculate the trigonometric moment of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        return np.sum(
            np.exp(1j * n * self.d.T) * np.tile(self.w, (self.dim, 1)), axis=1
        )

    @beartype
    def apply_function(self, f: Callable) -> "HypertoroidalDiracDistribution":
        dist = super().apply_function(f)
        dist.d = np.mod(dist.d, 2 * np.pi)
        return dist

    def to_toroidal_wd(self):
        from .toroidal_dirac_distribution import ToroidalDiracDistribution

        assert self.dim == 2, "The dimension must be 2"
        twd = ToroidalDiracDistribution(self.d, self.w)
        return twd

    @beartype
    def marginalize_to_1D(self, dimension: int | np.int32 | np.int64):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d[:, dimension], self.w)

    @beartype
    def marginalize_out(self, dimensions: int | list[int]):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        remaining_dims = list(range(self.dim))
        remaining_dims = [dim for dim in remaining_dims if dim != dimensions]
        return CircularDiracDistribution(self.d[:, remaining_dims], self.w)

    @beartype
    def shift(self, shift_by) -> "HypertoroidalDiracDistribution":
        assert shift_by.shape[-1] == self.dim
        hd = copy.copy(self)
        hd.d = np.mod(self.d + np.reshape(shift_by, (1, -1)), 2 * np.pi)
        return hd

    def entropy(self):
        # Implement the entropy calculation here.
        raise NotImplementedError("Entropy calculation is not implemented")

    def to_wd(self):
        assert self.dim == 1
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d, self.w)
