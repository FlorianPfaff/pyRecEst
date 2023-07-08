from collections.abc import Callable

import numpy as np
from beartype import beartype

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CustomCircularDistribution(
    AbstractCustomDistribution, AbstractCircularDistribution
):
    @beartype
    def __init__(self, f_: Callable, scale_by: float = 1, shift_by: float = 0):
        """
        Initializes a new instance of the CustomCircularDistribution class.

        Args:
            f_ (callable): The function to be used for the distribution.
            scale_by (float, optional): The scale factor for the distribution. Defaults to 1.
            shift_by (float, optional): The shift for the distribution. Defaults to 0.

        Note:
            It is the user's responsibility to ensure that f_ is a valid circular density,
            i.e., 2pi-periodic, nonnegative and normalized.
        """
        AbstractCircularDistribution.__init__(self)
        AbstractCustomDistribution.__init__(self, f_, scale_by)
        self.shift_by = shift_by

    @beartype
    def pdf(self, xs: np.ndarray):
        """
        Computes the probability density function at xs.

        Args:
            xs (np.ndarray): The values at which to evaluate the pdf.

        Returns:
            np.ndarray: The value of the pdf at xs.
        """
        return AbstractCustomDistribution.pdf(
            self, np.mod(xs + self.shift_by, 2 * np.pi)
        )

    @beartype
    def integrate(self, integration_boundaries: np.ndarray | None = None) -> float:
        """
        Computes the integral of the pdf over the given boundaries.

        Args:
            integration_boundaries (np.ndarray, optional): The boundaries of the integral.
                Defaults to [0, 2 * np.pi].

        Returns:
            float: The value of the integral.
        """
        if integration_boundaries is None:
            integration_boundaries = np.array([0, 2 * np.pi])
        return AbstractCircularDistribution.integrate(self, integration_boundaries)
