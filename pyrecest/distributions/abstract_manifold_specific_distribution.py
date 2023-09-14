import numbers
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from beartype import beartype


class AbstractManifoldSpecificDistribution(ABC):
    """
    Abstract base class for distributions catering to specific manifolds.
    Should be inerhited by (abstract) classes limited to specific manifolds.
    """

    @beartype
    def __init__(self, dim: int):
        self._dim = dim

    @abstractmethod
    def get_manifold_size(self) -> float:
        pass

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

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Convenient access to a reasonable "mean" for different manifolds.

        :return: The mean of the distribution.
        :rtype: np.ndarray
        """
        
    def set_mode(self, _):
        """
        Set the mode of the distribution
        """
        raise NotImplementedError("set_mode is not implemented for this distribution")

    @beartype
    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        """Obtain n samples from the distribution."""
        return self.sample_metropolis_hastings(n)

    # jscpd:ignore-start
    @beartype
    def sample_metropolis_hastings(
        self,
        n: int | np.int32 | np.int64,
        burn_in: int | np.int32 | np.int64 = 10,
        skipping: int | np.int32 | np.int64 = 5,
        proposal: Callable | None = None,
        start_point: np.number | numbers.Real | np.ndarray | None = None,
    ) -> np.ndarray:
        # jscpd:ignore-end
        """Metropolis Hastings sampling algorithm."""

        if proposal is None or start_point is None:
            raise NotImplementedError(
                "Default proposals and starting points should be set in inheriting classes."
            )

        total_samples = burn_in + n * skipping
        s = np.empty(
            (
                total_samples,
                self.input_dim,
            ),
        )
        s.fill(np.nan)
        x = start_point
        i = 0
        pdfx = self.pdf(x)

        while i < total_samples:
            x_new = proposal(x)
            pdfx_new = self.pdf(x_new)
            a = pdfx_new / pdfx
            if a.item() > 1 or a.item() > np.random.rand():
                s[i, :] = x_new.squeeze()
                x = x_new
                pdfx = pdfx_new
                i += 1

        relevant_samples = s[burn_in::skipping, :]
        return np.squeeze(relevant_samples)
