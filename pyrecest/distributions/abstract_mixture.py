import warnings

import numpy as np
from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType
from .abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class AbstractMixture(AbstractDistributionType):
    """
    Abstract base class for mixture distributions.
    """

    @beartype
    def __init__(self, dists: list[AbstractManifoldSpecificDistribution], weights=None):
        AbstractDistributionType.__init__(self)
        num_distributions = len(dists)

        if weights is None:
            weights = np.ones(num_distributions) / num_distributions
        else:
            weights = np.asarray(weights)

        if num_distributions != len(weights):
            raise ValueError("Sizes of distributions and weights must be equal")

        if not all(dists[0].dim == dist.dim for dist in dists):
            raise ValueError("All distributions must have the same dimension")

        non_zero_indices = np.nonzero(weights)[0]

        if len(non_zero_indices) < len(weights):
            warnings.warn(
                "Elements with zero weights detected. Pruning elements in mixture with weight zero."
            )
            dists = [dists[i] for i in non_zero_indices]
            weights = weights[non_zero_indices]

        self.dists = dists

        if abs(np.sum(weights) - 1) > 1e-10:
            warnings.warn("Weights of mixture do not sum to one.")
            self.w = weights / np.sum(weights)
        else:
            self.w = weights

    @property
    def input_dim(self) -> int:
        return self.dists[0].input_dim

    @beartype
    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        d = np.random.choice(len(self.w), size=n, p=self.w)

        occurrences = np.bincount(d, minlength=len(self.dists))
        count = 0
        s = np.empty((n, self.input_dim))
        for i, occ in enumerate(occurrences):
            if occ != 0:
                s[count : count + occ, :] = self.dists[i].sample(occ)  # noqa: E203
                count += occ

        order = np.argsort(d)
        s = s[order, :]  # noqa: E203

        return s

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        assert xs.shape[-1] == self.input_dim, "Dimension mismatch"

        p = np.zeros(1) if xs.ndim == 1 else np.zeros(xs.shape[0])

        for i, dist in enumerate(self.dists):
            p += self.w[i] * dist.pdf(xs)

        return p
