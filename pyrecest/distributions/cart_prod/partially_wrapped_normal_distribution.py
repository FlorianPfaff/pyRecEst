import copy

import numpy as np
from scipy.stats import multivariate_normal
from itertools import product

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class PartiallyWrappedNormalDistribution(AbstractHypercylindricalDistribution):
    def __init__(
        self, mu: np.ndarray, C: np.ndarray, bound_dim: int | np.int32 | np.int64
    ):
        assert bound_dim >= 0, "bound_dim must be non-negative"
        assert np.ndim(mu) == 1, "mu must be a 1-dimensional array"
        assert C.shape == (np.size(mu), np.size(mu)), "C must match size of mu"
        assert np.allclose(C, C.T), "C must be symmetric"
        assert np.all(np.linalg.eigvals(C) > 0), "C must be positive definite"
        assert bound_dim <= np.size(mu)
        assert np.ndim(mu) == 1
        if bound_dim > 0:  # This decreases the need for many wrappings
            mu[:bound_dim] = np.mod(mu[:bound_dim], 2 * np.pi)

        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim=bound_dim, lin_dim=np.size(mu) - bound_dim
        )

        self.mu = mu
        self.mu[:bound_dim] = np.mod(self.mu[:bound_dim], 2 * np.pi)
        self.C = C

    def _reshape_and_generate_offsets_transposed(self, matrix, n_wrappings):
        matrix = np.atleast_2d(matrix)
        n = matrix.shape[0]
        offset_values = [i*2*np.pi for i in range(-n_wrappings, n_wrappings+1)]
        all_combinations = list(product(offset_values, repeat=self.bound_dim))
        offset_matrix = np.array(all_combinations).T
        offset_matrix = np.tile(offset_matrix, (n, 1, 1))
        bounded_matrix = matrix[:, :self.bound_dim, np.newaxis] + offset_matrix
        linear_matrix = matrix[:, self.bound_dim:, np.newaxis]
        expanded_linear_matrix = np.tile(linear_matrix, (1, 1, (1+2*n_wrappings)**self.bound_dim))
        combined_matrix = np.concatenate((bounded_matrix, expanded_linear_matrix), axis=1)
        return np.transpose(combined_matrix, (2, 0, 1))
    
    def pdf(self, xs: np.ndarray, n_wrappings: int = 3) -> np.ndarray:
        mvn = multivariate_normal(self.mu, self.C)
        transposed_matrix = self._reshape_and_generate_offsets_transposed(xs, n_wrappings)
        flattened_data = transposed_matrix.reshape(-1, transposed_matrix.shape[-1])
        pdf_values = mvn.pdf(flattened_data)
        return pdf_values.reshape(transposed_matrix.shape[0], transposed_matrix.shape[1]).sum(axis=0)
      
    def mode(self):
        """
        Determines the mode of the distribution, i.e., the point where the pdf is largest.
        Returns:
            m (lin_dim + bound_dim,) vector: the mode
        """
        return self.mu

    def set_mode(self, new_mode: np.ndarray):
        self.mu = copy.copy(new_mode)
        return self

    def hybrid_moment(self):
        """
        Calculates mean of [x1, x2, .., x_lin_dim, cos(x_(linD+1), sin(x_(linD+1)), ..., cos(x_(linD+boundD), sin(x_(lin_dim+bound_dim))]
        Returns:
            mu (linD+2,): expectation value of [x1, x2, .., x_lin_dim, cos(x_(lin_dim+1), sin(x_(lin_dim+1)), ..., cos(x_(lin_dim+bound_dim), sin(x_(lin_dim+bound_dim))]
        """
        mu = np.empty((2 * self.bound_dim + self.lin_dim))
        mu[2 * self.bound_dim :] = self.mu[self.bound_dim :]  # noqa: E203
        for i in range(self.bound_dim):
            mu[2 * i] = np.cos(self.mu[i]) * np.exp(-self.C[i, i] / 2)  # noqa: E203
            mu[2 * i + 1] = np.sin(self.mu[i]) * np.exp(  # noqa: E203
                -self.C[i, i] / 2
            )
        return mu

    def hybrid_mean(self):
        return self.mu

    def linear_mean(self):
        return self.mu[-self.lin_dim :]  # noqa: E203

    def periodic_mean(self):
        return self.mu[: self.bound_dim]

    def sample(self, n: int):
        """
        Sample n points from the distribution
        Parameters:
            n (int): number of points to sample
        """
        assert n > 0, "n must be positive"
        s = np.random.multivariate_normal(self.mu, self.C, n)
        s[:, : self.bound_dim] = np.mod(s[:, : self.bound_dim], 2 * np.pi)  # noqa: E203
        return s

    def to_gaussian(self):
        return GaussianDistribution(self.mu, self.C)

    def linear_covariance(self):
        return self.C[self.bound_dim :, self.bound_dim :]  # noqa: E203

    def marginalize_periodic(self):
        return GaussianDistribution(
            self.mu[self.bound_dim :],  # noqa: E203
            self.C[self.bound_dim :, self.bound_dim :],  # noqa: E203
        )

    def marginalize_linear(self):
        return HypertoroidalWrappedNormalDistribution(
            self.mu[: self.bound_dim],
            self.C[: self.bound_dim, : self.bound_dim],  # noqa: E203
        )
