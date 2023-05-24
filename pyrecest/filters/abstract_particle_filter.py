import numpy as np
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.distributions import AbstractLinearDistribution
from typing import Callable, Optional   

from .abstract_filter_type import AbstractFilterType


class AbstractParticleFilter(AbstractFilterType):
    def __init__(self):
        AbstractFilterType.__init__(self, None)

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(lambda x: x, noise_distribution)

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: Optional[AbstractLinearDistribution] = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        assert (
            noise_distribution is None
            or self.filter_state.dim == noise_distribution.dim
        )

        if function_is_vectorized:
            self.filter_state.d = f(self.filter_state.d)
        else:
            self.filter_state = self.filter_state.apply_function(f)

        if noise_distribution is not None:
            if not shift_instead_of_add:
                noise = noise_distribution.sample(self.filter_state.w.size)
                self.filter_state.d = self.filter_state.d + noise
            else:
                for i in range(self.filter_state.d.shape[1]):
                    noise_curr = noise_distribution.set_mean(self.filter_state.d[i, :])
                    self.filter_state.d[i, :] = noise_curr.sample(1)

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        assert (
            samples.shape[0] == weights.size
        ), "samples and weights must match in size"

        weights = weights / np.sum(weights)
        n = self.filter_state.w.size
        noise_ids = np.random.choice(weights.size, n, p=weights)
        d = np.zeros((n, self.filter_state.dim))
        for i in range(n):
            d[i, :] = f(self.filter_state.d[i, :], samples[noise_ids[i]])

        self.filter_state.d = d

    def update_identity(self, noise_distribution, z, shift_instead_of_add=True):
        assert z is None or np.size(z) == noise_distribution.dim
        assert np.ndim(z) == 1 or np.ndim(z) == 0 and noise_distribution.dim == 1
        if not shift_instead_of_add:
            raise NotImplementedError()

        noise_for_likelihood = noise_distribution.set_mode(z)
        likelihood = noise_for_likelihood.pdf
        self.update_nonlinear(likelihood)

    def update_nonlinear(self, likelihood, z=None):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            assert (
                z is None
            ), "Cannot pass a density and a measurement. To assume additive noise, use update_identity."
            likelihood = likelihood.pdf

        if z is None:
            self.filter_state = self.filter_state.reweigh(likelihood)
        else:
            self.filter_state = self.filter_state.reweigh(lambda x: likelihood(z, x))

        self.filter_state.d = self.filter_state.sample(self.filter_state.d.shape[0])
        self.filter_state.w = np.full(
            self.filter_state.d.shape[0], 1 / self.filter_state.d.shape[0]
        )

    def association_likelihood(self, likelihood):
        likelihood_val = np.sum(
            likelihood.pdf(self.filter_state().d) * self.filter_state().w
        )
        return likelihood_val
