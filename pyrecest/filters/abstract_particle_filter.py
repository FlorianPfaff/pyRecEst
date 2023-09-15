from collections.abc import Callable

import numpy as np
from beartype import beartype
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter_type import AbstractFilterType


class AbstractParticleFilter(AbstractFilterType):
    def __init__(self, initial_filter_state=None):
        AbstractFilterType.__init__(self, initial_filter_state)

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(f=lambda x: x, noise_distribution=noise_distribution)

    @beartype
    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
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

    @beartype
    def update_identity(
        self,
        noise_distribution: AbstractManifoldSpecificDistribution,
        measurement,
        shift_instead_of_add: bool = True,
    ):
        assert measurement is None or np.size(measurement) == noise_distribution.dim
        assert (
            np.ndim(measurement) == 1
            or np.ndim(measurement) == 0
            and noise_distribution.dim == 1
        )
        if not shift_instead_of_add:
            raise NotImplementedError()

        likelihood = noise_distribution.set_mode(measurement).pdf
        self.update_nonlinear_using_likelihood(likelihood)

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            self.filter_state = self.filter_state.reweigh(likelihood)
        else:
            self.filter_state = self.filter_state.reweigh(
                lambda x: likelihood(measurement, x)
            )

        self.filter_state.d = self.filter_state.sample(self.filter_state.w.shape[0])
        self.filter_state.w = (
            1 / self.filter_state.w.shape[0] * np.ones_like(self.filter_state.w)
        )

    @beartype
    def association_likelihood(self, likelihood: AbstractManifoldSpecificDistribution):
        likelihood_val = np.sum(
            likelihood.pdf(self.filter_state.d) * self.filter_state.w
        )
        return likelihood_val
