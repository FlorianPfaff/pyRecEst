
import numpy as np

from .hypertoroidal_particle_filter import HypertoroidalParticleFilter


class CircularParticleFilter(HypertoroidalParticleFilter):
    def __init__(self, n_particles: int | np.int32 | np.int64) -> None:
        """
        Initialize the CircularParticleFilter.

        :param n_particles: number of particles
        """
        super().__init__(n_particles, 1)

    def compute_association_likelihood(self, likelihood) -> np.float64:
        """
        Compute the likelihood of association based on the PDF of the likelihood
        and the filter state.

        :param likelihood: likelihood object with a PDF method
        :return: association likelihood value
        """
        likelihood_val = np.sum(
            likelihood.pdf(self.filter_state.d) * self.filter_state.w
        )
        return likelihood_val
