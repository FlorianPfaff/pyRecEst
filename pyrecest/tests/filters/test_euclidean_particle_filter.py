import numpy as np
import unittest

from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter
import copy

class EuclideanParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.C_prior = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        self.mu = np.array([5, 6, 7])
        self.prior = GaussianDistribution(self.mu, self.C_prior)
        self.sys_noise_default = GaussianDistribution(np.zeros_like(self.mu), 0.5 * self.C_prior)
        self.pf_init = EuclideanParticleFilter(n_particles=500, dim=3)
        self.forced_mean = np.array([1, 2, 3])

    def test_predict_update_cycle_3d(self):
        np.random.seed(42)
        pf = copy.deepcopy(self.pf_init)
        pf.set_state(self.prior)
        
        for _ in range(50):
            pf.predict_identity(GaussianDistribution(np.zeros(3), self.C_prior))
            self.assertEqual(pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                pf.update_identity(self.sys_noise_default, self.forced_mean)

        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_almost_equal(pf.get_point_estimate(), self.forced_mean, decimal=1)

    def test_predict_nonlinear_nonadditive(self):
        np.random.seed(42)
        pf = copy.deepcopy(self.pf_init)
        pf.set_state(self.prior)
        
        n = 5
        samples = np.random.rand(n, 3)
        weights = np.ones((n)) / n
        
        f = lambda x, w: x + w
        pf.predict_nonlinear_nonadditive(f, samples, weights)
        est = pf.get_point_estimate()
        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(est, self.prior.mu + np.mean(samples, axis=0), atol=0.1)

    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        np.random.seed(42)
        pf = copy.deepcopy(self.pf_init)
        pf.set_state(self.prior.set_mean(np.ones(3) + np.pi / 2))
        
        force_first_particle_pos = np.array([1.1, 2, 3])
        pf.filter_state.d[0, :] = force_first_particle_pos
        for _ in range(50):
            self.assertEqual(pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                pf.update_identity(self.sys_noise_default, self.forced_mean)

        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(pf.get_point_estimate(), force_first_particle_pos, atol=1e-10)

if __name__ == "__main__":
    unittest.main()
