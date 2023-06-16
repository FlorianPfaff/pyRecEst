import unittest
import numpy as np
from pyrecest.distributions.circle.circular_dirac_distribution import CircularDiracDistribution
from pyrecest.filters.circular_particle_filter import CircularParticleFilter
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions import WrappedNormalDistribution, HypertoroidalDiracDistribution
from pyrecest.distributions.circle.circular_uniform_distribution import CircularUniformDistribution


class CircularParticleFilterTest(unittest.TestCase):
    
    def setUp(self):
        self.n_particles = 30
        self.filter = CircularParticleFilter(self.n_particles)
        self.dist = self.filter.filter_state
        self.wn = WrappedNormalDistribution(1.3, 0.8)

    def test_estimate(self):
        self.assertTrue(np.allclose(self.dist.trigonometric_moment(1), 0, atol=1e-10))

    def test_set_state(self):
        # sanity check
        self.filter.filter_state = self.dist
        dist1 = self.filter.filter_state
        self.assertIsInstance(dist1, HypertoroidalDiracDistribution)
        self.assertEqual(dist1.dim, 1)
        np.testing.assert_almost_equal(self.dist.d, dist1.d)
        np.testing.assert_almost_equal(self.dist.w, dist1.w)
        
        self.filter.set_state(VonMisesDistribution(0, 1))
        self.assertEqual(len(self.dist.d), self.n_particles)

    def test_sampling(self):
        positions = np.arange(0, 1.1, 0.1)
        dist3 = CircularDiracDistribution(positions)
        np.random.seed(0)
        num_samples = 20
        samples = dist3.sample(num_samples)
        self.assertEqual(samples.shape, (num_samples,))
        for i in range(num_samples):
            self.assertIn(samples[i], positions)

    def test_prediction(self):
        # test prediction
        self.filter.filter_state = self.dist
        f = lambda x: x
        self.filter.predict_nonlinear(f, self.wn)
        dist2 = self.filter.filter_state
        self.assertIsInstance(dist2, HypertoroidalDiracDistribution)
        self.assertEqual(dist2.dim, 1)
        
        self.filter.set_state(self.dist)
        self.filter.predict_identity(self.wn)
        dist2_identity = self.filter.filter_state
        self.assertIsInstance(dist2_identity, HypertoroidalDiracDistribution)
        self.assertEqual(dist2_identity.dim, 1)
        np.testing.assert_almost_equal(dist2.w, dist2_identity.w)

    def test_nonlinear_prediction_without_noise(self):
        # nonlinear test without noise
        self.filter.set_state(self.dist)
        f = lambda x: x**2
        no_noise = CircularDiracDistribution(np.array([0]))
        self.filter.predict_nonlinear(f, no_noise)
        predicted = self.filter.filter_state
        self.assertIsInstance(predicted, HypertoroidalDiracDistribution)
        dist_f = self.dist.apply_function(f)
        np.testing.assert_almost_equal(predicted.d, dist_f.d, decimal=10)
        np.testing.assert_almost_equal(predicted.w, dist_f.w, decimal=10)


    def test_update(self):
        # test update
        np.random.seed(0)
        self.filter.set_state(self.dist)
        h = lambda x: x
        z = 0
        likelihood = lambda z, x: self.wn.pdf(z - h(x))
        self.filter.update_nonlinear(likelihood, z)
        dist3a = self.filter.filter_state
        self.assertIsInstance(dist3a, CircularDiracDistribution)
        self.filter.set_state(self.dist)
        self.filter.update_identity(self.wn, z)
        dist3b = self.filter.filter_state
        self.assertIsInstance(dist3b, CircularDiracDistribution)

    def test_association_likelihood(self):
        dist = CircularDiracDistribution(np.array([1, 2, 3]), np.array([1/3, 1/3, 1/3]))
        pf = CircularParticleFilter(3)
        pf.set_state(dist)

        self.assertAlmostEqual(pf.association_likelihood(CircularUniformDistribution()), 1 / (2 * np.pi), places=10)
        self.assertGreater(pf.association_likelihood(VonMisesDistribution(2, 1)), 1 / (2 * np.pi))

        self.filter.set_state(CircularDiracDistribution(np.arange(0, 1.1, 0.1)))
        likelihood = lambda _, x: x == 0.5
        self.filter.update_nonlinear(likelihood, 42)
        estimation = self.filter.filter_state
        self.assertIsInstance(estimation, CircularDiracDistribution)
        for i in range(len(estimation.d)):
            self.assertEqual(estimation.d[i], 0.5)

        # test update with single parameter likelihood
        np.random.seed(0)
        self.filter_state = self.dist
        wn = WrappedNormalDistribution(1.3, 0.8)
        likelihood = lambda x: wn.pdf(-x)
        self.filter.update_nonlinear(likelihood)
        dist3c = self.filter_state
        self.assertIsInstance(dist3c, HypertoroidalDiracDistribution)

if __name__ == "__main__":
    unittest.main()
