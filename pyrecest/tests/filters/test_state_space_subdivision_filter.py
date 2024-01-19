import unittest
import numpy as np
from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import StateSpaceSubdivisionDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import CustomHypercylindricalDistribution
from pyrecest.distributions.hypersphere_subset.hemispherical_uniform_distribution import HemisphericalUniformDistribution
from pyrecest.distributions import (HyperhemisphericalUniformDistribution, GaussianDistribution)

class StateSpaceSubdivisionDistributionTest(unittest.TestCase):

    def test_constructor(self):
        n = 100
        rbd = StateSpaceSubdivisionDistribution(HyperhemisphericalGridDistribution.from_distribution(
            HyperhemisphericalUniformDistribution(3), n),
            np.tile(GaussianDistribution(np.array([0, 0, 0]), 1000 * np.eye(3)), (n, 1)))

        self.assertEqual(rbd.linear_distributions.shape, (100, 1))
        self.assertEqual(rbd.gd.grid_values.shape, (100, 1))

    # TODO more tests

    def test_from_function_h2xr(self):
        np.random.seed(0)  # Equivalent to rng default in MATLAB
        hud = HemisphericalUniformDistribution()
        gauss = GaussianDistribution(0, 1)
        cd = CustomHypercylindricalDistribution(lambda x: hud.pdf(x[:3, :]) * gauss.pdf(x[3, :]), 3, 1)

        apd = StateSpaceSubdivisionDistribution.from_function(lambda x: cd.pdf(x), 101, 3, 1, 'hyperhemisphere')

        points_bounded = np.random.randn(3, 100)
        points_cart = np.vstack((points_bounded / np.linalg.norm(points_bounded, axis=0), np.random.randn(1, 100)))

        np.testing.assert_allclose(apd.pdf(points_cart, 'nearest_neighbor', 'nearest_neighbor'), cd.pdf(points_cart), atol=5e-17)
        np.testing.assert_allclose(apd.pdf(points_cart, 'nearest_neighbor', 'grid_default'), cd.pdf(points_cart), atol=5e-17)

if __name__ == '__main__':
    unittest.main()


"""

import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution

from pyrecest.distributions.hypersphere_subset.hemispherical_grid_distribution import HemisphericalGridDistribution
from pyrecest.distributions.hypersphere_subset.hemispherical_uniform_distribution import HemisphericalUniformDistribution
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import StateSpaceSubdivisionGaussianDistribution
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.hypersphere_subset.custom_hemispherical_distribution import CustomHemisphericalDistribution

class StateSpaceSubdivisionFilterTest:
    def test_update(self):
        n = 10
        gd_init = HemisphericalGridDistribution.from_distribution(HemisphericalUniformDistribution(), n)
        apd = StateSpaceSubdivisionGaussianDistribution(gd_init, np.array([GaussianDistribution(np.array([1, 1]), 100 * np.eye(2)) for _ in range(n)]))

        lpf = StateSpaceSubdivisionFilter()
        lpf.set_state(apd)

        # Update with the same distribution
        lpf.update(None, apd.linear_distributions)
        assert np.allclose(np.stack([d.mu for d in lpf.apd.linear_distributions]), np.tile(np.array([1, 1]), (n, 1)).T)
        assert np.allclose(np.stack([d.cov for d in lpf.apd.linear_distributions]), np.tile(50 * np.eye(2), (n, 1, 1)))

        # Update with a non-diagonal distribution
        gaussian_likelihood = GaussianDistribution(np.array([3, 5]), np.array([[25, 5], [5, 25]]))
        posterior = lpf.apd.linear_distributions[0].multiply(gaussian_likelihood)
        lpf.update(None, gaussian_likelihood)
        assert np.allclose(np.stack([d.mu for d in lpf.apd.linear_distributions]), np.tile(posterior.mu, (n, 1)).T, rtol=1e-15)
        assert np.allclose(np.stack([d.cov for d in lpf.apd.linear_distributions]), np.tile(posterior.cov, (n, 1, 1)), rtol=1e-15)

        # Update with a VMF and no linear part
        vmf_full_sphere = VonMisesFisherDistribution(np.array([0, 1, 1]) / np.sqrt(2), 1)
        noise = CustomHemisphericalDistribution(lambda x: vmf_full_sphere.pdf(x) + vmf_full_sphere.pdf(-x))
        lpf.update(noise)
        assert np.allclose(np.stack([d.mu for d in lpf.apd.linear_distributions]), np.tile(posterior.mu, (n, 1)).T, rtol=1e-15)
        assert np.allclose(np.stack([d.cov for d in lpf.apd.linear_distributions]), np.tile(posterior.cov, (n, 1, 1)), rtol=1e-15)

        # Should be the same as directly approximating the density
        gd_forced_norm = HemisphericalGridDistribution.from_distribution(noise, n).normalize(tol=0)
        assert np.allclose(lpf.apd.gd.grid_values, gd_forced_norm.grid_values, rtol=5e-16)

        # Update with a VMF and Gaussian with all Gaussians equal
        likelihood = GaussianDistribution(np.array([1, -1]), np.array([[15, -0.5], [-0.5, 15]]))
        posteriors = np.array([d.multiply(likelihood) for d in lpf.apd.linear_distributions])
        lpf.update(noise, likelihood)

        assert np.allclose(np.stack([d.mu for d in lpf.apd.linear_distributions]), np.stack([p.mu for p in posteriors]), rtol=5e-15)
        assert np.allclose(np.stack([d.cov for d in lpf.apd.linear_distributions]), np.stack([p.cov for p in posteriors]), rtol=5e-15)

        posterior_numerical = CustomHemisphericalDistribution(lambda x: noise.pdf(x) * noise.pdf(x))
        gd_forced_norm = HemisphericalGridDistribution.from_distribution(posterior_numerical, n).normalize(tol=0)

        assert np.allclose(lpf.apd.gd.grid_values, gd_forced_norm.grid_values, rtol=5e-16)

        # Final test with different linear distributions
        likelihoods = np.array([GaussianDistribution(np.array([i, -i]), np.array([[15, i / 10], [i / 10, 15]])) for i in range(1, n + 1)])
        posteriors = np.array([lpf.apd.linear_distributions[0].multiply(likelihood) for likelihood in likelihoods])
        lpf.update(noise, likelihoods)

        assert np.allclose(np.stack([d.mu for d in lpf.apd.linear_distributions]), np.stack([p.mu for p in posteriors]), rtol=5e-15)
        assert np.allclose(np.stack([d.cov for d in lpf.apd.linear_distributions]), np.stack([p.cov for p in posteriors]), rtol=5e-15)

        assert np.sum(np.abs(lpf.apd.gd.grid_values - (1 / lpf.apd.gd.get_manifold_size()))) > np.sum(np.abs(gd_forced_norm.grid_values - (1 / lpf.apd.gd.get_manifold_size())))


"""