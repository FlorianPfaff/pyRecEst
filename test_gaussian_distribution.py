import unittest
import numpy as np
from scipy.stats import multivariate_normal
from gaussian_distribution import GaussianDistribution
import scipy

class GaussianDistributionTest(unittest.TestCase):
    def test_gaussian_distribution_3d(self):
        mu = np.array([2, 3, 4])
        C = np.array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 0.1]])
        g = GaussianDistribution(mu, C)

        xa = np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
        self.assertTrue(np.allclose(g.pdf(xa), multivariate_normal.pdf(xa.T, mu, C).T, rtol=1e-10))

        n = 10
        s = g.sample(n)
        self.assertEqual(s.shape, (3, n))
    
    def test_mode(self):
        mu = np.array([1, 2, 3])
        C = np.array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 1]])
        g = GaussianDistribution(mu, C)

        self.assertTrue(np.allclose(g.mode(), mu, atol=1e-6))
    
    def test_shift(self):
        mu = np.array([1, 2, 3])
        C = np.array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 1]])
        g = GaussianDistribution(mu, C)

        shift_by = np.array([2, -2, 3])
        g_shifted = g.shift(shift_by)

        self.assertTrue(np.allclose(g_shifted.mode(), mu + shift_by, atol=1e-6))
    
    def test_marginalization(self):
        mu = np.array([1, 2])
        C = np.array([[1.1, 0.4], [0.4, 0.9]])
        g = GaussianDistribution(mu, C)

        grid = np.linspace(-10, 10, 30)
        dist_marginalized = g.marginalize_out(1)

        def marginlized_1D_via_integral(x):
            return np.array([scipy.integrate.quad(lambda y: g.pdf(np.array([xCurr, y])), -np.inf, np.inf)[0] for xCurr in x])

        self.assertTrue(np.allclose(dist_marginalized.pdf(grid), marginlized_1D_via_integral(grid), atol=1E-9))
    

if __name__ == '__main__':
    unittest.main()