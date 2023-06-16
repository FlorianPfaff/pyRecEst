import unittest

import numpy as np
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)


class CircularUniformDistributionTest(unittest.TestCase):
    def test_pdf(self):
        cu = CircularUniformDistribution()
        x = np.array([1, 2, 3, 4, 5, 6])

        # Test pdf
        np.testing.assert_allclose(cu.pdf(x), 1 / (2 * np.pi) * np.ones(x.shape))

    def test_shift(self):
        cu = CircularUniformDistribution()
        cu2 = cu.shift(3)
        x = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(cu2.pdf(x), 1 / (2 * np.pi) * np.ones(x.shape))

    def test_cdf(self):
        cu = CircularUniformDistribution()
        x = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(cu.cdf(x), cu.cdf_numerical(x))

    def test_cdf_with_shift(self):
        cu = CircularUniformDistribution()
        x = np.array([1, 2, 3, 4, 5, 6])
        cu2 = cu.shift(3)
        np.testing.assert_allclose(cu2.cdf(x), cu2.cdf_numerical(x))

    def test_trigonometric_moment(self):
        cu = CircularUniformDistribution()
        np.testing.assert_allclose(
            cu.trigonometric_moment(0), cu.trigonometric_moment_numerical(0)
        )
        np.testing.assert_allclose(cu.trigonometric_moment(0), 1)

    def test_trigonometric_moment_with_shift(self):
        cu = CircularUniformDistribution()
        np.testing.assert_allclose(
            cu.trigonometric_moment(1), cu.trigonometric_moment_numerical(1), atol=1e-10
        )
        np.testing.assert_allclose(cu.trigonometric_moment(1), 0, atol=1e-10)

    def test_integral(self):
        cu = CircularUniformDistribution()
        np.testing.assert_allclose(cu.integrate(), cu.integrate_numerically())
        np.testing.assert_allclose(cu.integrate(), 1)

    def test_integral_with_range(self):
        cu = CircularUniformDistribution()
        np.testing.assert_allclose(
            cu.integrate([1, 4]), cu.integrate_numerically([1, 4])
        )
        np.testing.assert_allclose(
            cu.integrate([-4, 11]), cu.integrate_numerically([-4, 11])
        )
        np.testing.assert_allclose(
            cu.integrate([2 * np.pi, -1]), cu.integrate_numerically([2 * np.pi, -1])
        )

    def test_mean(self):
        cu = CircularUniformDistribution()
        with self.assertRaises(ValueError):
            cu.mean_direction()

    def test_entropy(self):
        cu = CircularUniformDistribution()
        np.testing.assert_allclose(cu.entropy(), cu.entropy_numerical())

    def test_sampling(self):
        cu = CircularUniformDistribution()
        n = 10
        s = cu.sample(n)
        np.testing.assert_allclose(s.shape[1], n)
