from math import pi
from pyrecest.backend import ones
from pyrecest.backend import array
import unittest

import numpy.testing as npt
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)


class CircularUniformDistributionTest(unittest.TestCase):
    def test_pdf(self):
        cu = CircularUniformDistribution()
        x = array([1, 2, 3, 4, 5, 6])

        # Test pdf
        npt.assert_allclose(cu.pdf(x), 1 / (2 * pi) * ones(x.shape))

    def test_shift(self):
        cu = CircularUniformDistribution()
        cu2 = cu.shift(3)
        x = array([1, 2, 3, 4, 5, 6])
        npt.assert_allclose(cu2.pdf(x), 1 / (2 * pi) * ones(x.shape))

    def test_cdf(self):
        cu = CircularUniformDistribution()
        x = array([1, 2, 3, 4, 5, 6])
        npt.assert_allclose(cu.cdf(x), cu.cdf_numerical(x))

    def test_cdf_with_shift(self):
        cu = CircularUniformDistribution()
        x = array([1, 2, 3, 4, 5, 6])
        cu2 = cu.shift(3)
        npt.assert_allclose(cu2.cdf(x), cu2.cdf_numerical(x))

    def test_trigonometric_moment(self):
        cu = CircularUniformDistribution()
        npt.assert_allclose(
            cu.trigonometric_moment(0), cu.trigonometric_moment_numerical(0)
        )
        npt.assert_allclose(cu.trigonometric_moment(0), 1)

    def test_trigonometric_moment_with_shift(self):
        cu = CircularUniformDistribution()
        npt.assert_allclose(
            cu.trigonometric_moment(1), cu.trigonometric_moment_numerical(1), atol=1e-10
        )
        npt.assert_allclose(cu.trigonometric_moment(1), 0, atol=1e-10)

    def test_integral(self):
        cu = CircularUniformDistribution()
        npt.assert_allclose(cu.integrate(), cu.integrate_numerically())
        npt.assert_allclose(cu.integrate(), 1)

    def test_integral_with_range(self):
        cu = CircularUniformDistribution()
        npt.assert_allclose(
            cu.integrate(array([1.0, 4.0])), cu.integrate_numerically(array([1.0, 4.0]))
        )
        npt.assert_allclose(
            cu.integrate(array([-4.0, 11.0])), cu.integrate_numerically(array([-4.0, 11.0]))
        )
        npt.assert_allclose(
            cu.integrate(array([2.0 * pi, -1.0])), cu.integrate_numerically(array([2.0 * pi, -1.0]))
        )

    def test_mean(self):
        cu = CircularUniformDistribution()
        with self.assertRaises(ValueError):
            cu.mean_direction()

    def test_entropy(self):
        cu = CircularUniformDistribution()
        npt.assert_allclose(cu.entropy(), cu.entropy_numerical())

    def test_sampling(self):
        cu = CircularUniformDistribution()
        n = 10
        s = cu.sample(n)
        npt.assert_allclose(s.shape[0], n)