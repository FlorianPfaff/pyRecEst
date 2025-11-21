import unittest
from math import pi

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import AbstractHypertoroidalDistribution
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi), array(0.0)), pi
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(0), array(2 * pi)), 0
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(
                array(pi / 4), array(7 * pi / 4)
            ),
            pi / 2,
            rtol=2e-07,
        )

    def test_plot_2d(self):
        mu = array([0.0, 1.0])
        sigma1 = 0.5
        sigma2 = 0.5
        rho = 0.5
        dist = ToroidalWrappedNormalDistribution(
            mu,
            array([[sigma1, sigma1 * sigma2 * rho], [sigma1 * sigma2 * rho, sigma2]]),
        )
        dist.plot()
