from math import pi
from pyrecest.backend import sum
from pyrecest.backend import stack
from pyrecest.backend import sqrt
from pyrecest.backend import meshgrid
from pyrecest.backend import linspace
from pyrecest.backend import array
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyrecest.distributions import (
    AbstractHypersphereSubsetDistribution,
    HypersphericalMixture,
    VonMisesFisherDistribution,
    WatsonDistribution,
)


class HypersphericalMixtureTest(unittest.TestCase):
    def test_pdf_3d(self):
        wad = WatsonDistribution(array([0.0, 0.0, 1.0]), -10.0)
        vmf = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)
        w = array([0.3, 0.7])
        smix = HypersphericalMixture([wad, vmf], w)

        phi, theta = meshgrid(
            linspace(0, 2 * pi, 10), linspace(-pi / 2, pi / 2, 10)
        )
        points = AbstractHypersphereSubsetDistribution.polar_to_cart(
            stack([phi.ravel(), theta.ravel()], axis=-1)
        )

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )

    def test_pdf_4d(self):
        wad = WatsonDistribution(array([0.0, 0.0, 0.0, 1.0]), -10)
        vmf = VonMisesFisherDistribution(array([0.0, 1.0, 0.0, 0.0]), 1)
        w = array([0.3, 0.7])
        smix = HypersphericalMixture([wad, vmf], w)

        a, b, c, d = np.mgrid[-1:1:4j, -1:1:4j, -1:1:4j, -1:1:4j]
        points = array([a.ravel(), b.ravel(), c.ravel(), d.ravel()]).T
        points = points / sqrt(sum(points**2, axis=1, keepdims=True))

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()