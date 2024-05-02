import unittest

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, diag, linalg, sum, tile
from pyrecest.distributions import (
    GaussianDistribution,
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.se3_cart_prod_stacked_distribution import (
    SE3CartProdStackedDistribution,
)
from pyrecest.distributions.se3_dirac_distribution import SE3DiracDistribution
import numpy.testing as npt

class SE3DiracDistributionTest(unittest.TestCase):
    def test_constructor(self):
        dSph = array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 4.0, 0.0, 0.5, 1.0, 1.0],
                [5.0, 10.0, 20, 30, 40, 50],
                [2.0, 31.0, 42, 3, 9.9, 5],
            ]
        ).T
        dSph = dSph / linalg.norm(dSph, None, -1).reshape(-1, 1)
        dLin = tile(array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]), (3, 1)).T
        w = array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        w = w / sum(w)
        SE3DiracDistribution(concatenate((dSph, dLin), axis=-1), w)

    def test_from_distribution(self):
        cpsd = SE3CartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(
                    array([1.0, 2.0, 3.0]).T, diag(array([3.0, 2.0, 1.0]))
                ),
            ]
        )
        dirac_dist = SE3DiracDistribution.from_distribution(cpsd, 100)
        npt.assert_allclose(dirac_dist.mean(), cpsd.mean())
        


if __name__ == "__main__":
    unittest.main()
