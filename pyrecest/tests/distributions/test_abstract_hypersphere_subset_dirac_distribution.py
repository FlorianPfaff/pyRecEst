import unittest
import numpy as np
import quaternion

from pyrecest.distributions import BinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import HyperhemisphericalBinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import HypersphericalDiracDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import HyperhemisphericalDiracDistribution

class AbstractHypersphereSubsetDiracDistributionTest(unittest.TestCase):

    def test_mean_axis(self):
        np.random.seed(0)
        q = np.quaternion(1, 2, 3, 4)  # create a quaternion object
        q = q.normalized()  # normalize the quaternion
        M = np.array([(q * np.quaternion(1, 0, 0, 0)).components,
                (q * np.quaternion(0, 1, 0, 0)).components,
                (q * np.quaternion(0, 0, 1, 0)).components,
                (q * np.quaternion(0, 0, 0, 1)).components])  # use quaternion multiplication from the library
        Z = np.array([-10, -2, -1, 0])

        bd = BinghamDistribution(Z, M)
        bdHemi = HyperhemisphericalBinghamDistribution(Z, M)
        wdFull = HypersphericalDiracDistribution.from_distribution(bd, 1001)
        wdHemi = HyperhemisphericalDiracDistribution.from_distribution(bdHemi, 1001)

        self.assertTrue(np.allclose(wdFull.mean_axis(), bd.mean_axis(), atol=0.1))
        self.assertTrue(np.allclose(wdHemi.mean_axis(), bd.mean_axis(), atol=0.1))

if __name__ == '__main__':
    unittest.main()
