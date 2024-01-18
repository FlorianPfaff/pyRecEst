import unittest
import numpy as np
import quaternion
from parameterized import parameterized
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import HyperhemisphericalBinghamDistribution

def generate_test_cases():
    test_cases = []
    for i in range(1, 5):
        if i == 1:
            M = np.eye(2)
            Z = np.array([-3, 0])
        elif i == 2:
            phi = 0.7
            M = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
            Z = np.array([-5, 0])
        elif i == 3:
            M = np.eye(4)
            Z = np.array([-10, -2, -1, 0])
        elif i == 4:
            q = np.qunaterion(1, 2, 3, 4).normalize()
            q1 = np.quaternion(1, 0, 0, 0)
            q2 = np.quaternion(0, 1, 0, 0)
            q3 = np.quaternion(0, 0, 1, 0)
            q4 = np.quaternion(0, 0, 0, 1)

            m1 = q * q1
            m2 = q * q2
            m3 = q * q3
            m4 = q * q4
            
            # Convert to array and stack horizontally
            M = np.hstack([quaternion.as_float_array(m1)[:, None], 
               quaternion.as_float_array(m2)[:, None], 
               quaternion.as_float_array(m3)[:, None], 
               quaternion.as_float_array(m4)[:, None]])
            
            
            Z = np.array([-10, -2, -1, 0])
            
        test_cases.append((M, Z))
    
    return test_cases

class HyperhemisphericalBinghamDistributionTest(unittest.TestCase):
    @parameterized.expand(generate_test_cases())
    def test_pdf(self, M, Z):
        B = HyperhemisphericalBinghamDistribution(Z, M)
        testpoints = np.random.rand(B.dim, 20)
        testpoints /= np.sum(testpoints, axis=0)
        for point in testpoints.T:
            expected = 2 / B.F * np.exp(point.T @ M @ np.diag(Z) @ M.T @ point)
            self.assertAlmostEqual(B.pdf(point), expected, places=10)

    @parameterized.expand(generate_test_cases())
    def test_sanity_check(self, M, Z):
        B = HyperhemisphericalBinghamDistribution(Z, M)
        self.assertIsInstance(B, HyperhemisphericalBinghamDistribution)
        np.testing.assert_array_almost_equal(B.M, M)
        np.testing.assert_array_almost_equal(B.Z, Z)
        self.assertEqual(B.dim, len(Z))

if __name__ == '__main__':
    unittest.main()
