import unittest

import numpy as np
from pyrecest.distributions import VMFDistribution

vectors_to_test_2d = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0] / np.sqrt(2),
        [1, 1, 2] / np.linalg.norm([1, 1, 2]),
        -np.array([1, 1, 2]) / np.linalg.norm([1, 1, 2]),
    ]
)

class TestVMFDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([1, 2, 3])
        self.mu = self.mu / np.linalg.norm(self.mu)
        self.kappa = 2
        self.vmf = VMFDistribution(self.mu, self.kappa)
        self.other = VMFDistribution(np.array([0, 0, 1]), self.kappa / 3)

    def test_vmf_distribution_3d_sanity_check(self):
        self.assertIsInstance(self.vmf, VMFDistribution)
        self.assertTrue(np.allclose(self.vmf.mu, self.mu))
        self.assertEqual(self.vmf.kappa, self.kappa)
        self.assertEqual(self.vmf.dim + 1, len(self.mu))

    def test_vmf_distribution_3d_mode(self):
        self.assertTrue(np.allclose(self.vmf.mode_numerical(), self.vmf.mode(), atol=1e-5))

    def test_vmf_distribution_3d_integral(self):
        self.assertAlmostEqual(self.vmf.integrate(), 1, delta=1e-5)

    def test_vmf_distribution_3d_multiplication(self):
        vmf_mul = self.vmf.multiply(self.other)
        vmf_mul2 = self.other.multiply(self.vmf)
        c = vmf_mul.pdf(np.array([1, 0, 0])) / (self.vmf.pdf(np.array([1, 0, 0])) * self.other.pdf(np.array([1, 0, 0])))
        x = np.array([0, 1, 0])
        self.assertAlmostEqual(self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul.pdf(x), delta=1e-10)
        self.assertAlmostEqual(self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul2.pdf(x), delta=1e-10)

    def test_vmf_distribution_3d_convolve(self):
        vmf_conv = self.vmf.convolve(self.other)
        self.assertTrue(np.allclose(vmf_conv.mu, self.vmf.mu, atol=1e-10))
        d = 3
        self.assertAlmostEqual(VMFDistribution.a_d(d, vmf_conv.kappa),
                               VMFDistribution.a_d(d, self.vmf.kappa) * VMFDistribution.a_d(d, self.other.kappa), delta=1e-10)
        
    def test_init_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VMFDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 7.22562325261744e-05)

    def test_init_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VMFDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 0.0318492506152322)

    def test_pdf_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VMFDistribution(mu, kappa)

        np.testing.assert_array_almost_equal(
            dist.pdf(vectors_to_test_2d),
            np.array(
                [
                    0.00428425301914546,
                    0.00428425301914546,
                    0.254024093013817,
                    0.0232421165060131,
                    1.59154943419939,
                    3.28042788159008e-09,
                ],
            ),
        )

    def test_pdf_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VMFDistribution(mu, kappa)

        xs_unnorm = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
            ]
        )
        xs = xs_unnorm / np.linalg.norm(xs_unnorm, axis=1, keepdims=True)

        np.testing.assert_array_almost_equal(
            dist.pdf(xs),
            np.array(
                [
                    0.0533786916025838,
                    0.0533786916025838,
                    0.0894615936690536,
                    0.00676539409350726,
                    0.0661093769275549,
                    0.0318492506152322,
                    0.0952456142366906,
                    0.0221063629087443,
                    0.0153438863274034,
                    0.13722300001807,
                ],
            ),
        )

    def test_mean_direction(self):
        mu = 1 / np.sqrt(2) * np.array([1, 1, 0])
        vmf = VMFDistribution(mu, 1)
        self.assertTrue(np.allclose(vmf.mean_direction(), mu, atol=1e-13))
        
    def test_hellinger_distance_2d(self):
        # 2D
        vmf1 = VMFDistribution(np.array([1, 0]), 0.9)
        vmf2 = VMFDistribution(np.array([0, 1]), 1.7)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf1), 0, delta=1e-10)
        self.assertAlmostEqual(vmf2.hellinger_distance(vmf2), 0, delta=1E-10)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf2), vmf1.hellinger_distance_numerical(vmf2), delta=1E-10)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf2), vmf2.hellinger_distance(vmf1), delta=1E-10)

    def test_hellinger_distance_3d(self):
        # 3D
        vmf1 = VMFDistribution(np.array([1, 0, 0]), 0.6)
        mu2 = np.array([1, 2, 3])
        vmf2 = VMFDistribution(mu2 / np.linalg.norm(mu2), 2.1)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf1), 0, delta=1E-10)
        self.assertAlmostEqual(vmf2.hellinger_distance(vmf2), 0, delta=1E-10)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf2), vmf1.hellinger_distance_numerical(vmf2), delta=1E-6)
        self.assertAlmostEqual(vmf1.hellinger_distance(vmf2), vmf2.hellinger_distance(vmf1), delta=1E-10)
        

if __name__ == "__main__":
    unittest.main()
