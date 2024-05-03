import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, log
from pyrecest.distributions import BinghamDistribution, WatsonDistribution


class TestWatsonDistribution(unittest.TestCase):
    def setUp(self):
        self.xs = array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 2.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=float,
        )
        self.xs = self.xs / linalg.norm(self.xs, axis=1).reshape((-1, 1))

    def test_constructor(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)

        self.assertIsInstance(w, WatsonDistribution)
        npt.assert_array_equal(w.mu, mu)
        self.assertEqual(w.kappa, kappa)
        self.assertEqual(w.input_dim, mu.shape[0])

    def test_pdf(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)

        expected_pdf_values = array(
            [
                0.0388240901641662,
                0.229710245437696,
                0.0595974246790006,
                0.121741272709942,
                0.186880524436683,
                0.186880524436683,
            ]
        )

        pdf_values = w.pdf(self.xs)
        npt.assert_array_almost_equal(pdf_values, expected_pdf_values, decimal=5)

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        "Test not supported for this backend",
    )
    def test_integrate(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        kappa = 2.0
        w = WatsonDistribution(mu, kappa)
        self.assertAlmostEqual(w.integrate(), 1, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        "Test not supported for this backend",
    )
    def test_to_bingham(self):
        mu = array([1.0, 0.0, 0.0])
        kappa = 2.0
        watson_dist = WatsonDistribution(mu, kappa)
        bingham_dist = watson_dist.to_bingham()
        self.assertIsInstance(bingham_dist, BinghamDistribution)
        npt.assert_array_almost_equal(
            watson_dist.pdf(self.xs), bingham_dist.pdf(self.xs), decimal=5
        )

    def test_ln_norm_const(self):
        # Create an instance of WatsonDistribution
        # Here mu is a normalized vector, let's use [1, 0, 0] for simplicity in 3D
        mu = array([1.0, 0.0, 0.0])
        kappa = 3.0  # Arbitrary concentration parameter

        # Instantiate WatsonDistribution
        watson_dist = WatsonDistribution(mu, kappa)

        # Calculate norm_const and ln_norm_const
        norm_const = watson_dist.norm_const
        ln_norm_const = watson_dist.ln_norm_const

        # Check if ln_norm_const is the ln of norm_const
        expected_ln_norm_const = log(norm_const)

        # Use allclose to compare the floating-point results within some tolerance
        npt.assert_allclose(ln_norm_const, expected_ln_norm_const, rtol=1e-6)


# Running the tests
if __name__ == "__main__":
    unittest.main()
