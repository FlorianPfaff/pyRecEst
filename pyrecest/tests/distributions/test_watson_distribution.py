from pyrecest.backend import linalg
from pyrecest.backend import array
import unittest

import numpy as np
from pyrecest.distributions import BinghamDistribution, WatsonDistribution


class TestWatsonDistribution(unittest.TestCase):
    def setUp(self):
        self.xs = array(
            [[1, 0, 0], [1, 2, 2], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, -1, -1]],
            dtype=float,
        )
        self.xs = self.xs / linalg.norm(self.xs, axis=1, keepdims=True)

    def test_constructor(self):
        mu = array([1, 2, 3])
        mu = mu / linalg.norm(mu)
        kappa = 2
        w = WatsonDistribution(mu, kappa)

        self.assertIsInstance(w, WatsonDistribution)
        np.testing.assert_array_equal(w.mu, mu)
        self.assertEqual(w.kappa, kappa)
        self.assertEqual(w.input_dim, mu.shape[0])

    def test_pdf(self):
        mu = array([1, 2, 3])
        mu = mu / linalg.norm(mu)
        kappa = 2
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
        np.testing.assert_almost_equal(pdf_values, expected_pdf_values, decimal=5)

    def test_integrate(self):
        mu = array([1, 2, 3])
        mu = mu / linalg.norm(mu)
        kappa = 2
        w = WatsonDistribution(mu, kappa)
        self.assertAlmostEqual(w.integrate(), 1, delta=1e-5)

    def test_to_bingham(self):
        mu = array([1.0, 0.0, 0.0])
        kappa = 2.0
        watson_dist = WatsonDistribution(mu, kappa)
        bingham_dist = watson_dist.to_bingham()
        self.assertIsInstance(bingham_dist, BinghamDistribution)
        np.testing.assert_almost_equal(
            watson_dist.pdf(self.xs), bingham_dist.pdf(self.xs), decimal=5
        )


if __name__ == "__main__":
    unittest.main()