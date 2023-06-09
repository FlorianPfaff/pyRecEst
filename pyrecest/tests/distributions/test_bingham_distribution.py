import unittest

import numpy as np
from pyrecest.distributions import BinghamDistribution

from .test_von_mises_fisher_distribution import vectors_to_test_2d


class TestBinghamDistribution(unittest.TestCase):
    def setUp(self):
        """Setup BinghamDistribution instance for testing."""
        M = np.array(
            [[1 / 3, 2 / 3, -2 / 3], [-2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]]
        )
        Z = np.array([-5, -3, 0])
        self.bd = BinghamDistribution(Z, M)

    def test_pdf(self):
        """Test pdf method with a fixed set of values."""
        expected_values = np.array(
            [
                0.0767812166360095,
                0.0145020985787277,
                0.0394207910410773,
                0.0267197897401937,
                0.0298598745474396,
                0.0298598745474396,
            ],
        )
        computed_values = self.bd.pdf(vectors_to_test_2d)
        np.testing.assert_array_almost_equal(
            computed_values,
            expected_values,
            err_msg="Expected and computed pdf values do not match.",
        )


if __name__ == "__main__":
    unittest.main()
