import unittest

import numpy as np
from pyrecest.utils.metrics import anees


class TestANEES(unittest.TestCase):
    def setUp(self):
        self.groundtruths = np.array([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]])
        self.uncertainties = np.array(
            [[[1, 0.5], [0.5, 2]], [[1, 0], [0, 1]], [[0.5, 0], [0, 1.5]]]
        )
        self.num_samples = 10000

    def test_ANEES_is_close_to_one(self):
        samples = []

        for i in range(len(self.groundtruths)):
            samples_for_i = np.random.multivariate_normal(
                mean=self.groundtruths[i],
                cov=self.uncertainties[i],
                size=self.num_samples,
            )
            samples.extend(samples_for_i)

        samples = np.array(samples)
        repeated_groundtruths = np.repeat(self.groundtruths, self.num_samples, axis=0)
        repeated_uncertainties = np.repeat(self.uncertainties, self.num_samples, axis=0)

        computed_ANEES = anees(samples, repeated_uncertainties, repeated_groundtruths)

        # Assert that computed ANEES is close to 1 with a tolerance of 0.05.
        np.testing.assert_almost_equal(
            computed_ANEES, self.groundtruths.shape[-1], decimal=2
        )


if __name__ == "__main__":
    unittest.main()