import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
import scipy.linalg

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, column_stack, diag, linspace, meshgrid
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.mu = array([5.0, 1.0])
        self.C = array([[2.0, 1.0], [1.0, 1.0]])
        self.dist_2d = PartiallyWrappedNormalDistribution(self.mu, self.C, 1)

    @unittest.skipIf(
        pyrecest.backend.__name__ in ("pyrecest.pytorch", "pyrecest.jax"),
        reason="Not supported on this backend",
    )
    def test_pdf_2d(self):
        expected_vals = array(
            [  # From Matlab implementation
                0.00719442236938856,
                0.0251110014500013,
                0.0531599904868136,
                0.0682587789359472,
                0.0531599904868136,
                0.0100784602259792,
                0.0351772826718058,
                0.0744703080006016,
                0.0956217682613369,
                0.0744703080006016,
                0.00119956714181477,
                0.00418690072543581,
                0.00886366890530323,
                0.0113811761595142,
                0.00886366890530323,
                0.000447592726560109,
                0.00156225212096022,
                0.00330728776602597,
                0.00424664155187776,
                0.00330728776602597,
                0.00719442236938856,
                0.0251110014500013,
                0.0531599904868136,
                0.0682587789359472,
                0.0531599904868136,
            ]
        )

        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([1.0, 2.0])), 1
        )
        x, y = meshgrid(linspace(0.0, 2.0 * pi, 5), linspace(-1.0, 3.0, 5))
        points = column_stack([x.T.ravel(), y.T.ravel()])
        npt.assert_allclose(hwn.pdf(points), expected_vals, atol=1e-7)

    @unittest.skipIf(
        pyrecest.backend.__name__ in ("pyrecest.pytorch", "pyrecest.jax"),
        reason="Not supported on this backend",
    )
    def test_pdf_3d(self):
        expected_vals = array(
            [  # From Matlab implementation
                1.38549279e-07, 2.31011127e-08, 1.38549279e-07, 3.50045772e-07,
                5.83651314e-08, 3.50045772e-07, 2.65062051e-07, 4.41953100e-08,
                2.65062051e-07, 7.33676745e-08, 1.22330115e-08, 7.33676745e-08,
                1.38549279e-07, 2.31011127e-08, 1.38549279e-07, 1.47337010e-05,
                2.45663413e-06, 1.47337010e-05, 3.72248039e-05, 6.20670422e-06,
                3.72248039e-05, 2.81874076e-05, 4.69984751e-06, 2.81874076e-05,
                7.80211480e-06, 1.30089117e-06, 7.80211480e-06, 1.47337010e-05,
                2.45663413e-06, 1.47337010e-05, 4.13009579e-04, 6.88634465e-05,
                4.13009579e-04, 1.04347174e-03, 1.73984005e-04, 1.04347174e-03,
                7.90138838e-04, 1.31744362e-04, 7.90138838e-04, 2.18705955e-04,
                3.64660934e-05, 2.18705955e-04, 4.13009579e-04, 6.88634465e-05,
                4.13009579e-04
            ]
        )

        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0, 7.0]), diag(array([1.0, 2.0, 3.0])), 2
        )
        x, y, z = meshgrid(linspace(0.0, 2.0 * pi, 3), linspace(0.0, 2.0 * pi, 3), linspace(-1.0, 3.0, 3))
        points = column_stack([x.T.ravel(), y.T.ravel(), z.T.ravel()])
        npt.assert_allclose(hwn.pdf(points), expected_vals, atol=1e-7)

    def test_hybrid_mean_2d(self):
        npt.assert_allclose(self.dist_2d.hybrid_mean(), self.mu)

    def test_hybrid_mean_4d(self):
        mu = array([5.0, 1.0, 3.0, 4.0])
        C = array(
            scipy.linalg.block_diag([[2.0, 1.0], [1.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]])
        )
        dist = PartiallyWrappedNormalDistribution(mu, C, 2)
        npt.assert_allclose(dist.hybrid_mean(), mu)

    def test_hybrid_moment_2d(self):
        # Validate against precalculated values
        npt.assert_allclose(
            self.dist_2d.hybrid_moment(),
            [0.10435348, -0.35276852, self.mu[-1]],
            rtol=5e-7,
        )


if __name__ == "__main__":
    unittest.main()
