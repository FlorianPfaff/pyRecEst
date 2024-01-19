import numpy as np
from scipy.integrate import nquad
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import HypertoroidalWrappedNormalDistribution
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import ToroidalVonMisesSineDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import HypertoroidalGridDistribution
import unittest
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import ToroidalFourierDistribution
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
from pyrecest.distributions.circle.custom_circular_distribution import CustomCircularDistribution

class HypertoroidalFourierDistributionTest(unittest.TestCase):
    def test_normalization_2D(self):
        unnormalized_coeffs_2D = np.fft.fftshift(np.fft.fftn(np.random.rand(3, 7) + 0.5))
        unnormalized_coeffs_2D[1, 3] = 1

        hfd_id = HypertoroidalFourierDistribution(unnormalized_coeffs_2D, 'identity')
        hfd_sqrt = HypertoroidalFourierDistribution(unnormalized_coeffs_2D, 'sqrt')

        def id_pdf(x, y):
            return hfd_id.pdf(np.array([x, y]))

        def sqrt_pdf(x, y):
            return hfd_sqrt.pdf(np.array([x, y]))

        result_id, _ = nquad(id_pdf, [[0, 2 * np.pi], [0, 2 * np.pi]])
        result_sqrt, _ = nquad(sqrt_pdf, [[0, 2 * np.pi], [0, 2 * np.pi]])

        self.assertAlmostEqual(result_id, 1)
        self.assertAlmostEqual(result_sqrt, 1)
    
    def test_integral_2d(self):
        # Test against implementation in toroidal (test case that this
        # works correctly exists in ToroidalFourierDistributionTest)
        kappa1 = 0.3
        kappa2 = 1.5
        lambda_ = 0.5
        coeffs = [5, 7]
        tvm = ToroidalVonMisesSineDistribution([1, 2], [kappa1, kappa2], lambda_)
        hfdId = HypertoroidalFourierDistribution.from_function(lambda x, y: np.reshape(tvm.pdf(np.array([x.flatten(), y.flatten()])), x.shape), coeffs, 'identity')
        hfdSqrt = HypertoroidalFourierDistribution.from_function(lambda x, y: np.reshape(tvm.pdf(np.array([x.flatten(), y.flatten()])), x.shape), coeffs, 'sqrt')

        tfdId = ToroidalFourierDistribution(hfdId.C, hfdId.transformation)
        tfdSqrt = ToroidalFourierDistribution(hfdSqrt.C, hfdSqrt.transformation)

        np.testing.assert_allclose(hfdId.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), tfdId.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), tfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdId.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), tfdId.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), tfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdId.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), tfdId.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), tfdSqrt.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), rtol=1e-4, atol=1e-4)

    def test_integral_2d(self):
        # Test against implementation in toroidal (test case that this
        # works correctly exists in ToroidalFourierDistributionTest)
        kappa1 = 0.3
        kappa2 = 1.5
        lambda_ = 0.5
        coeffs = [5, 7]
        tvm = ToroidalVonMisesSineDistribution([1, 2], [kappa1, kappa2], lambda_)
        hfdId = HypertoroidalFourierDistribution.from_function(lambda x, y: np.reshape(tvm.pdf(np.array([x.flatten(), y.flatten()])), x.shape), coeffs, 'identity')
        hfdSqrt = HypertoroidalFourierDistribution.from_function(lambda x, y: np.reshape(tvm.pdf(np.array([x.flatten(), y.flatten()])), x.shape), coeffs, 'sqrt')

        tfdId = ToroidalFourierDistribution(hfdId.C, hfdId.transformation)
        tfdSqrt = ToroidalFourierDistribution(hfdSqrt.C, hfdSqrt.transformation)

        np.testing.assert_allclose(hfdId.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), tfdId.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), tfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdId.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), tfdId.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), tfdSqrt.integrate(np.array([0, 0]), np.array([np.pi, 2 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdId.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), tfdId.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hfdSqrt.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), tfdSqrt.integrate(np.array([0, -1]), np.array([3 * np.pi, 5 * np.pi])), rtol=1e-4, atol=1e-4)
    
    def test_marginalize_2D_to_1D_tvm(self):
        tvm = ToroidalVonMisesSineDistribution([1, 2], [2, 3], 3)
        for transformation in ['identity', 'sqrt']:
            hfd = HypertoroidalFourierDistribution.from_distribution(tvm, [53, 53], transformation)
            for d in range(1, 3):
                vm = tvm.marginalize_to_1D(d)
                fd1 = hfd.marginalize_to_1D(d)
                fd2 = hfd.marginalize_out(1 + (d == 1))
                # Replace `testCase.verifyEqual` with an appropriate assertion function

    def test_marginalize_2D_to_1D_twn(self):
        twn = ToroidalWrappedNormalDistribution([3, 4], 2 * np.array([[1, 0.8], [0.8, 1]]))
        for transformation in ['identity', 'sqrt']:
            hfd = HypertoroidalFourierDistribution.from_distribution(twn, [71, 53], transformation)
            grid = np.linspace(-np.pi, 3 * np.pi, 300)
            pass #TODO

    def test_marginalize_3D_to_2D(self):
        twn = HypertoroidalWrappedNormalDistribution([3, 4, 6], 2 * np.array([[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 2]]))
        for transformation in ['identity', 'sqrt']:
            pass #TODO

    def test_marginalize_3D_to_1D(self):
        twn = HypertoroidalWrappedNormalDistribution([3, 4, 6], 2 * np.array([[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 2]]))
        for transformation in ['identity', 'sqrt']:
            hfd = HypertoroidalFourierDistribution.from_distribution(twn, [41, 53, 53], transformation)
            grid = np.linspace(-np.pi, 3 * np.pi, 100)
            pass #TODO

    def test_plotting(self):
        dist = ToroidalVonMisesSineDistribution([1, 2], [2, 3], 3)
        tfd = ToroidalFourierDistribution.from_distribution(dist, [51, 201], 'identity')
        pass #TODO

    @staticmethod
    def test_approx_from_even_grid_1D():
        hwn = WrappedNormalDistribution(1, 0.5)
        hgd = HypertoroidalGridDistribution.from_distribution(hwn, 4)
        hfdId = HypertoroidalFourierDistribution.from_function_values(
            hgd.grid_values.reshape(hgd.n_grid_points), 5, 'identity')
        np.testing.assert_allclose(hfdId.pdf(hgd.get_grid()), hgd.grid_values, atol=1E-15)

        hfdSqrt = HypertoroidalFourierDistribution.from_function_values(
            hgd.grid_values.reshape(hgd.n_grid_points), 5, 'sqrt')
        np.testing.assert_allclose(hfdSqrt.pdf(hgd.get_grid()), hgd.grid_values, atol=0.05)

    @staticmethod
    def test_approx_from_even_grid_2D():
        hwn = HypertoroidalWrappedNormalDistribution([1, 1], 0.1 * np.eye(2))
        hgd = HypertoroidalGridDistribution.from_distribution(hwn, [4, 4])

        hfdId = HypertoroidalFourierDistribution.from_function_values(
            hgd.grid_values.reshape(hgd.n_grid_points), [5, 5], 'identity')
        np.testing.assert_allclose(hfdId.pdf(hgd.get_grid()).T, hgd.grid_values, atol=1E-16)

        hfdSqrt = HypertoroidalFourierDistribution.from_function_values(
            hgd.grid_values.reshape(hgd.n_grid_points), [5, 5], 'sqrt')
        np.testing.assert_allclose(hfdSqrt.pdf(hgd.get_grid()).T, hgd.grid_values, atol=0.1)

    @staticmethod
    def test_match_function_odd_1D():
        fNeeds6 = lambda x: 1 / (2 * np.pi) + 0.05 * np.cos(x) + 0.05 * np.cos(2 * x) + 0.05 * np.cos(
            3 * x) + 0.05 * np.sin(x) - 0.05 * np.sin(2 * x)
        fNeeds7 = lambda x: 1 / (2 * np.pi) + 0.05 * np.cos(x) + 0.05 * np.cos(2 * x) + 0.05 * np.cos(
            3 * x) + 0.05 * np.sin(x) - 0.05 * np.sin(2 * x) + 0.05 * np.sin(3 * x)
        cdNeeds6 = CustomCircularDistribution(fNeeds6)
        cdNeeds7 = CustomCircularDistribution(fNeeds7)

        hgdNeeds6Has6 = HypertoroidalGridDistribution.from_distribution(cdNeeds6, 6)
        hgdNeeds6Has7 = HypertoroidalGridDistribution.from_distribution(cdNeeds6, 7)
        hgdNeeds7Has6 = HypertoroidalGridDistribution.from_distribution(cdNeeds7, 6)
        # TODO