from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
import unittest
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
import numpy.testing as npt
from pyrecest.backend import array, sin, random, fft, pi, linspace, sqrt, meshgrid, column_stack, sum
from pyrecest.distributions.circle.wrapped_normal_distribution import WrappedNormalDistribution as WNDistribution
import warnings
from parameterized import parameterized

def integrate2d(hfd, N=100):
    x = linspace(0, 2*pi, N, endpoint=False) + (pi/N)
    y = linspace(0, 2*pi, N, endpoint=False) + (pi/N)
    X, Y = meshgrid(x, y, indexing='ij')
    pts = column_stack((X.flatten(), Y.flatten()))
    p = hfd.pdf(pts)
    area = (2*pi/N)**2
    return sum(p)*area

def integrate3d(hfd, N=30):
    x = linspace(0, 2*pi, N, endpoint=False) + (pi/N)
    y = linspace(0, 2*pi, N, endpoint=False) + (pi/N)
    z = linspace(0, 2*pi, N, endpoint=False) + (pi/N)
    X, Y, Z = meshgrid(x, y, z, indexing='ij')
    pts = column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    p = hfd.pdf(pts)
    vol = (2*pi/N)**3
    return sum(p)*vol


class HypertoroidalFourierDistributionTest(unittest.TestCase):
    def test_normalization_warnings(self):
            with self.assertWarns(UserWarning):
                HypertoroidalFourierDistribution(array([[0,0,0],
                                                            [0,-1,0],
                                                            [0,0,0]]), 'identity')
            with self.assertRaises(ValueError):
                HypertoroidalFourierDistribution(array([[0,0,0],
                                                            [0,1e-201,0],
                                                            [0,0,0]]), 'identity')

    @parameterized.expand([
        ("2d_identity", "identity", (3, 7), (1, 3)),
        ("2d_sqrt", "sqrt", (3, 7), (1, 3)),
        ("3d_identity", "identity", (3, 11, 7), (1, 5, 3)),
        ("3d_sqrt", "sqrt", (3, 11, 7), (1, 5, 3)),
    ])
    def test_normalization_nd(self, _, transform, shape, index):
        random.seed(0)
        arr = random.rand(*shape) + 0.5
        unnormalizedCoeffs = fft.fftshift(fft.fftn(arr))
        unnormalizedCoeffs[index] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hfd = HypertoroidalFourierDistribution(unnormalizedCoeffs, transformation=transform)
        if hfd.dim == 2:
            val = integrate2d(hfd, N=30)
        elif hfd.dim == 3:
            val = integrate3d(hfd, N=30)
        self.assertAlmostEqual(val, 1.0, places=4)

    def test_constructor(self):
        # Create a 3x3 coefficient matrix
        coeffs = array([[0, 0, 0],
                           [0, 1/(sqrt(2*pi)**2), 0],
                           [0, 0, 0]])
        hfd = HypertoroidalFourierDistribution(coeffs, 'sqrt')
        self.assertEqual(hfd.coeff_mat.shape, (3,3))
        self.assertEqual(hfd.transformation, 'sqrt')

    def test_sine_function(self):
        # Create a sine function
        def sine_function(x, y):
            return sin(x)*sin(y)

        # Create a HypertoroidalFourierDistribution from the sine function
        coeffs = (3, 3)
        hfd = HypertoroidalFourierDistribution.from_function(sine_function, coeffs, 2)
        self.assertIsInstance(hfd, HypertoroidalFourierDistribution)
        self.assertEqual(hfd.coeff_mat.shape, coeffs)

    @parameterized.expand([
        ("identity", "identity"),
        ("sqrt", "sqrt"),
    ])
    def test_from_distribution_1d(self, _, transform):
        wn = WNDistribution(array(1.0), array(1.0))
        hfd = HypertoroidalFourierDistribution.from_distribution(wn, 101, transform)
        xvals = linspace(0, 2*pi, 100, endpoint=False)
        p1 = wn.pdf(xvals)
        p2 = hfd.pdf(xvals)
        npt.assert_allclose(p1, p2, atol=1e-8)

    def test_truncation_1d(self):
        wn = WNDistribution(array(1.0), array(1.0))
        hfd1 = HypertoroidalFourierDistribution.from_distribution(wn, 101)
        hfd2 = hfd1.truncate(51)
        self.assertEqual(hfd2.coeff_mat.shape, (51,))
        xvals = linspace(0, 2*pi, 100, endpoint=False)
        p1 = hfd1.pdf(xvals)
        p2 = hfd2.pdf(xvals)
        npt.assert_allclose(p1, p2, atol=1e-8)

    @parameterized.expand([
        ("identity", "identity"),
        ("sqrt", "sqrt"),
    ])
    def test_from_function_2d(self, _, transform):
        xTest, yTest = meshgrid(linspace(0, 2*pi, 10),
                                    linspace(0, 2*pi, 10),
                                    indexing='ij')
        coeffs = (13, 15)
        # Loop over some parameter combinations
        mu = array([1.0, 0.0])
        sigma1 = 1.0
        sigma2 = 2.0
        rho = 0.5
        dist = ToroidalWrappedNormalDistribution(mu, array([[sigma1, sigma1*sigma2*rho], [sigma1*sigma2*rho, sigma2]]))
        # Define a function f(x,y) that returns the pdf of tvm on a grid.
        def f_from_function(x, y):
            # x and y are arrays of the same shape.
            pts = column_stack((x.flatten(), y.flatten()))
            return dist.pdf(pts).flatten()
        
        hfd = HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 2, desired_transformation=transform)
        self.assertIsInstance(hfd, HypertoroidalFourierDistribution)
        self.assertEqual(hfd.coeff_mat.shape, coeffs)
        
        pts_test = column_stack((xTest.flatten(), yTest.flatten()))
        orig_pdf_vals = dist.pdf(pts_test)
        approx_pdf_vals = hfd.pdf(pts_test)
        npt.assert_allclose(orig_pdf_vals, approx_pdf_vals, atol=1e-5)
        with self.assertRaises(ValueError):
            HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 'abc')

if __name__ == '__main__':
    unittest.main()