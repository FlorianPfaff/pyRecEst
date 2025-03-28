import numpy as np
from scipy.integrate import nquad
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import HypertoroidalWrappedNormalDistribution
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import ToroidalVonMisesSineDistribution
import unittest
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
from pyrecest.distributions.circle.custom_circular_distribution import CustomCircularDistribution
import numpy.testing as npt
from pyrecest.backend import array, random, fft, pi
from pyrecest.distributions.circle.wrapped_normal_distribution import WrappedNormalDistribution as WNDistribution
import warnings

def integrate2d(hfd, N=100):
    x = np.linspace(0, 2*np.pi, N, endpoint=False) + (np.pi/N)
    y = np.linspace(0, 2*np.pi, N, endpoint=False) + (np.pi/N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pts = np.column_stack((X.flatten(), Y.flatten()))
    p = hfd.pdf(pts)
    area = (2*np.pi/N)**2
    return np.sum(p)*area

def integrate3d(hfd, N=30):
    x = np.linspace(0, 2*np.pi, N, endpoint=False) + (np.pi/N)
    y = np.linspace(0, 2*np.pi, N, endpoint=False) + (np.pi/N)
    z = np.linspace(0, 2*np.pi, N, endpoint=False) + (np.pi/N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    p = hfd.pdf(pts)
    vol = (2*np.pi/N)**3
    return np.sum(p)*vol


class HypertoroidalFourierDistributionTest(unittest.TestCase):
    def test_normalization_2d(self):
        np.random.seed(0)
        # Generate unnormalized coefficients for 2D:
        arr = np.random.rand(3, 7) + 0.5
        unnormalizedCoeffs2D = np.fft.fftshift(np.fft.fftn(arr))
        unnormalizedCoeffs2D[1, 3] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hfdId = HypertoroidalFourierDistribution(unnormalizedCoeffs2D, 'identity')
            hfdSqrt = HypertoroidalFourierDistribution(unnormalizedCoeffs2D, 'sqrt')
        val_id = integrate2d(hfdId, N=100)
        val_sqrt = integrate2d(hfdSqrt, N=100)
        self.assertAlmostEqual(val_id, 1.0, places=4)
        self.assertAlmostEqual(val_sqrt, 1.0, places=4)
        # Test warnings for negative values:
        with self.assertWarns(UserWarning):
            HypertoroidalFourierDistribution(np.array([[0,0,0],
                                                        [0,-1,0],
                                                        [0,0,0]]), 'identity')
        with self.assertRaises(ValueError):
            HypertoroidalFourierDistribution(np.array([[0,0,0],
                                                        [0,1e-201,0],
                                                        [0,0,0]]), 'identity')
    

    def test_normalization_3d(self):
        enableExpensive = True
        if enableExpensive:
            np.random.seed(0)
            arr = np.random.rand(3, 11, 7) + 0.5
            unnormalizedCoeffs3D = np.fft.fftshift(np.fft.fftn(arr))
            unnormalizedCoeffs3D[1, 5, 3] = 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                hfdId = HypertoroidalFourierDistribution(unnormalizedCoeffs3D, 'identity')
                hfdSqrt = HypertoroidalFourierDistribution(unnormalizedCoeffs3D, 'sqrt')
            val_id = integrate3d(hfdId, N=30)
            val_sqrt = integrate3d(hfdSqrt, N=30)
            self.assertAlmostEqual(val_id, 1.0, places=4)
            self.assertAlmostEqual(val_sqrt, 1.0, places=4)

    def test_constructor(self):
        # Create a 3x3 coefficient matrix
        coeffs = np.array([[0, 0, 0],
                           [0, 1/(np.sqrt(2*np.pi)**2), 0],
                           [0, 0, 0]])
        hfd = HypertoroidalFourierDistribution(coeffs, 'sqrt')
        self.assertEqual(hfd.coeff_mat.shape, (3,3))
        self.assertEqual(hfd.transformation, 'sqrt')
            
    def test_truncation_1d(self):
        # Create a dummy 1D white-noise distribution.
        wn = WNDistribution(array(1.0), array(1.0))
        hfd1 = HypertoroidalFourierDistribution.from_distribution(wn, 101)
        hfd2 = hfd1.truncate(51)
        self.assertEqual(hfd2.coeff_mat.shape, (51,))
        xvals = np.linspace(0, 2*np.pi, 100, endpoint=False).reshape(-1, 1)
        p1 = hfd1.pdf(xvals)
        p2 = hfd2.pdf(xvals)
        npt.assert_allclose(p1, p2, atol=1e-8)
        
    def test_from_function_2d(self):
        xTest, yTest = np.meshgrid(np.linspace(0, 2*np.pi, 10),
                                    np.linspace(0, 2*np.pi, 10),
                                    indexing='ij')
        coeffs = (13, 15)
        mu = np.array([1.0, 0.0])
        # Loop over some parameter combinations
        mu = np.array([1.0, 0.0])
        for sigma1 in [0.2, 1]:
            for sigma2 in [0.5, 2]:
                for rho in [0, 0.5]:
                    dist = ToroidalWrappedNormalDistribution(mu, array([[sigma1, sigma1*sigma2*rho], [sigma1*sigma2*rho, sigma2]]))
                    # Define a function f(x,y) that returns the pdf of tvm on a grid.
                    def f_from_function(x, y):
                        # x and y are arrays of the same shape.
                        pts = np.column_stack((x.flatten(), y.flatten()))
                        return dist.pdf(pts).reshape(x.shape)
                    
                    hfdId = HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 2, 'identity')
                    hfdSqrt = HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 2, 'sqrt')
                    hfdLog = HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 2, 'log')
                    self.assertIsInstance(hfdId, HypertoroidalFourierDistribution)
                    self.assertIsInstance(hfdSqrt, HypertoroidalFourierDistribution)
                    self.assertEqual(hfdId.coeff_mat.shape, coeffs)
                    self.assertEqual(hfdSqrt.coeff_mat.shape, coeffs)
                    
                    pts_test = np.column_stack((xTest.flatten(), yTest.flatten()))
                    tvm_pdf_vals = dist.pdf(pts_test)
                    id_pdf_vals = hfdId.pdf(pts_test)
                    sqrt_pdf_vals = hfdSqrt.pdf(pts_test)
                    npt.assert_allclose(tvm_pdf_vals, id_pdf_vals, atol=1e-5)
                    npt.assert_allclose(tvm_pdf_vals, sqrt_pdf_vals, atol=1e-6)
        with self.assertRaises(ValueError):
            HypertoroidalFourierDistribution.from_function(f_from_function, coeffs, 'abc')

if __name__ == '__main__':
    unittest.main()