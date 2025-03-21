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

class HypertoroidalFourierDistributionTest(unittest.TestCase):
    def test_normalization_2D(self):
        unnormalized_coeffs_2D = fft.fftshift(fft.fftn(random.rand(3, 7) + 0.5))
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
    