import unittest
import numpy as np
from pyrecest.distributions.circle.wrapped_cauchy_distribution import WrappedCauchyDistribution
from pyrecest.distributions.circle.custom_circular_distribution import CustomCircularDistribution

class WrappedCauchyDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = 0
        self.gamma = 0.5
        self.xs = np.arange(7)
        self.dist = WrappedCauchyDistribution(self.mu, self.gamma)

    def test_pdf(self):
        
        def pdf_wrapped(x, mu, gamma, terms=2000):
            summation = 0
            for k in range(-terms, terms+1):
                summation += gamma / (np.pi * (gamma**2 + (x - mu + 2 * np.pi * k)**2))
            return summation


        sinh_fun = lambda xs: 1 / (2 * np.pi) * np.sinh(self.gamma) / (np.cosh(self.gamma) - np.cos(xs - self.mu))
                
        custom_with_formular = CustomCircularDistribution(sinh_fun)
        custom_wrapped = CustomCircularDistribution(lambda xs: np.array([pdf_wrapped(x, self.mu, self.gamma) for x in xs]))
        
        np.testing.assert_allclose(custom_with_formular.pdf(xs=self.xs), custom_wrapped.pdf(self.xs), atol=0.0001)
        np.testing.assert_allclose(custom_wrapped.pdf(xs=self.xs), custom_wrapped.pdf(self.xs))


    def test_cdf(self):
        np.testing.assert_allclose(self.dist.cdf(np.array([1])), self.dist.integrate(np.array([0, 1])))

if __name__ == '__main__':
    unittest.main()