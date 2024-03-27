import numpy as np
from scipy.stats import vonmises


class GeneralizedKSineSkewedDistribution:
    def __init__(self, mu, kappa, lambda_, m):
        self.mu = np.mod(mu, 2 * np.pi)
        self.kappa = kappa
        self.lambda_ = lambda_
        self.m = m

        self.validate_parameters()

    def validate_parameters(self):
        assert -1.0 <= self.lambda_ and self.lambda_ <= 1.0
        assert isinstance(self.m, int) and self.m >= 1

    def pdf(self, xa):
        if self.m == 1:
            # Evaluate the von Mises distribution and multiply by (1 + lambda_ * sin(xa - mu))
            vm_pdf = vonmises.pdf(xa, self.kappa, loc=self.mu)
            adjustment = (1 + self.lambda_ * np.sin(xa - self.mu))
            return vm_pdf * adjustment
        else:
            raise NotImplementedError("This m is not yet supported")

    def shift(self, angle):
        if not np.isscalar(angle):
            raise ValueError("angle must be a scalar")
        new_dist = GeneralizedKSineSkewedDistribution(self.mu + angle, self.lambda_, self.m)
        return new_dist
