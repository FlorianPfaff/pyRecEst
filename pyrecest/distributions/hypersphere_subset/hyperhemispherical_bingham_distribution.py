from .bingham_distribution import BinghamDistribution
from .abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution


class HyperhemisphericalBinghamDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self, Z_, M_):
        self.distFullSphere = BinghamDistribution(Z_, M_)
        self.dim = self.distFullSphere.dim

    def pdf(self, xs):
        return 2 * self.distFullSphere.pdf(xs)

    def Z(self):
        return self.distFullSphere.Z

    def M(self):
        return self.distFullSphere.M

    def F(self):
        return self.distFullSphere.F

    def dF(self):
        return self.distFullSphere.dF

    def sample(self, n):
        sFull = self.distFullSphere.sample(n)
        s = sFull * (-1) ** (sFull[-1, :] < 0)  # Mirror to upper hemisphere
        return s

    def multiply(self, B2):
        B = HyperhemisphericalBinghamDistribution(self.Z(), self.M())
        B.distFullSphere = self.distFullSphere.multiply(B2.distFullSphere)
        return B

    def compose(self, B2):
        B = HyperhemisphericalBinghamDistribution(self.Z(), self.M())
        B.distFullSphere = self.distFullSphere.compose(B2.distFullSphere)
        return B

    def mode(self):
        return self.distFullSphere.mode()

    def mean_axis(self):
        return self.distFullSphere.mean_axis()
