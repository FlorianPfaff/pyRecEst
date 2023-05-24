""" Abstract base class for all filters for Euclidean domains """
from .abstract_manifold_specific_filter import AbstractManifoldSpecificFilter


class AbstractEuclideanFilter(AbstractManifoldSpecificFilter):
    """Abstract base class for all filters for Euclidean domains"""

    def get_point_estimate(self):
        est = self.get_estimate().mean()
        return est
