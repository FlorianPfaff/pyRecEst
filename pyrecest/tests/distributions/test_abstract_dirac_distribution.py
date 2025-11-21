import unittest

import matplotlib

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import random


class TestAbstractDiracDistribution(unittest.TestCase):
    def _test_plot_helper(self, name, dist, dim, dirac_cls, **kwargs):
        if dirac_cls is None:
            return  # Prevent failure if no classes are set

        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")

        # Seed the random number generator for reproducibility
        random.seed(0)
        # Sample data and create LinearDiracDistribution instance
        # pylint: disable=not-callable
        ddist = dirac_cls(d=dist.sample(10), w=None, **kwargs)

        try:
            # Attempt to plot
            ddist.plot()
        except (ValueError, RuntimeError) as e:
            self.fail(f"{name}: Plotting failed for dimension {dim} with error: {e}")
