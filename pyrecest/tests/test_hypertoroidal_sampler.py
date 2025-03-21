import unittest
from math import pi
import numpy.testing as npt
# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ones, diff, zeros, std
from pyrecest.sampling.hypertoroidal_sampler import CircularUniformSampler


class TestCircularUniformSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = CircularUniformSampler()

    def test_sample_stochastic_shape(self):
        n_samples = 1000
        samples = self.sampler.sample_stochastic(n_samples)

        # Check that the returned shape matches the requested number of samples
        npt.assert_equal(samples.shape[0], n_samples)

    def test_sample_stochastic_range(self):
        n_samples = 1000
        samples = self.sampler.sample_stochastic(n_samples)

        # Check that all samples are within the range [0, 2*pi)
        npt.assert_array_less(-samples, np.zeros(n_samples))  # samples >= 0.0
        npt.assert_array_less(samples, 2.0 * pi * np.ones(n_samples))  # samples < 2.0 * pi

    def test_get_grid_shape(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that the returned shape matches the grid density parameter
        npt.assert_equal(grid_points.shape[0], grid_density_parameter)

    def test_get_grid_range_and_equidistance(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that all grid points are within the range [0, 2*pi)
        npt.assert_array_less(-grid_points, zeros(grid_density_parameter))  # grid_points >= 0.0
        npt.assert_array_less(grid_points, 2.0 * pi * ones(grid_density_parameter))  # grid_points < 2.0 * pi

        # Check that the grid points are equidistant
        expected_diff = 2 * pi / grid_density_parameter
        actual_diffs = std(diff(grid_points))
        npt.assert_allclose(actual_diffs, expected_diff, atol=1e-6)


if __name__ == "main":
unittest.main()