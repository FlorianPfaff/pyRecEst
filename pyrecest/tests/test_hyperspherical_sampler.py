import importlib.util
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, linspace, pi, random
from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

from ..sampling.hyperspherical_sampler import (
    AbstractHopfBasedS3Sampler,
    DriscollHealySampler,
    FibonacciHopfSampler,
    HealpixHopfSampler,
    HealpixSampler,
    SphericalCoordinatesBasedFixedResolutionSampler,
    SphericalFibonacciSampler,
)

healpy_installed = importlib.util.find_spec("healpy") is not None


class TestHypersphericalGridGenerationFunction(unittest.TestCase):
    @parameterized.expand(
        [
            ("healpix", 2, 48, "n_side"),
            ("driscoll_healy", 2, 91, "l_max"),
            ("fibonacci", 12, 12, "n_samples"),
            ("spherical_fibonacci", 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_sphere(
        self, method, grid_density_parameter, grid_points_expected, desc_key
    ):
        samples, grid_specific_description = get_grid_hypersphere(
            method, grid_density_parameter
        )

        self.assertEqual(
            samples.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {samples.shape[0]}",
        )
        self.assertEqual(
            samples.shape[1],
            3,
            f"Expected 3-dimensional-output but got {samples.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_specific_description[desc_key], grid_density_parameter)

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        reason="Not supported on this backend",
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_hypersphere(self):
        samples, _ = get_grid_hypersphere("healpix_hopf", 0)

        self.assertEqual(
            samples.shape[0], 72, f"Expected {72} points but got {samples.shape[0]}"
        )
        self.assertEqual(
            samples.shape[1],
            4,
            f"Expected 4-dimensional-output but got {samples.shape[1]}-dimensional output",
        )


class TestHypersphericalSampler(unittest.TestCase):
    @parameterized.expand(
        [
            (HealpixSampler(), 2, 48, "n_side"),
            (DriscollHealySampler(), 2, 91, "l_max"),
            (SphericalFibonacciSampler(), 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_samplers(
        self, sampler, grid_density_parameter, grid_points_expected, desc_key
    ):
        grid, grid_description = sampler.get_grid(grid_density_parameter)

        self.assertEqual(
            grid.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            3,
            f"Expected 3-dimensional-output but got {grid.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_description[desc_key], grid_density_parameter)

    @parameterized.expand([(0, 72), (1, 648)])
    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        reason="Not supported on this backend",
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_healpix_hopf_sampler(self, input_value, expected_grid_points):
        sampler = HealpixHopfSampler()
        dim = 3
        grid, _ = sampler.get_grid(input_value)

        self.assertEqual(
            grid.shape[0],
            expected_grid_points,
            f"Expected {expected_grid_points} points but got {grid.shape[1]}",
        )
        self.assertEqual(
            grid.shape[1],
            dim + 1,
            f"Expected {dim+1}-dimensional-output but got {grid.shape[1]}-dimensional output",
        )

    def test_fibonacci_hopf_sampler(self):
        sampler = FibonacciHopfSampler()
        grid_density_parameter = [12, 4]
        grid, _ = sampler.get_grid(grid_density_parameter)

        expected_points = grid_density_parameter[0] * grid_density_parameter[1]
        self.assertEqual(
            grid.shape[0],
            expected_points,
            f"Expected {expected_points} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            4,
            f"Expected 4-dimensional-output but got {grid.shape[1]}-dimensional output",
        )


class TestSphericalCoordinatesBasedFixedResolutionSampler(unittest.TestCase):
    def test_get_grid_spherical_coordinates(self):
        # Create an instance of the sampler
        sampler = SphericalCoordinatesBasedFixedResolutionSampler()

        # Define the resolution parameters for latitude and longitude
        grid_density_parameter = array(
            [10, 20]
        )  # 10 latitude lines, 20 longitude lines

        # Call the method
        phi, theta, _ = sampler.get_grid_spherical_coordinates(grid_density_parameter)

        expected_phi = linspace(
            0.0, 2 * pi, num=grid_density_parameter[0], endpoint=False
        )
        expected_theta = linspace(
            pi / (grid_density_parameter[1] + 1),
            pi,
            num=grid_density_parameter[1],
            endpoint=False,
        )

        # Check if the first and last values of the generated phi and theta are as expected
        # This assumes that you have access to phi and theta which might not be the case here
        npt.assert_allclose(phi, expected_phi)
        npt.assert_allclose(theta, expected_theta)


class TestHopfConversion(unittest.TestCase):
    def test_conversion(self):
        # Generate a sample matrix of size (n, 4) containing unit vectors.
        n = 100  # sample size
        random_vectors = random.normal(size=(n, 4))
        unit_vectors = random_vectors / linalg.norm(random_vectors, axis=1)[:, None]

        # Pass the quaternions through the conversion functions
        θ, ϕ, ψ = AbstractHopfBasedS3Sampler.quaternion_to_hopf_yershova(unit_vectors)
        recovered_quaternions = (
            AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(θ, ϕ, ψ)
        )

        # Check if the original quaternions are close to the recovered quaternions.
        npt.assert_allclose(unit_vectors, recovered_quaternions, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
