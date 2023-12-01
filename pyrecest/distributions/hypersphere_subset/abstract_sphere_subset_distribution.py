from math import pi

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arccos,
    arctan2,
    atleast_2d,
    column_stack,
    cos,
    linspace,
    meshgrid,
    ndim,
    sin,
    where,
)

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractSphereSubsetDistribution(AbstractHypersphereSubsetDistribution):
    """
    This is an abstract class for a distribution over a sphere.
    """

    def __init__(self):
        """
        Initialize the AbstractSphereSubsetDistribution instance.
        """
        super().__init__(2)

    @staticmethod
    def sph_to_cart(angles1, angles2, mode="inclination") -> tuple:
        """
        Convert spherical coordinates to Cartesian coordinates.
        For different convetions, see https://en.wikipedia.org/wiki/Spherical_coordinate_system
        Supported convetions:
        inclination and azimuth (θ_inc, φ_az,right)
        azimuth, and colatitude
        Refer to the respective functions for more information.
        """
        assert ndim(angles1) == 1 and ndim(angles2) == 1, "Inputs must be 1-dimensional"
        if mode == "inclination":
            # This follows the (θ_inc, φ_az,right) convention
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_inclination(
                angles1, angles2
            )
        elif mode == "elevation":
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_elevation(
                angles1, angles2
            )
        elif mode == "colatitude":
            coords = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
                column_stack((angles1, angles2)), mode=mode
            )
            coords = atleast_2d(coords)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        else:
            raise ValueError("Mode must be either 'colatitude' or 'elevation'")

        return x, y, z

    @staticmethod
    def cart_to_sph(x, y, z, mode="inclineation") -> tuple:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            x (): X coordinates.
            y (): Y coordinates.
            z (): Z coordinates.

        Returns:
            tuple: Spherical coordinates.
        """
        assert ndim(x) == 1 and ndim(y) == 1 and ndim(z), "Inputs must be 1-dimensional"
        if mode == "inclination":
            phi, theta = AbstractSphereSubsetDistribution._cart_to_sph_inclination(
                x, y, z
            )
        elif mode == "elevation":
            phi, theta = AbstractSphereSubsetDistribution._cart_to_sph_elevation(
                x, y, z
            )
        elif mode == "colatitude":
            angles = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
                column_stack((x, y, z)), mode=mode
            )
            phi, theta = angles[:, 0], angles[:, 1]
        else:
            raise ValueError(
                "Mode must be either 'inclination', 'colatitude', or 'elevation'"
            )

        return phi, theta

    @staticmethod
    def _sph_to_cart_inclination(theta_inc, phi_az_right) -> tuple:
        # Conversion for the (r, θ_inc, φ_az,right) convention. Used by many textbooks.
        # Downside is that it does not generalize well to higher dimensions.
        assert ndim(theta_inc) == 1 and ndim(
            phi_az_right
        ), "Inputs must be 1-dimensional"
        x = sin(phi_az_right) * cos(theta_inc)
        y = sin(phi_az_right) * sin(theta_inc)
        z = cos(phi_az_right)
        return x, y, z

    @staticmethod
    def _sph_to_cart_elevation(azimuth, elevation, map_to_inclination=False) -> tuple:
        """
        Convert spherical coordinates (using elevation) to Cartesian coordinates.
        Assumes a radius of 1.

        Args:
            azimuth (): Azimuth angles.
            elevation (): Elevation angles.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert (
            ndim(azimuth) == 1 and ndim(elevation) == 1
        ), "Inputs must be 1-dimensional"
        # elevation is π/2 - colatitude, so we calculate colatitude from elevation
        if map_to_inclination:
            inclination = pi / 2 - elevation
            return AbstractSphereSubsetDistribution._sph_to_cart_inclination(
                inclination, elevation
            )

        x = cos(elevation) * cos(azimuth)
        y = cos(elevation) * sin(azimuth)
        z = sin(elevation)
        return x, y, z

    @staticmethod
    def plot_unit_sphere():
        # Define the number of points to generate around the circle
        num_points = 1000

        # Generate theta and phi angles (in radians)
        theta = linspace(0, 2 * pi, num_points)
        phi = linspace(0, pi, num_points)

        # Create a meshgrid for theta and phi angles
        theta, phi = meshgrid(theta, phi)

        # Calculate the x, y, and z coordinates
        x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_inclination(theta, phi)

        # Plot the unit circle in 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, color="c", alpha=0.7)

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("Unit Circle in 3D Space")

        plt.show()

    @staticmethod
    def _cart_to_sph_inclination(x, y, z) -> tuple:
        assert ndim(x) == 1 and ndim(y) == 1 and ndim(z)
        azimuth = arctan2(y, x)
        azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)
        colatitude = arccos(z)
        return azimuth, colatitude

    @staticmethod
    def _cart_to_sph_elevation(x, y, z) -> tuple:
        assert ndim(x) == 1 and ndim(y) == 1 and ndim(z) == 1
        azimuth = arctan2(y, x)
        azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)
        elevation = pi / 2 - arccos(z)  # elevation is π/2 - colatitude
        return azimuth, elevation
