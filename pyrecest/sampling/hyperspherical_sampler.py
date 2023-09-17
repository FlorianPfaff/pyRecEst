import itertools
from abc import abstractmethod

import numpy as np
from beartype import beartype
from pyrecest.distributions import (
    AbstractSphericalDistribution,
    HypersphericalUniformDistribution,
)

from .abstract_sampler import AbstractSampler
from .hypertoroidal_sampler import CircularUniformSampler


@beartype
def get_grid_hypersphere(method: str, grid_density_parameter: int):
    if method == "healpix":
        samples, grid_specific_description = HealpixSampler().get_grid(
            grid_density_parameter
        )
    elif method == "driscoll_healy":
        samples, grid_specific_description = DriscollHealySampler().get_grid(
            grid_density_parameter
        )
    elif method in ("fibonacci", "spherical_fibonacci"):
        samples, grid_specific_description = SphericalFibonacciSampler().get_grid(
            grid_density_parameter
        )
    elif method == "healpix_hopf":
        samples, grid_specific_description = HealpixHopfSampler().get_grid(
            grid_density_parameter
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return samples, grid_specific_description


get_grid_sphere = get_grid_hypersphere


class AbstractHypersphericalUniformSampler(AbstractSampler):
    @beartype
    def sample_stochastic(self, n_samples: int, dim: int) -> np.ndarray:
        return HypersphericalUniformDistribution(dim).sample(n_samples)

    @abstractmethod
    def get_grid(self, grid_density_parameter: int, dim: int):
        raise NotImplementedError()


class AbstractSphericalUniformSampler(AbstractHypersphericalUniformSampler):
    def sample_stochastic(
        self, n_samples: int, dim: int = 2
    ):  # Only having dim there for interface compatibility
        assert dim == 2
        return HypersphericalUniformDistribution(2).sample(n_samples)


class AbstractSphericalCoordinatesBasedSampler(AbstractSphericalUniformSampler):
    @abstractmethod
    def get_grid_spherical_coordinates(
        self, grid_density_parameter: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        raise NotImplementedError()

    @beartype
    def get_grid(self, grid_density_parameter: int) -> tuple[np.ndarray, dict]:
        phi, theta, grid_specific_description = self.get_grid_spherical_coordinates(
            grid_density_parameter
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        grid = np.column_stack((x, y, z))

        return grid, grid_specific_description


class HealpixSampler(AbstractHypersphericalUniformSampler):
    @beartype
    def get_grid(self, grid_density_parameter: int) -> tuple[np.ndarray, dict]:
        import healpy as hp

        n_side = grid_density_parameter
        n_areas = hp.nside2npix(n_side)
        x, y, z = hp.pix2vec(n_side, np.arange(n_areas))
        grid = np.column_stack((x, y, z))

        grid_specific_description = {
            "scheme": "healpix",
            "n_side": grid_density_parameter,
        }

        return grid, grid_specific_description


class DriscollHealySampler(AbstractSphericalCoordinatesBasedSampler):
    @beartype
    def get_grid_spherical_coordinates(
        self, grid_density_parameter: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        import pyshtools as pysh

        grid = pysh.SHGrid.from_zeros(grid_density_parameter)

        # Get the longitudes (phi) and latitudes (theta) directly from the grid
        phi_deg_mat = grid.lons()
        theta_deg_mat = grid.lats()

        phi_theta_stacked_deg = np.array(
            list(itertools.product(phi_deg_mat, theta_deg_mat))
        )
        phi_theta_stacked_rad = np.radians(phi_theta_stacked_deg)

        phi = phi_theta_stacked_rad[:, 0]
        theta = phi_theta_stacked_rad[:, 1]

        grid_specific_description = {
            "scheme": "driscoll_healy",
            "l_max": grid_density_parameter,
            "n_lat": grid.nlat,
            "n_lon": grid.nlon,
        }

        return phi, theta, grid_specific_description


class SphericalFibonacciSampler(AbstractSphericalCoordinatesBasedSampler):
    @beartype
    def get_grid_spherical_coordinates(
        self, grid_density_parameter: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        indices = np.arange(0, grid_density_parameter, dtype=float) + 0.5
        phi = np.pi * (1 + 5**0.5) * indices
        theta = np.arccos(1 - 2 * indices / grid_density_parameter)
        grid_specific_description = {
            "scheme": "spherical_fibonacci",
            "n_samples": grid_density_parameter,
        }
        return phi, theta, grid_specific_description


class AbstractHopfBasedS3Sampler(AbstractHypersphericalUniformSampler):
    @staticmethod
    @beartype
    def hopf_coordinates_to_quaterion_yershova(
        θ: np.ndarray, ϕ: np.ndarray, ψ: np.ndarray
    ):
        """
        One possible way to index the S3-sphere via the hopf fibration.
        Using the convention from
        "Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration"
        by
        Anna Yershova, Swati Jain, Steven M. LaValle, Julie C. Mitchell
        """
        quaterions = np.empty((θ.shape[0], 4))

        quaterions[:, 0] = np.cos(θ / 2) * np.cos(ψ / 2)
        quaterions[:, 1] = np.cos(θ / 2) * np.sin(ψ / 2)
        quaterions[:, 2] = np.sin(θ / 2) * np.cos(ϕ + ψ / 2)
        quaterions[:, 3] = np.sin(θ / 2) * np.sin(ϕ + ψ / 2)
        return quaterions

    @staticmethod
    @beartype
    def quaternion_to_hopf_yershova(q: np.ndarray):
        θ = 2 * np.arccos(np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2))
        ϕ = np.arctan2(q[:, 3], q[:, 2]) - np.arctan2(q[:, 1], q[:, 0])
        ψ = 2 * np.arctan2(q[:, 1], q[:, 0])
        return θ, ϕ, ψ


# pylint: disable=too-many-locals
class HealpixHopfSampler(AbstractHopfBasedS3Sampler):
    @beartype
    def get_grid(self, grid_density_parameter: int | list[int]):
        """
        Hopf coordinates are (θ, ϕ, ψ) where θ and ϕ are the angles for the sphere and ψ is the angle on the circle
        First parameter is the number of points on the sphere, second parameter is the number of points on the circle.
        """
        import healpy as hp

        if isinstance(grid_density_parameter, int):
            grid_density_parameter = [grid_density_parameter]

        s3_points_list = []

        for i in range(grid_density_parameter[0] + 1):
            if np.size(grid_density_parameter) == 2:
                n_sample_circle = grid_density_parameter[1]
            else:
                n_sample_circle = 2**i * 6

            psi_points = CircularUniformSampler().get_grid(n_sample_circle)

            assert np.size(psi_points) != 0

            nside = 2**i
            numpixels = hp.nside2npix(nside)

            healpix_points = np.empty((numpixels, 2))
            for j in range(numpixels):
                theta, phi = hp.pix2ang(nside, j, nest=True)
                healpix_points[j] = [theta, phi]

            for j in range(len(healpix_points)):
                for k in range(len(psi_points)):
                    temp = np.array(
                        [healpix_points[j, 0], healpix_points[j, 1], psi_points[k]]
                    )
                    s3_points_list.append(temp)

        s3_points = np.vstack(s3_points_list)  # Need to stack like this and unpack
        grid = AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(
            s3_points[:, 0], s3_points[:, 1], s3_points[:, 2]
        )

        grid_specific_description = {
            "scheme": "healpix_hopf",
            "layer-parameter": grid_density_parameter,
        }
        return grid, grid_specific_description
