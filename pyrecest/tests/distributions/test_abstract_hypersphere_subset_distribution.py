import unittest
from math import gamma, pi

import numpy.testing as npt
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    cos,
    isclose,
    linalg,
    ones,
    prod,
    random,
    sin,
    sqrt,
    squeeze,
)
from pyrecest.distributions import (
    AbstractHypersphereSubsetDistribution,
    HypersphericalUniformDistribution,
    VonMisesFisherDistribution,
)
from scipy.integrate import nquad


class TestAbstractHypersphereSubsetDistribution(unittest.TestCase):
    @staticmethod
    def _hyperspherical_to_cartesian_2_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 2-sphere (embedded in R^3).
        """
        theta, phi = angles[:, 0], angles[:, 1]

        x = sin(theta) * sin(phi)
        y = sin(theta) * cos(phi)
        z = cos(theta)
        return column_stack([x, y, z])

    @staticmethod
    def _hyperspherical_to_cartesian_3_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 3-sphere (embedded in R^4).
        """
        theta1, theta2, theta3 = angles[:, 0], angles[:, 1], angles[:, 2]

        w = sin(theta1) * sin(theta2) * sin(theta3)
        x = sin(theta1) * sin(theta2) * cos(theta3)
        y = sin(theta1) * cos(theta2)
        z = cos(theta1)
        return column_stack([w, x, y, z])

    @staticmethod
    def _hyperspherical_to_cartesian_4_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 4-sphere (embedded in R^5).
        """
        theta1, theta2, theta3, theta4 = (
            angles[:, 0],
            angles[:, 1],
            angles[:, 2],
            angles[:, 3],
        )

        v = sin(theta1) * sin(theta2) * sin(theta3) * sin(theta4)
        w = sin(theta1) * sin(theta2) * sin(theta3) * cos(theta4)
        x = sin(theta1) * sin(theta2) * cos(theta3)
        y = sin(theta1) * cos(theta2)
        z = cos(theta1)
        return column_stack([v, w, x, y, z])

    @staticmethod
    def hyperspherical_to_cartesian(r, *angles):
        """
        Convert hyperspherical coordinates to Cartesian coordinates in n-dimensions.
        """
        sin_values = [sin(angle) for angle in angles]
        cos_values = [cos(angle) for angle in angles]

        coordinates = []
        for i in range(len(angles)):
            coord = r * prod(sin_values[:i]) * cos_values[i]
            coordinates.append(coord)

        last_coord = r * prod(sin_values)
        coordinates.append(last_coord)

        return tuple(coordinates)

    @staticmethod
    def uniform_on_sn(*coordinates):
        """
        Calculate the probability density for a point to be on the surface of an n-dimensional sphere (S^(n-1)).
        """
        n = len(coordinates)
        norm = sqrt(sum(coord**2 for coord in coordinates))

        if isclose(norm, 1):
            surface_area = 2 * pi ** (n / 2) / gamma(n / 2)
            return 1 / surface_area
        else:
            return 0

    @staticmethod
    def _hyperspherical_to_cartesian_s2(r, theta, phi):
        """x = r * sin(theta) * sin(phi)
        #y = r * sin(theta) * cos(phi)
        #z = r * cos(theta)
        #return x, y, z"""
        z = r * sin(theta) * sin(phi)
        y = r * sin(theta) * cos(phi)
        x = r * cos(theta)
        return x, y, z

    @staticmethod
    def _hyperspherical_to_cartesian_s3(r, chi, theta, phi):
        """#x = r * sin(chi) * sin(theta) * sin(phi)
        #y = r * sin(chi) * sin(theta) * cos(phi)
        #z = r * sin(chi) * cos(theta)
        #w = r * cos(chi)
        #return x, y, z, w"""
        w = r * sin(chi) * sin(theta) * sin(phi)
        z = r * sin(chi) * sin(theta) * cos(phi)
        y = r * sin(chi) * cos(theta)
        x = r * cos(chi)
        return x, y, z, w

    @staticmethod
    def _uniform_on_s2(x, y, z):
        norm = sqrt(x**2 + y**2 + z**2)
        return 1 / (4 * pi) if isclose(norm, 1) else 0

    @staticmethod
    def _uniform_on_s3(x, y, z, w):
        norm = sqrt(x**2 + y**2 + z**2 + w**2)
        return 1 / (2 * pi**2) if isclose(norm, 1) else 0

    @staticmethod
    def _uniform_on_sn(*coordinates):
        """
        Calculate the probability density for a point to be on the surface of an n-dimensional sphere (S^(n-1)).

        Parameters:
        - coordinates (float): Cartesian coordinates of the point (x1, x2, ..., xn).

        Returns:
        - float: Probability density if the point is on the surface of the unit n-sphere, otherwise 0.
        """
        n = len(coordinates)  # Dimension of the sphere
        norm = sqrt(sum(coord**2 for coord in coordinates))

        # Check if the point is on the surface of the unit n-sphere
        if isclose(norm, 1):
            # Calculate the surface area of the unit n-sphere
            surface_area = 2 * pi ** (n / 2) / gamma(n / 2)
            # Return the reciprocal of the surface area
            return 1 / surface_area
        else:
            return 0

    # Adjusted integrand function for spherical coordinates
    @staticmethod
    def _integrand_s2(phi, theta):
        (
            x,
            y,
            z,
        ) = TestAbstractHypersphereSubsetDistribution._hyperspherical_to_cartesian_s2(
            1, theta, phi
        )  # radius = 1 for the unit sphere
        return TestAbstractHypersphereSubsetDistribution._uniform_on_s2(x, y, z) * sin(
            theta
        )  # Jacobian factor for spherical coordinates

    # Adjusted integrand function for hyperspherical coordinates
    @staticmethod
    def _integrand_s3(phi, theta, chi):
        (
            x,
            y,
            z,
            w,
        ) = TestAbstractHypersphereSubsetDistribution._hyperspherical_to_cartesian_s3(
            1, chi, theta, phi
        )  # radius = 1 for the unit 3-sphere
        return (
            TestAbstractHypersphereSubsetDistribution._uniform_on_s3(x, y, z, w)
            * sin(chi) ** 2
            * sin(theta)
        )  # Jacobian factor for hyperspherical coordinates

    @parameterized.expand(
        [
            # 2D case
            (2, (pi / 6, pi / 4), _integrand_s2),
            # 3D case
            (3, (pi / 6, pi / 4, pi / 3), _integrand_s3),
        ]
    )
    def test_integrand_sn(self, dimension, angles, integrand_function):
        # Calculating values for given dimension
        value_s = integrand_function(*angles)
        uniform_s = HypersphericalUniformDistribution(dimension)
        value_sn = uniform_s._get_integrand_hypersph_fun(uniform_s.pdf)(*angles)

        # Testing with assert_almost_equal
        npt.assert_allclose(value_s, value_sn, atol=1e-6)

    @parameterized.expand(
        [
            # 2D case
            (_integrand_s2, [[0, 2 * pi], [0, pi]]),
            # 3D case
            (_integrand_s3, [[0, 2 * pi], [0, pi], [0, pi]]),
        ]
    )
    def test_integrate(self, integrand_method, bounds):
        """Test the locally defined integrands (testing the test class, not the actual distribution)"""
        # Perform the integration
        integral_result, _ = nquad(integrand_method, bounds)
        npt.assert_allclose(integral_result, 1, atol=0.001)

    @parameterized.expand(
        [
            # 2D case for HypersphericalUniformDistribution
            (HypersphericalUniformDistribution(2), [[0, 2 * pi], [0, pi]]),
            # 3D case for HypersphericalUniformDistribution
            (HypersphericalUniformDistribution(3), [[0, 2 * pi], [0, pi], [0, pi]]),
            # 2D case for VonMisesFisherDistribution
            (
                VonMisesFisherDistribution(ones(3) / sqrt(3), 1.0),
                [[0, 2 * pi], [0, pi]],
            ),
            # 3D case for VonMisesFisherDistribution
            (
                VonMisesFisherDistribution(ones(4) / sqrt(4), 1.0),
                [[0, 2 * pi], [0, pi], [0, pi]],
            ),
        ]
    )
    def test_integrate_general_formula(self, dist, bounds):
        """Test the integration with the actual distribution"""
        # Perform the integration
        integral_result, _ = nquad(dist._get_integrand_hypersph_fun(dist.pdf), bounds)
        npt.assert_allclose(integral_result, 1, atol=0.001)

    # Now we test these explicit implementations against the provided generic function
    @parameterized.expand(
        [
            (2, _hyperspherical_to_cartesian_s2, 1),
            (3, _hyperspherical_to_cartesian_s3, 1),
            (2, _hyperspherical_to_cartesian_s2, 10),
            (3, _hyperspherical_to_cartesian_s3, 10),
        ]
    )
    def test_hyperspherical_to_cartesian_specific(
        self, dimensions, specific_function, n_samples: int
    ):
        # Test for single elements
        for _ in range(10):
            size_samples = (n_samples, dimensions) if n_samples > 1 else dimensions
            angles = 2.0 * pi * random.uniform(size=size_samples)
            cartesian_specific = squeeze(column_stack(specific_function(1, *angles.T)))
            cartesian_given = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
                angles, mode="colatitude"
            )
            npt.assert_allclose(cartesian_specific, cartesian_given)

    @parameterized.expand(
        [
            ("colatitude",),
            ("elevation",),
            ("inclination",),
        ]
    )
    def test_cart_to_sph_to_cart(self, mode):
        # Create some Cartesian coordinates
        x = array([1.0, 0.0, 0.0])
        y = array([0.0, 1.0, 0.0])
        z = array([0.0, 0.0, 1.0])

        # Convert to spherical coordinates and back
        angles = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
            column_stack((x, y, z)), mode=mode
        )
        cart_res = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles, mode=mode
        )

        # The new Cartesian coordinates should be close to the original ones
        npt.assert_allclose(cart_res[:, 0], x, atol=1e-7)
        npt.assert_allclose(cart_res[:, 1], y, atol=1e-7)
        npt.assert_allclose(cart_res[:, 2], z, atol=1e-7)

    def test_pdf_hyperspherical_coords_1d(self):
        mu_ = array([0.5, 1.0]) / linalg.norm(array([0.5, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_1d(phi):
            return vmf.pdf(array([sin(phi), cos(phi)]).T)

        phi_test = array([1.0, 2.0, 0.0, 0.3, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi_test), fangles_1d(phi_test)
        )

    def test_pdf_hyperspherical_coords_2d(self):
        mu_ = array([0.5, 1.0, 1.0]) / linalg.norm(array([0.5, 1.0, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_2d(phi1, phi2):
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2),
                        r * cos(phi1) * sin(phi2),
                        r * cos(phi2),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test), fangles_2d(phi1_test, phi2_test)
        )

    def test_pdf_hyperspherical_coords_3d(self):
        mu_ = array([0.5, 1.0, 1.0, -0.5]) / linalg.norm(array([0.5, 1.0, 1.0, -0.5]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_3d(phi1, phi2, phi3):
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi2) * sin(phi3),
                        r * cos(phi3),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])
        phi3_test = phi2_test + 0.2

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test, phi3_test),
            fangles_3d(phi1_test, phi2_test, phi3_test),
        )


if __name__ == "__main__":
    unittest.main()
