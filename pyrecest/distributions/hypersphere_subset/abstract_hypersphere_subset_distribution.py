from abc import abstractmethod
from collections.abc import Callable
from math import pi
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    array,
    atleast_2d,
    column_stack,
    cos,
    empty,
    float64,
    full,
    int32,
    int64,
    linalg,
    log,
    sin,
    sort,
    sqrt,
    squeeze,
    zeros,
)
from scipy.integrate import nquad, quad
from scipy.special import gamma

from ..abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


class AbstractHypersphereSubsetDistribution(AbstractBoundedDomainDistribution):
    @property
    def input_dim(self) -> int:
        return self.dim + 1

    def mean_direction(self):
        return self.mean_direction_numerical()

    @staticmethod
    @abstractmethod
    def get_full_integration_boundaries(dim: Union[int, int32, int64]):
        pass

    def mean_direction_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        mu = empty(self.dim + 1)

        if 1 <= self.dim <= 3:
            for i in range(self.dim + 1):

                def f(x, i=i):
                    return x[i] * self.pdf(x)

                # pylint: disable=cell-var-from-loop
                fangles = self.gen_fun_hyperspherical_coords(f, self.dim)

                # Casts the floats to arrays, relevant for operations on torch.tensors
                # that are not backward compatible
                def fangles_array(*args):
                    tensors = [array([arg], dtype=float64) for arg in args]
                    result = fangles(*tensors)
                    return result.item()

                if self.dim == 1:
                    mu[i], _ = quad(
                        fangles_array,
                        integration_boundaries[0, 0],
                        integration_boundaries[0, 1],
                        epsabs=1e-3,
                        epsrel=1e-3,
                    )
                elif self.dim == 2:
                    mu[i], _ = nquad(
                        fangles_array,
                        integration_boundaries,
                        opts={"epsabs": 1e-3, "epsrel": 1e-3},
                    )
                elif self.dim == 3:
                    mu[i], _ = nquad(
                        fangles_array,
                        integration_boundaries,
                        opts={"epsabs": 1e-3, "epsrel": 1e-3},
                    )
        else:
            raise ValueError("Unsupported")

        if linalg.norm(mu) < 1e-9:
            print(
                "Warning: Density may not actually have a mean direction because integral yields a point very close to the origin."
            )

        mu = mu / linalg.norm(mu)
        return mu

    def gen_pdf_hyperspherical_coords(self):
        """
        Generate the PDF in hyperspherical coordinates.

        :return: A function that computes the PDF value at given angles.
        """
        return AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
            self.pdf, self.dim
        )

    @staticmethod
    def gen_fun_hyperspherical_coords(f: Callable, dim: Union[int, int32, int64]):
        def generate_input(angles):
            dim_eucl = dim + 1
            angles = column_stack(angles)
            input_arr = zeros((angles.shape[0], dim_eucl))
            # Start at last, which is just cos
            input_arr[:, -1] = cos(angles[:, -1])
            sin_product = sin(angles[:, -1])
            # Now, iterate over all from end to back and accumulate the sines
            for i in range(2, dim_eucl):
                # All except the final one have a cos factor as their last one
                input_arr[:, -i] = sin_product * cos(angles[:, -i])
                sin_product *= sin(angles[:, -i])
            # The last one is all sines
            input_arr[:, 0] = sin_product
            return squeeze(input_arr)

        def fangles(*angles):
            input_arr = generate_input(angles)
            return f(input_arr)

        return fangles

    def moment(self):
        return self.moment_numerical()

    def moment_numerical(self):
        m = full(
            (
                self.dim + 1,
                self.dim + 1,
            ),
            float("NaN"),
        )

        def f_gen(i, j):
            def f(points):
                return self.pdf(points) * points[i] * points[j]

            return f

        def g_gen(f_hypersph_coords, dim):
            if dim == 1:

                def g_1d(phi):
                    return f_hypersph_coords(array(phi))

                return g_1d
            if dim == 2:

                def g_2d(phi1, phi2):
                    return f_hypersph_coords(array(phi1), array(phi2)) * sin(phi2)

                return g_2d
            if dim == 3:

                def g_3d(phi1, phi2, phi3):
                    return (
                        f_hypersph_coords(array(phi1), array(phi2), array(phi3))
                        * sin(phi2)
                        * sin(phi3) ** 2
                    )

                return g_3d

            raise ValueError("Dimension not supported.")

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                f_curr = f_gen(i, j)
                fangles = self.__class__.gen_fun_hyperspherical_coords(f_curr, self.dim)
                g_curr = g_gen(fangles, self.dim)
                m[i, j] = self.__class__.integrate_fun_over_domain(g_curr, self.dim)

        return m

    @staticmethod
    def _compute_mean_axis_from_moment(moment_matrix):
        D, V = linalg.eig(moment_matrix)
        Dsorted = sort(D)
        Vsorted = V[:, D.argsort()]
        if abs(Dsorted[-1] / Dsorted[-2]) < 1.01:
            print("Eigenvalues are very similar. Axis may be unreliable.")
        if Vsorted[-1, -1] >= 0:
            m = Vsorted[:, -1]
        else:
            m = -Vsorted[:, -1]
        return m

    def mean_axis(self):
        mom = self.moment()
        return AbstractHypersphereSubsetDistribution._compute_mean_axis_from_moment(mom)

    def mean_axis_numerical(self):
        mom = self.moment_numerical()
        return AbstractHypersphereSubsetDistribution._compute_mean_axis_from_moment(mom)

    def integrate(self, integration_boundaries):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        return self.integrate_numerically(integration_boundaries)

    @staticmethod
    @abstractmethod
    def integrate_fun_over_domain(
        f_hypersph_coords: Callable, dim: Union[int, int32, int64]
    ):
        # Overwrite with a function that specifies the integration_boundaries for the type of HypersphereSubsetDistribution
        pass

    @staticmethod
    def integrate_fun_over_domain_part(
        f_hypersph_coords: Callable,
        integration_boundaries,
    ):
        dim = integration_boundaries.shape[1]
        if dim == 1:
            
            def g(phi):  # type: ignore
                return f_hypersph_coords(array(phi))
        elif dim == 2:

            def g(phi1, phi2):  # type: ignore
            def g(phi1, phi2):  # type: ignore
                return f_hypersph_coords(array(phi1), array(phi2)) * sin(phi2)

        elif dim == 3:

            def g(phi1, phi2, phi3):  # type: ignore
            def g(phi1, phi2, phi3):  # type: ignore
                return (
                    f_hypersph_coords(array(phi1), array(phi2), array(phi3))
                    * sin(phi2)
                    * (sin(phi3)) ** 2
                )

        else:
            raise ValueError("Dimension not supported.")
        
        i, _ = nquad(g, integration_boundaries)

        i, _ = nquad(g, integration_boundaries)

        return i

    def integrate_numerically(self, integration_boundaries=None):
        """integration_boundaries have to be given in (hyper)spherical coordinates"""
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        f = self.gen_pdf_hyperspherical_coords()
        return AbstractHypersphereSubsetDistribution.integrate_fun_over_domain_part(
            f, integration_boundaries
        )

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        raise NotImplementedError("Method is not implemented yet.")

    def entropy_numerical(self):
        def entropy_f_gen():
            def f(points):
                return self.pdf(points) * log(self.pdf(points))

            return f

        f_entropy = entropy_f_gen()
        fangles_entropy = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_entropy, self.dim
            )
        )

        entropy_integral = self.__class__.integrate_fun_over_domain(
            fangles_entropy, self.dim
        )

        return -entropy_integral

    def _distance_f_gen(self, other, distance_func):
        def f(points):
            return distance_func(self.pdf(points), other.pdf(points))

        return f

    def hellinger_distance_numerical(self, other, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def hellinger_distance(pdf1, pdf2):
            return (sqrt(pdf1) - sqrt(pdf2)) ** 2

        f_hellinger = self._distance_f_gen(other, hellinger_distance)
        fangles_hellinger = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_hellinger, self.dim
            )
        )

        distance_integral = self.__class__.integrate_fun_over_domain(
            fangles_hellinger, self.dim
        )

        return 0.5 * distance_integral

    def total_variation_distance_numerical(self, other, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def total_variation_distance(pdf1, pdf2):
            return abs(pdf1 - pdf2)

        f_total_variation = self._distance_f_gen(other, total_variation_distance)
        fangles_total_variation = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_total_variation, self.dim
            )
        )

        distance_integral = self.__class__.integrate_fun_over_domain(
            fangles_total_variation, self.dim
        )

        return 0.5 * distance_integral

    @staticmethod
    def hypersph_to_cart(hypersph_coords):
        hypersph_coords = atleast_2d(hypersph_coords)

        coords = zeros(
            (
                hypersph_coords.shape[0],
                hypersph_coords.shape[1] + 1,
            )
        )
        coords[:, 0] = sin(hypersph_coords[:, 0]) * cos(hypersph_coords[:, 1])
        coords[:, 1] = sin(hypersph_coords[:, 0]) * sin(hypersph_coords[:, 1])
        coords[:, 2] = cos(hypersph_coords[:, 0])
        for i in range(2, hypersph_coords.shape[1]):
            coords[:, :-i] *= sin(hypersph_coords[:, i])  # noqa: E203
            coords[:, -i] = cos(hypersph_coords[:, i])
        return squeeze(coords)

    @staticmethod
    def compute_unit_hypersphere_surface(dim: Union[int, int32, int64]) -> float:
        if dim == 1:
            surface_area = 2.0 * pi
        elif dim == 2:
            surface_area = 4.0 * pi
        elif dim == 3:
            surface_area = 2.0 * pi**2
        else:
            surface_area = 2.0 * pi ** ((dim + 1) / 2) / gamma((dim + 1) / 2)
        return surface_area
