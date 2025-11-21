# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    atleast_2d,
    complex128,
    full_like,
    imag,
    isreal,
    real,
    sqrt,
    zeros,
)

# pylint: disable=E0611
from scipy.special import sph_harm_y

from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_spherical_harmonics_distribution import (
    AbstractSphericalHarmonicsDistribution,
)


class SphericalHarmonicsDistributionReal(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation="identity"):
        if not all(isreal(coeff_mat)):
            raise ValueError("Coefficients must be real")
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)

    @staticmethod
    def real_spherical_harmonic_basis_function(n, m, theta, phi):
        y_lm = sph_harm_y(n, m, theta, phi)

        if m < 0:
            y_nm_real = -sqrt(2.0) * imag(y_lm)
        elif m == 0:
            y_nm_real = real(y_lm)
        else:
            y_nm_real = (-1) ** m * sqrt(2.0) * real(y_lm)

        return y_nm_real

    def value(self, xs):
        xs = atleast_2d(xs)
        vals = zeros(xs.shape[0])
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            xs[:, 0], xs[:, 1], xs[:, 2]
        )

        for n_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-n_curr, n_curr + 1):
                # Evaluate it for all query points at once
                y_lm_real = SphericalHarmonicsDistributionReal.real_spherical_harmonic_basis_function(
                    n_curr, m_curr, theta, phi
                )
                vals += self.coeff_mat[n_curr, n_curr + m_curr] * y_lm_real

        return vals

    def to_spherical_harmonics_distribution_complex(self):
        from .spherical_harmonics_distribution_complex import (
            SphericalHarmonicsDistributionComplex,
        )

        if self.transformation != "identity":
            raise NotImplementedError("Transformation currently not supported")

        real_coeff_mat = self.coeff_mat
        complex_coeff_mat = full_like(real_coeff_mat, float("NaN"), dtype=complex128)

        for n in range(real_coeff_mat.shape[0]):
            for m in range(-n, n + 1):
                if m < 0:
                    complex_coeff_mat[n, n + m] = (
                        1j * real_coeff_mat[n, n + m] + real_coeff_mat[n, n - m]
                    ) / sqrt(2.0)
                elif m > 0:
                    complex_coeff_mat[n, n + m] = (
                        (-1) ** m
                        * (-1j * real_coeff_mat[n, n - m] + real_coeff_mat[n, n + m])
                        / sqrt(2.0)
                    )
                else:  # m == 0
                    complex_coeff_mat[n, n] = real_coeff_mat[n, n]

        return SphericalHarmonicsDistributionComplex(
            complex_coeff_mat, self.transformation
        )
