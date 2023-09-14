from abc import abstractmethod

import numpy as np
import scipy.integrate
import scipy.optimize
from beartype import beartype
from scipy.integrate import nquad, dblquad

from ..hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractHypercylindricalDistribution(AbstractLinPeriodicCartProdDistribution):
    def __init__(
        self, bound_dim: int | np.int32 | np.int64, lin_dim: int | np.int32 | np.int64
    ):
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)

    @abstractmethod
    def pdf(self, xs):
        pass

    def integrate(self, integration_boundaries=None):
        return self.integrate_numerically(integration_boundaries)

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.get_reasonable_integration_boundaries()

        def f(*args):
            return self.pdf(np.array(args))

        integration_result = nquad(f, integration_boundaries.T)[0]

        return integration_result

    @beartype
    def get_reasonable_integration_boundaries(self, scalingFactor=10) -> np.ndarray:
        """
        Returns reasonable integration boundaries for the specific distribution
        based on the mode and covariance.
        """
        left = np.zeros((self.bound_dim + self.lin_dim))
        right = 2*np.pi * np.ones((self.bound_dim + self.lin_dim))
        P = self.linear_covariance()
        m = self.mode()

        for i in range(self.bound_dim, self.bound_dim + self.lin_dim):
            left[i] = m[i] - scalingFactor * np.sqrt(
                P[i - self.bound_dim, i - self.bound_dim]
            )
            right[i] = m[i] + scalingFactor * np.sqrt(
                P[i - self.bound_dim, i - self.bound_dim]
            )

        return np.vstack((left, right))
    
    def hybrid_moment_numerical(self):
        assert self.bound_dim == 1, "Only implemented for bound_dim = 1"
        m = np.empty(2 * self.bound_dim + self.lin_dim)  # Initialize the results array

        # Integration boundaries
        left, right = self.get_reasonable_integration_boundaries()

        if self.lin_dim == 1:
            m[0] = dblquad(lambda x, y: np.cos(x) * self.pdf(np.array([x, y])),
                           left[0], right[0], lambda _: left[1], lambda _: right[1])[0]
            m[1] = dblquad(lambda x, y: np.sin(x) * self.pdf(np.array([x, y])),
                           left[0], right[0], lambda _: left[1], lambda _: right[1])[0]
            m[2] = dblquad(lambda x, y: y * self.pdf(np.array([x, y])),
                           left[0], right[0], lambda _: left[1], lambda _: right[1])[0]
        elif self.lin_dim == 2:
            def integrand1(x, y, z):
                return np.cos(x) * self.pdf(np.array([x, y, z]))

            def integrand2(x, y, z):
                return np.sin(x) * self.pdf(np.array([x, y, z]))

            def integrand3(x, y, z):
                return y * self.pdf(np.array([x, y, z]))

            def integrand4(x, y, z):
                return z * self.pdf(np.array([x, y, z]))

            m[0] = nquad(integrand1, [[left[0], right[0]], [left[1], right[1]], [left[2], right[2]]])[0]
            m[1] = nquad(integrand2, [[left[0], right[0]], [left[1], right[1]], [left[2], right[2]]])[0]
            m[2] = nquad(integrand3, [[left[0], right[0]], [left[1], right[1]], [left[2], right[2]]])[0]
            m[3] = nquad(integrand4, [[left[0], right[0]], [left[1], right[1]], [left[2], right[2]]])[0]
        else:
            raise ValueError("lin_dim>2 not supported.")

        return m

    def mode(self):
        """Find the mode of the distribution by calling mode_numerical."""
        return self.mode_numerical()

    def linear_covariance(self, approximate_mean=None):
        """
        Calculates the linear covariance, or calls linear_covariance_numerical
        if a non-numerical solution doesn't exist.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, uses NaNs to flag for calculation.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None:
            approximate_mean = np.full((self.lin_dim,), np.nan)

        assert approximate_mean.shape[0] == self.lin_dim

        return self.linear_covariance_numerical(approximate_mean)

    def linear_covariance_numerical(self, approximate_mean=None):
        """
        Numerically calculates the linear covariance.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, calculates the mean.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None or np.any(np.isnan(approximate_mean)):
            approximate_mean = self.linear_mean_numerical()

        if self.bound_dim == 1 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y: (y - approximate_mean) ** 2 * self.pdf([x, y]),
                [[0, 2 * np.pi], [-np.inf, np.inf]],
            )
        elif self.bound_dim == 2 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y, z: (z - approximate_mean) ** 2 * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [0, 2 * np.pi], [-np.inf, np.inf]],
            )
        elif self.bound_dim == 1 and self.lin_dim == 2:
            C = np.empty((2, 2))
            C[0, 0], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0]) ** 2 * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )
            C[0, 1], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0])
                * (z - approximate_mean[1])
                * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )
            C[1, 0] = C[0, 1]
            C[1, 1], _ = nquad(
                lambda x, y, z: (z - approximate_mean[1]) ** 2 * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )
        else:
            raise ValueError("Cannot determine linear covariance for this dimension.")

        return C

    def condition_on_linear(self, input_lin, normalize=True):
        """
        Condition on linear.

        Parameters:
        lin_input : ndarray
            Input array.
        normalize : bool, optional
            If True (default), normalizes the distribution.

        Returns:
        dist : CustomHypertoroidalDistribution
            The distribution after conditioning.
        """
        assert (
            np.size(input_lin) == self.lin_dim and np.ndim(input_lin) <= 1
        ), "Input should be of size (lin_dim,)."

        def f_cond_unnorm(x, input_lin=input_lin):
            n_inputs = np.size(x) // x.shape[-1] if np.ndim(x) > 1 else np.size(x)
            input_repeated = np.tile(input_lin, (n_inputs, 1))
            return self.pdf(np.column_stack((x, input_repeated)))

        dist = CustomHypertoroidalDistribution(f_cond_unnorm, self.bound_dim)

        if normalize:  # Conditional need not be normalized
            dist = dist.normalize()

        return dist

    def condition_on_periodic(self, input_periodic, normalize=True):
        """
        Conditions the distribution on periodic variables

        Arguments:
            input_periodic: ndarray
                Input data, assumed to have shape (self.bound_dim,)
            normalize: bool
                If True, normalizes the distribution

        Returns:
            dist: CustomLinearDistribution
                CustomLinearDistribution instance
        """
        assert (
            np.size(input_periodic) == self.bound_dim and np.ndim(input_periodic) <= 1
        ), "Input should be of size (lin_dim,)."

        input_periodic = np.mod(input_periodic, 2 * np.pi)

        def f_cond_unnorm(x, input_periodic=input_periodic):
            n_inputs = np.size(x) // x.shape[-1] if np.ndim(x) > 1 else np.size(x)
            input_repeated = np.tile(input_periodic, (n_inputs, 1))
            return self.pdf(np.column_stack((input_repeated, x)))

        dist = CustomLinearDistribution(f_cond_unnorm, self.lin_dim)

        if normalize:  # Conditional may not be normalized
            dist = dist.normalize()

        return dist

    def linear_mean_numerical(self):
        # Define the integrands for the mean calculation
        if self.lin_dim == 1 and self.bound_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y: (y * self.pdf([x, y]))[0],
                [[0, 2 * np.pi], [-np.inf, np.inf]],
            )[0]
        elif self.bound_dim == 2 and self.lin_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [[0, 2 * np.pi], [0, 2 * np.pi], [-np.inf, np.inf]],
            )[0]
        elif self.bound_dim == 1 and self.lin_dim == 2:
            mu = np.empty(2)
            mu[0] = scipy.integrate.nquad(
                lambda x, y, z: (y * self.pdf([x, y, z]))[0],
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )[0]
            mu[1] = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )[0]
        else:
            raise ValueError("Cannot determine linear mean for this dimension.")

        return mu

    def mode_numerical(self, starting_point=None):
        """
        Find the mode of the distribution numerically.

        Parameters:
        starting_point : ndarray, optional
          The starting point for the optimization.
          If None, uses [pi * ones(self.bound_dim); zeros(self.lin_dim)]

        Returns:
        m : ndarray
          The mode of the distribution.
        """
        if starting_point is None:
            starting_point = np.concatenate(
                [np.pi * np.ones(self.bound_dim), np.zeros(self.lin_dim)]
            )

        # Define bounds for the optimization
        bounds = [
            (0, 2 * np.pi) if i < self.bound_dim else (-np.inf, np.inf)
            for i in range(self.bound_dim + self.lin_dim)
        ]

        # Perform the optimization
        res = scipy.optimize.minimize(
            lambda x: -self.pdf(x), starting_point, bounds=bounds
        )

        # Check if the optimization might have stopped early
        if np.allclose(res.x, starting_point):
            print(
                "Warning: Mode was at the starting point. This may indicate the optimizer stopped early."
            )

        return res.x

    @property
    def input_dim(self):
        return self.dim
