import numpy as np
from beartype import beartype
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from pyrecest.distributions import GaussianDistribution

from .abstract_euclidean_filter import AbstractEuclideanFilter


class KalmanFilter(AbstractEuclideanFilter):
    @beartype
    def __init__(
        self, initial_state: GaussianDistribution | tuple[np.ndarray, np.ndarray]
    ):
        """
        Initialize the Kalman filter with the initial state.

        :param initial_state: Provide GaussianDistribution or mean and covariance as initial state.
        """
        if isinstance(initial_state, GaussianDistribution):
            dim_x = initial_state.dim
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            dim_x = len(initial_state[0])
        else:
            raise ValueError(
                "initial_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )

        self._filter_state = FilterPyKalmanFilter(dim_x=dim_x, dim_z=dim_x)
        self.filter_state = initial_state

    @property
    def filter_state(
        self,
    ) -> (
        GaussianDistribution | tuple[np.ndarray, np.ndarray]
    ):  # It can only return GaussianDistribution, allowing the output of tuples just serves to prevent mypy linter warnings
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)

    @beartype
    @filter_state.setter
    def filter_state(
        self, new_state: GaussianDistribution | tuple[np.ndarray, np.ndarray]
    ):
        """
        Set the filter state.

        :param new_state: Provide GaussianDistribution or mean and covariance as state.
        """
        if isinstance(new_state, GaussianDistribution):
            self._filter_state.x = new_state.mu
            self._filter_state.P = new_state.C
        elif isinstance(new_state, tuple) and len(new_state) == 2:
            self._filter_state.x = new_state[0]
            self._filter_state.P = new_state[1]
        else:
            raise ValueError(
                "new_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )

    @beartype
    def predict_identity(self, sys_noise_cov: np.ndarray, sys_input: np.ndarray = None):
        """
        Predicts the next state assuming identity transition matrix.

        :param sys_noise_mean: System noise mean.
        :param sys_input: System noise covariance.
        """
        system_matrix = np.eye(self._filter_state.x.shape[0])
        B = np.eye(system_matrix.shape[0]) if sys_input is not None else None
        self._filter_state.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=sys_input)

    @beartype
    def predict_linear(
        self,
        system_matrix: np.ndarray,
        sys_noise_cov: np.ndarray,
        sys_input: np.ndarray | None = None,
    ):
        """
        Predicts the next state assuming a linear system model.

        :param system_matrix: System transition matrix.
        :param sys_noise_cov: System noise covariance.
        :param sys_input: System input.
        """
        if sys_input is not None and system_matrix.shape[0] != sys_input.shape[0]:
            raise ValueError(
                "The number of rows in system_matrix should match the number of elements in sys_input"
            )

        B = np.eye(system_matrix.shape[0]) if sys_input is not None else None
        self._filter_state.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=sys_input)

    @beartype
    def update_identity(self, measurement: np.ndarray, meas_noise_cov: np.ndarray):
        """
        Update the filter state with measurement, assuming identity measurement matrix.

        :param measurement: Measurement.
        :param meas_noise_cov: Measurement noise covariance.
        """
        self.update_linear(measurement=measurement, measurement_matrix=np.eye(self.dim), meas_noise_cov=meas_noise_cov)
        
    @beartype
    def update_linear(
        self,
        measurement: np.ndarray,
        measurement_matrix: np.ndarray,
        meas_noise_cov: np.ndarray,
    ):
        """
        Update the filter state with measurement, assuming a linear measurement model.

        :param measurement: Measurement.
        :param measurement_matrix: Measurement matrix.
        :param meas_noise_cov: Covariance matrix for measurement.
        """
        self._filter_state.dim_z = measurement_matrix.shape[0]
        self._filter_state.update(z=measurement, R=meas_noise_cov, H=measurement_matrix)

    @beartype
    def get_point_estimate(self) -> np.ndarray:
        """Returns the mean of the current filter state."""
        return self._filter_state.x
