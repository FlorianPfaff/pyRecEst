import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.kalman_filter import KalmanFilter
from pyrecest.filters.random_matrix_tracker import RandomMatrixTracker


class TestRandomMatrixTracker(unittest.TestCase):
    def setUp(self):
        self.initial_state = np.array([1, 2])
        self.initial_covariance = np.array([[0.1, 0], [0, 0.1]])
        self.initial_extent = np.array([[1, 0.1], [0.1, 1]])
        self.measurement_noise = np.array([[0.2, 0], [0, 0.2]])

        self.tracker = RandomMatrixTracker(
            self.initial_state, self.initial_covariance, self.initial_extent
        )

    def test_initialization(self):
        np.testing.assert_array_equal(self.tracker.kinematic_state, self.initial_state)
        np.testing.assert_array_equal(self.tracker.covariance, self.initial_covariance)
        np.testing.assert_array_equal(self.tracker.extent, self.initial_extent)

    def test_get_point_estimate(self):
        expected = np.concatenate(
            [self.initial_state, np.array(self.initial_extent).flatten()]
        )
        np.testing.assert_array_equal(self.tracker.get_point_estimate(), expected)

    def test_get_point_estimate_kinematics(self):
        np.testing.assert_array_equal(
            self.tracker.get_point_estimate_kinematics(), self.initial_state
        )

    def test_get_point_estimate_extent(self):
        np.testing.assert_array_equal(
            self.tracker.get_point_estimate_extent(), self.initial_extent
        )

    def test_predict(self):
        delta_t = 0.1
        Cw = np.array([[0.05, 0.0], [0.0, 0.05]])
        tau = 1.0

        system_matrix = np.eye(2)  # 2-D random walk

        # Call the predict method
        self.tracker.predict(delta_t, Cw, system_matrix, tau)

        # Check if state and state covariance are updated correctly
        expected_state = np.array([1.0, 2.0])
        expected_covariance = self.initial_covariance + Cw
        expected_extent = self.initial_extent

        np.testing.assert_array_almost_equal(
            self.tracker.kinematic_state, expected_state, decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.tracker.covariance, expected_covariance, decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.tracker.extent, expected_extent, decimal=5
        )

    @parameterized.expand(
        [
            (
                "smaller",
                np.array([[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]]),
                "The extent should now be smaller since the measurements are closely spaced",
            ),
            (
                "larger",
                np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
                "The extent should now be larger since the measurements are spaced more widely",
            ),
        ]
    )
    def test_update(self, name, offset, _):
        ys = np.array([self.initial_state + offset_row for offset_row in offset]).T
        Cv = np.array([[0.1, 0.0], [0.0, 0.1]])
        H = np.eye(np.size(self.initial_state))

        # Call the update method
        self.tracker.update(ys, H, Cv)

        # Check if state, state covariance, and extent are updated correctly. Use KF for comparison
        kf = KalmanFilter(
            GaussianDistribution(self.initial_state, self.initial_covariance)
        )
        kf.update_linear(
            np.mean(ys, axis=1), H, (self.initial_extent + Cv) / ys.shape[1]
        )

        np.testing.assert_array_almost_equal(
            self.tracker.kinematic_state, kf.get_point_estimate(), decimal=5
        )
        np.testing.assert_array_almost_equal(
            self.tracker.covariance, kf.filter_state.C, decimal=5
        )

        # Check if extent has changed as expected
        if name == "smaller":
            np.testing.assert_array_less(
                np.zeros(2), np.linalg.eig(self.initial_extent - self.tracker.extent)[0]
            )
        elif name == "larger":
            np.testing.assert_array_less(
                np.zeros(2), np.linalg.eig(self.tracker.extent - self.initial_extent)[0]
            )
        else:
            raise ValueError(f"Invalid test name: {name}")

    @patch("matplotlib.pyplot.show")
    def test_draw_extent_3d(self, mock_show):
        self.tracker = RandomMatrixTracker(
            np.zeros(3),
            np.eye(3),
            np.diag([1, 2, 3]),
            kinematic_state_to_pos_matrix=np.eye(3),
        )
        self.tracker.plot_point_estimate()

        # Check that the plot was created
        self.assertIsInstance(plt.gcf(), plt.Figure)

        # Check that the plot is shown
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
