import unittest
import numpy as np
from pyrecest.backend import random, zeros_like, mod, pi, column_stack, linspace, arange
from pyrecest.distributions.abstract_se2_distribution import AbstractSE2Distribution


class AbstractSE2DistributionTest(unittest.TestCase):

    def test_plot_trajectory(self):
        AbstractSE2Distribution.plot_trajectory(linspace(0, np.pi / 4, 10), column_stack([arange(1, 11), arange(1, 11)]), animate=True)
        AbstractSE2Distribution.plot_trajectory(linspace(0, np.pi / 4, 10), column_stack([arange(1, 11), arange(1, 11)]), animate=False)
        AbstractSE2Distribution.plot_trajectory(linspace(0, np.pi / 4, 10), column_stack([arange(1, 11), arange(1, 11)]), pos_color=[1, 0, 1], angle_color=[0, 1, 1])
        AbstractSE2Distribution.plot_trajectory(linspace(0, np.pi / 4, 10), column_stack([arange(1, 11), arange(1, 11)]), fade=False)

    def test_dual_quaternion_to_angle_pos(self):
        x = 10 * random.rand(size=(10, 3)) - 5
        dqs = AbstractSE2Distribution.angle_pos_to_dual_quaternion(x)

        x_converted = zeros_like(x)
        x_converted[:, 0], x_converted[:, 1:] = AbstractSE2Distribution.dual_quaternion_to_angle_pos(dqs)
        np.testing.assert_allclose(x_converted, column_stack([mod(x[:, 0], 2 * pi), x[:, 1:]]), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
