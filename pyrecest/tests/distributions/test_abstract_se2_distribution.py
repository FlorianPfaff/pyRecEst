import unittest
import numpy.testing as npt
from pyrecest.backend import random, mod, pi, column_stack, linspace, arange
from pyrecest.distributions.abstract_se2_distribution import AbstractSE2Distribution
from parameterized import parameterized
import matplotlib.pyplot as plt

class AbstractSE2DistributionTest(unittest.TestCase):

    @parameterized.expand([
        ("animate_on_fade_on", True, True, None, None),
        ("animate_on_fade_off", True, True, None, None),
        ("animate_off_fade_on", False, True, None, None),
        ("animate_off_fade_off", False, False, None, None),
        ("custom_colors", False, False, [1, 0, 1], [0, 1, 1]),
    ])
    def test_plot_trajectory(self, _, animate, fade, pos_color, angle_color):
        plt.close('all')
        periodic_states = linspace(0, pi / 4, 10)
        lin_states = column_stack([arange(1, 11), arange(1, 11)])
        AbstractSE2Distribution.plot_trajectory(
            periodic_states,
            lin_states,
            animate=animate,
            fade=fade,
            pos_color=pos_color,
            angle_color=angle_color
        )

    def test_dual_quaternion_to_angle_pos(self):
        x = 10 * random.uniform(size=(10, 3)) - 5
        dqs = AbstractSE2Distribution.angle_pos_to_dual_quaternion(x)

        angle_converted, pos_converted = AbstractSE2Distribution.dual_quaternion_to_angle_pos(dqs)
        x_converted = column_stack([angle_converted, pos_converted])
        expected = column_stack([mod(x[:, 0], 2 * pi), x[:, 1:]])
        npt.assert_allclose(x_converted, expected, rtol=4e-6)


if __name__ == '__main__':
    unittest.main()
