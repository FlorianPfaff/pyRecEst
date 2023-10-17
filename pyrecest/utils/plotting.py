from pyrecest.backend import linalg
from math import pi
from pyrecest.backend import sqrt
from pyrecest.backend import sin
from pyrecest.backend import reshape
from pyrecest.backend import outer
from pyrecest.backend import ones
from pyrecest.backend import linspace
from pyrecest.backend import cos
from pyrecest.backend import array
import matplotlib.pyplot as plt
import numpy as np


def plot_ellipsoid(center, shape_matrix, scaling_factor=1, color="blue"):
    if center.shape[0] == 2:
        plot_ellipsoid_2d(center, shape_matrix, scaling_factor, color)
    elif center.shape[0] == 3:
        plot_ellipsoid_3d(center, shape_matrix, scaling_factor, color)
    else:
        raise ValueError("Only 2D and 3D ellipsoids are supported.")


def plot_ellipsoid_2d(center, shape_matrix, scaling_factor=1, color="blue"):
    xs = linspace(0, 2 * pi, 100)
    ps = scaling_factor * shape_matrix @ np.column_stack((cos(xs), sin(xs)))
    plt.plot(ps[0] + center[0], ps[1] + center[1], color=color)
    plt.show()


def plot_ellipsoid_3d(center, shape_matrix, scaling_factor=1, color="blue"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u = linspace(0, 2 * pi, 100)
    v = linspace(0, pi, 100)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(np.size(u)), cos(v))

    V, D = linalg.eig(shape_matrix)
    all_coords = V @ sqrt(D) @ array(
        [x.ravel(), y.ravel(), z.ravel()]
    ) + center.reshape(-1, 1)
    x = reshape(all_coords[0], x.shape)
    y = reshape(all_coords[1], y.shape)
    z = reshape(all_coords[2], z.shape)

    ax.plot_surface(
        scaling_factor * x,
        scaling_factor * y,
        scaling_factor * z,
        color=color,
        alpha=0.7,
        linewidth=0,
        antialiased=False,
    )
    plt.show()