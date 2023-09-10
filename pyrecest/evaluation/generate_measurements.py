import numpy as np
from beartype import beartype
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    GaussianDistribution,
    VonMisesFisherDistribution,
    WatsonDistribution,
)
from scipy.stats import poisson
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, Point, Polygon


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def generate_measurements(groundtruth, simulation_config):
    """
    Generate measurements based on the given groundtruth and scenario parameters.

    Parameters:
        groundtruth (ndarray): Ground truth data.
        simulation_config (dict): Dictionary containing scenario parameters.

    Returns:
        measurements (list): List of generated measurements at each time step.
        Is a list because the number of measurements can vary at each time step.
        Comprises timesteps elements, each of which is a numpy array of shape
        (n_meas_at_individual_time_step[t], n_dim).
    """
    assert "n_meas_at_individual_time_step" not in simulation_config or np.shape(
        simulation_config["n_meas_at_individual_time_step"]
    ) == (simulation_config["n_timesteps"],)
    measurements = np.empty(simulation_config["n_timesteps"], dtype=np.ndarray)

    if simulation_config.get("mtt", False) and simulation_config.get("eot", False):
        raise NotImplementedError(
            "Multiple extended object tracking is currently not supported."
        )

    if simulation_config.get("eot", False):
        shape = simulation_config["target_shape"]
        sample_on = simulation_config["sample_on"]
        assert isinstance(
            shape, Polygon
        ), "Currently only shapely polygons are supported as target shapes."

        for t in range(simulation_config["n_timesteps"]):
            if groundtruth[0].shape[-1] == 2:
                curr_shape = translate(
                    shape, groundtruth[0][..., 0], yoff=groundtruth[0][..., 1]
                )
            elif groundtruth[0].shape[-1] == 3:
                curr_shape = rotate(
                    translate(
                        shape, groundtruth[0][..., 1], yoff=groundtruth[0][..., 2]
                    ),
                    angle=groundtruth[0][..., 0],
                    origin="centroid",
                )
            else:
                raise ValueError(
                    "Currently only R^2 and SE(2) scenarios are supported."
                )

            if simulation_config.get("n_meas_at_individual_time_step", None):
                assert (
                    "intensity_lambda" not in simulation_config
                ), "Cannot use both intensity_lambda and n_meas_at_individual_time_step."
                n_meas_curr = simulation_config["n_meas_at_individual_time_step"][t]
            else:
                if sample_on == "vertices":
                    n_meas_curr = generate_n_measurements_PPP(
                        curr_shape.length, simulation_config["intensity_lambda"]
                    )
                elif sample_on == "surface":
                    n_meas_curr = generate_n_measurements_PPP(
                        curr_shape.area, simulation_config["intensity_lambda"]
                    )
                else:
                    raise ValueError(
                        "sample_on must be either 'vertices' or 'surface'."
                    )

            if sample_on == "vertices":
                measurements[t] = random_points_on_boundary(curr_shape, n_meas_curr)
            elif sample_on == "surface":
                measurements[t] = random_points_within(curr_shape, n_meas_curr)
            else:
                raise ValueError("sample_on must be either 'vertices' or 'surface'.")

    elif simulation_config.get("mtt", False):
        assert (
            simulation_config["clutter_rate"] == 0
        ), "Clutter currently not supported."

        n_observations = np.random.binomial(
            1,
            simulation_config["detection_probability"],
            (simulation_config["n_timesteps"], simulation_config["n_targets"]),
        )

        for t in range(simulation_config["n_timesteps"]):
            n_meas_at_t = np.sum(n_observations[t, :])
            measurements[t] = np.nan * np.zeros(
                (simulation_config["meas_matrix_for_each_target"].shape[0], n_meas_at_t)
            )

            meas_no = 0
            for target_no in range(simulation_config["n_targets"]):
                if n_observations[t, target_no] == 1:
                    meas_no += 1
                    measurements[t][meas_no - 1, :] = np.dot(
                        simulation_config["meas_matrix_for_each_target"],
                        groundtruth[t, target_no, :],
                    ) + simulation_config["meas_noise"].sample(1)
                else:
                    assert (
                        n_observations[t, target_no] == 0
                    ), "Multiple measurements currently not supported."

            assert meas_no == n_meas_at_t, "Mismatch in number of measurements."

    else:
        if "meas_generator" in simulation_config:
            raise NotImplementedError(
                "Scenarios based on a 'measGenerator' are currently not supported."
            )
        for t in range(simulation_config["n_timesteps"]):
            n_meas = simulation_config["n_meas_at_individual_time_step"][t]
            meas_noise = simulation_config["meas_noise"]

            if isinstance(meas_noise, AbstractHypertoroidalDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = np.mod(
                    np.squeeze(
                        np.tile(
                            groundtruth[t - 1],
                            (
                                n_meas,
                                1,
                            ),
                        )
                        + noise_samples
                    ),
                    2 * np.pi,
                )

            elif isinstance(
                meas_noise, (VonMisesFisherDistribution, WatsonDistribution)
            ):
                curr_dist = meas_noise
                curr_dist.mu = groundtruth[t - 1]
                measurements[t] = curr_dist.sample(n_meas)

            elif isinstance(meas_noise, GaussianDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = np.squeeze(
                    np.tile(
                        groundtruth[t - 1],
                        (
                            n_meas,
                            1,
                        ),
                    )
                    + noise_samples
                )

    return measurements


@beartype
def generate_n_measurements_PPP(area: float, intensity_lambda: float) -> int:
    # Compute the expected number of points
    expected_num_points = intensity_lambda * area
    # Get the actual number of points to generate as a realization from a Poisson distribution
    return poisson.rvs(expected_num_points)


@beartype
def random_points_within(poly: Polygon, num_points: int) -> np.ndarray:
    min_x, min_y, max_x, max_y = poly.bounds
    points = np.empty((num_points,), dtype=Point)

    for i in range(num_points):
        random_point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        while not random_point.within(poly):
            random_point = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            
        points[i] = random_point

    return np.array(points)


@beartype
def random_points_on_boundary(poly: Polygon, num_points: int) -> np.ndarray:
    points = np.empty((num_points,), dtype=Point)

    if isinstance(poly.boundary, LineString):
        lines = [poly.boundary]
    elif isinstance(poly.boundary, MultiLineString):
        lines = list(poly.boundary)

    for i in range(num_points):
        # Compute total perimeter
        perimeter = poly.length

        # Generate a random distance along the perimeter
        distance = np.random.uniform(0, perimeter)

        # Traverse the edges to place the point
        for line in lines:
            if distance < line.length:
                points[i] = line.interpolate(distance)
                break
            distance -= line.length

    return np.array([(point.x, point.y) for point in points])
