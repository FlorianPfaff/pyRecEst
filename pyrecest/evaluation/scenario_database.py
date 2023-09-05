import warnings

import numpy as np
from pyrecest.distributions import GaussianDistribution


def scenario_database(scenario, scenario_customization_params=None):
    scenario_param = {
        "initial_prior": lambda: "Scenario param not initialized",
        "timesteps": None,
        "allSeeds": None,
    }

    if scenario == "R2randomWalk":
        scenario_param["manifold_type"] = "Euclidean"
        scenario_param["timesteps"] = 10
        scenario_param["initial_prior"] = GaussianDistribution(
            np.zeros(2), 0.5 * np.eye(2)
        )
        scenario_param["meas_noise"] = GaussianDistribution(
            np.zeros(2), 0.5 * np.eye(2)
        )
        scenario_param["sys_noise"] = GaussianDistribution(np.zeros(2), 0.5 * np.eye(2))
        scenario_param["gen_next_state_without_noise_is_vectorized"] = True
    else:
        if scenario_customization_params is None:
            raise ValueError("Scenario not recognized.")

        warnings.warn(
            "Scenario not recognized. Assuming scenario_customization_params contains all parameters."
        )
        scenario_param = scenario_customization_params

        return scenario_param

    return scenario_param
