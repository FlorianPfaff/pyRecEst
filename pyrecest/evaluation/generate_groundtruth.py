import numpy as np


# pylint: disable=too-many-branches
def generate_groundtruth(scenario_param, x0=None):
    """
    Generate ground truth based on the given scenario parameters.

    Parameters:
        scenario_param (dict): Dictionary containing scenario parameters.
        x0 (ndarray): Starting point (optional)

    Returns:
        groundtruth (ndarray): Generated ground truth.
    """
    if x0 is None:
        x0 = scenario_param["initial_prior"].sample(1)
        
    assert (
        np.ndim(x0) == 1
        and scenario_param["n_targets"] == 1
        or x0.shape[0] == scenario_param["n_targets"]
    ), "Mismatch in number of targets."

    # Initialize ground truth
    groundtruth = np.empty(
        (scenario_param["timesteps"], scenario_param["n_targets"], x0.shape[-1])
    )

    for target_no in range(scenario_param["n_targets"]):
        groundtruth[0, target_no, :] = x0

        if "inputs" in scenario_param:
            assert (
                scenario_param["inputs"].shape[1] == scenario_param["timesteps"] - 1
            ), "Mismatch in number of timesteps."

        if "gen_next_state_with_noise" in scenario_param:
            for t in range(1, scenario_param["timesteps"]):
                if "inputs" not in scenario_param or scenario_param["inputs"] is None:
                    groundtruth[t, target_no, :] = scenario_param[
                        "gen_next_state_with_noise"
                    ](groundtruth[t - 1, target_no, :])
                else:
                    groundtruth[t, target_no, :] = scenario_param[
                        "gen_next_state_with_noise"
                    ](
                        groundtruth[t - 1, target_no, :],
                        scenario_param["inputs"][t - 1, :],
                    )

        elif "sys_noise" in scenario_param:
            for t in range(1, scenario_param["timesteps"]):
                if "gen_next_state_without_noise" in scenario_param:
                    if (
                        "inputs" not in scenario_param
                        or scenario_param["inputs"] is None
                    ):
                        state_to_add_noise_to = scenario_param[
                            "gen_next_state_without_noise"
                        ](groundtruth[t - 1, target_no, :])
                    else:
                        state_to_add_noise_to = scenario_param[
                            "gen_next_state_without_noise"
                        ](
                            groundtruth[:, t - 1, target_no],
                            scenario_param["inputs"][t - 1, :],
                        )
                else:
                    assert (
                        "inputs" not in scenario_param
                        or scenario_param["inputs"] is None
                    ), "No inputs accepted for the identity system model."
                    state_to_add_noise_to = groundtruth[t - 1, target_no, :]

                groundtruth[t, target_no, :] = state_to_add_noise_to + scenario_param[
                    "sys_noise"
                ].sample(1)

        else:
            raise ValueError("Cannot generate groundtruth.")

    if scenario_param["n_targets"] == 1:
        groundtruth = groundtruth[:, 0, :]
    return groundtruth
