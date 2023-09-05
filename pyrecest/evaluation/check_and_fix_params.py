from pyrecest.distributions import AbstractManifoldSpecificDistribution


def check_and_fix_params(scenario_param):
    # Initialize default values if they are not present
    scenario_param.setdefault("use_transition", False)
    scenario_param.setdefault("use_likelihood", False)
    scenario_param.setdefault("n_targets", 1)
    scenario_param.setdefault('apply_sys_noise_times', [True] * (scenario_param['timesteps'] - 1) + [False])


    # Check for 'timesteps'
    timesteps = scenario_param["timesteps"]
    if timesteps is not None:
        assert timesteps > 0 and isinstance(timesteps, int)

    if "MTT" in scenario_param.get("manifold_type", ""):
        assert (
            "n_meas_at_individual_time_step" not in scenario_param
            and "meas_per_step" not in scenario_param
        )
        scenario_param.setdefault("detectionProbability", 1)
        scenario_param.setdefault("clutter_rate", 0)

        assert (
            "observed_area" in scenario_param or scenario_param["clutter_rate"] == 0
        ), "Can only add clutter if observedArea is set."

    elif (
        "n_meas_at_individual_time_step" in scenario_param
        and "meas_per_step" in scenario_param
    ):
        raise ValueError(
            "Do not provide n_meas_at_individual_time_step and meas_per_step at the same time."
        )

    elif (
        "n_meas_at_individual_time_step" not in scenario_param
        and "meas_per_step" in scenario_param
    ):
        assert isinstance(
            scenario_param["meas_per_step"], int
        ), "meas_per_step must be an integer"
        assert scenario_param["meas_per_step"] > 0, "meas_per_step must be positive"

        scenario_param["n_meas_at_individual_time_step"] = [
            scenario_param["meas_per_step"]
        ] * scenario_param["timesteps"]
        del scenario_param["meas_per_step"]

        assert all(
            x > 0 for x in scenario_param["n_meas_at_individual_time_step"]
        ), "n_meas_at_individual_time_step must contain positive values"
        assert all(
            isinstance(x, int) for x in scenario_param["n_meas_at_individual_time_step"]
        ), "n_meas_at_individual_time_step must contain integer values"

    elif "n_meas_at_individual_time_step" not in scenario_param:
        scenario_param["n_meas_at_individual_time_step"] = [1] * scenario_param[
            "timesteps"
        ]

    assert isinstance(
        scenario_param["initial_prior"], AbstractManifoldSpecificDistribution
    )

    return scenario_param
