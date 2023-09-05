from pyrecest.filters import HypertoroidalParticleFilter, KalmanFilter


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def configure_for_filter(filter_param, scenario_param, precalculated_params=None):
    if precalculated_params is not None:
        raise NotImplementedError("No filters using precalculated parameters have been implemented so far.")
    # Check for likelihood and measurement noise in scenario parameters
    likelihood_for_filter = scenario_param.get("likelihood", None)
    meas_noise_for_filter = scenario_param.get("meas_noise", None)

    # Switch-case based on filter name
    filter_name = filter_param["name"]

    if filter_name == "kf":
        configured_filter = KalmanFilter(scenario_param["initial_prior"])
        meas_noise_for_filter = meas_noise_for_filter.covariance()
        if scenario_param.get("inputs") is None:

            def prediction_routine():  # type: ignore
                return configured_filter.predict_identity(scenario_param["sys_noise"].covariance())
        else:

            def prediction_routine(curr_input):  # type: ignore
                return configured_filter.predict_identity(
                    scenario_param["sys_noise"], curr_input
                )

    elif filter_name == "twn":
        raise NotImplementedError("ToroidalWNFilter not implemented yet")

    elif filter_name in ["iff", "sqff"]:
        raise NotImplementedError("IF and SQFF not implemented yet")

    elif filter_name == "htgf":
        raise NotImplementedError("HTGF not implemented yet")

    elif filter_name == "fig":
        raise NotImplementedError("FIG not implemented yet")

    elif filter_name == "ishf":
        raise NotImplementedError("ISHF not implemented yet")

    elif filter_name == "pf":
        no_particles = filter_param.get("parameter", None)

        if no_particles is None or no_particles == 0:
            raise ValueError("Using zero particles does not make sense")

        manifold_type = scenario_param.get("manifold_type")

        if manifold_type in ["circle", "hypertorus"]:
            assert (
                scenario_param.get("inputs") is None
            ), "Inputs currently not supported for the current setting."

            configured_filter = HypertoroidalParticleFilter(
                no_particles, scenario_param["initial_prior"].dim
            )
            configured_filter.set_state(scenario_param["initial_prior"])

            if "gen_next_state_with_noise" in scenario_param:

                def prediction_routine():  # type: ignore
                    return configured_filter.predict_nonlinear(
                        scenario_param["gen_next_state_with_noise"],
                        None,
                        scenario_param.get("genNextStateWithNoiseIsVectorized", False),
                    )

            elif "sys_noise" in scenario_param:

                def prediction_routine():  # type: ignore
                    return configured_filter.predict_nonlinear(
                        lambda x: x, scenario_param["sys_noise"], True
                    )

        elif manifold_type in [
            "hypersphere",
            "hypersphere_general",
            "hypersphere_symmetric",
        ]:
            assert (
                scenario_param.get("inputs") is None
            ), "Inputs currently not supported for the current setting."
            raise NotImplementedError(
                "HypersphericalParticleFilter not implemented yet"
            )

    elif filter_name in ["sgf", "hgf"]:
        raise NotImplementedError("SGF and HGF not implemented yet")

    elif filter_name == "vmf":
        raise NotImplementedError("VMF not implemented yet")

    elif filter_name == "s3f":
        raise NotImplementedError("S3F not implemented yet")

    elif filter_name == "se2bf":
        raise NotImplementedError("SE2BF not implemented yet")

    elif filter_name == "se2ukfm":
        raise NotImplementedError("SE2UKFM not implemented yet")

    elif filter_name == "dummy":
        raise NotImplementedError("Dummy not implemented yet")

    elif filter_name == "gnn":
        raise NotImplementedError("GNN not implemented yet")

    else:
        raise ValueError("Filter currently unsupported")

    return (
        configured_filter,
        prediction_routine,
        likelihood_for_filter,
        meas_noise_for_filter,
    )
