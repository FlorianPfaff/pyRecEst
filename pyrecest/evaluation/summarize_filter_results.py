import warnings

import numpy as np

from .determine_all_deviations import determine_all_deviations
from .get_distance_function import get_distance_function
from .get_extract_mean import get_extract_mean


# pylint: disable=too-many-arguments,too-many-locals
def summarize_filter_results(
    scenario_param,
    filter_configs,
    runtimes,
    groundtruths,
    run_failed,
    last_filter_states=None,
    last_estimates=None,
    **_
):
    if last_filter_states is not None and last_estimates is not None:
        warnings.warn(
            "Provided both last_filter_states and last_estimates. Using last_estimates."
        )
        filter_results = last_estimates
    elif last_filter_states is not None:
        filter_results = last_filter_states
    else:
        raise ValueError(
            "Either last_filter_states or last_estimates must be provided."
        )

    if groundtruths.shape[1] < 1000:
        warnings.warn("Using less than 1000 runs. This may lead to unreliable results.")

    extract_mean = get_extract_mean(scenario_param["manifold_type"])
    distance_function = get_distance_function(scenario_param["manifold_type"])
    errors_all = determine_all_deviations(
        filter_results, extract_mean, distance_function, groundtruths
    )
    errors_mean = np.mean(errors_all, axis=1)
    errors_std = np.std(errors_all, axis=1)
    times_mean = np.mean(runtimes, axis=1)
    failure_rates = np.sum(run_failed, axis=1) / run_failed.shape[1]

    results_summarized = filter_configs
    for d, err, std, time, fail_rate in zip(
        results_summarized, errors_mean, errors_std, times_mean, failure_rates
    ):
        d["error_mean"] = err
        d["error_std"] = std
        d["time_mean"] = time
        d["failure_rate"] = fail_rate

    return results_summarized
