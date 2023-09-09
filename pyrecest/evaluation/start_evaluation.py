import datetime
import os
import random
from typing import Any

import numpy as np

from .check_and_fix_params import check_and_fix_params
from .iterate_configs_and_runs import iterate_configs_and_runs
from .scenario_database import scenario_database


# pylint: disable=R0913,R0914
# @beartype
def start_evaluation(
    scenario: str | dict[str, Any],
    filter_configs: list[dict[str, Any]],
    n_runs: int,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    scenario_customization_params: None | dict = None,
    tolerate_failure: bool = False,
    initial_seed: None | int | np.uint32 = None,
    consecutive_seed: bool = False,
    auto_warning_on_off: bool = False,
) -> tuple[
    dict,
    list[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray[np.ndarray],
]:
    """
    Main function for evaluating filters.

    Parameters:
        scenario (Union[str, Dict[str, Any]]): Scenario name or full parameterization.
        filters (List[Dict[str, Any]]): List of filters with names and parameters.
        no_runs (int): Number of runs.

    Optional Parameters:
        save_folder (str): The directory where results will be saved. Default is '.'.
        plot_each_step (bool): Whether to plot each step or not. Useful for debugging. Default is False.
        convert_to_point_estimate_during_runtime (bool): Whether to convert to point estimates during runtime or not. Default is False.
        extract_all_point_estimates (bool): Whether to extract all point estimates or not. Default is False.
        scenario_customization_params (Union[None, Dict]): Optional scenario parameters. Default is None.
        tolerate_failure (bool): Whether to tolerate failures or not. Default is False.
        initial_seed (Union[None, np.uint32]): The initial random seed. Default is a random 32-bit integer.
        consecutive_seed (bool): Whether to use consecutive seeds or not. Default is False.
        auto_warning_on_off (bool): Whether to automatically turn warnings on or off. Useful for debugging. Default is False.

    Returns:
        Tuple: results, groundtruths, and scenario parameters.
    """

    filter_configs = [
        {"name": f["name"], "parameter": p}
        for f in filter_configs
        for p in (f["parameter"] or [None])
    ]

    if initial_seed is None:
        initial_seed = np.uint32(random.randint(1, 0xFFFFFFFF))  # nosec

    if isinstance(scenario, dict):
        scenario_param = scenario
        scenario_param["name"] = "custom"
    else:
        scenario_param = scenario_database(scenario, scenario_customization_params)
        scenario_param["name"] = scenario

    scenario_param["plot"] = plot_each_step
    scenario_param = check_and_fix_params(scenario_param)

    if consecutive_seed:
        scenario_param["all_seeds"] = list(range(initial_seed, initial_seed + n_runs))
    else:
        random.seed(initial_seed)
        scenario_param["all_seeds"] = [
            random.randint(1, 0xFFFFFFFF) for _ in range(n_runs)  # nosec
        ]

    (
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
    ) = iterate_configs_and_runs(
        scenario_param,
        filter_configs,
        n_runs,
        convert_to_point_estimate_during_runtime,
        extract_all_point_estimates,
        tolerate_failure,
        auto_warning_on_off,
    )

    date_and_time = datetime.datetime.now()
    filename = os.path.join(
        save_folder,
        f"{scenario_param['name']}_{date_and_time.strftime('%Y-%m-%d--%H-%M-%S')}.npy",
    )
    np.save(
        filename,
        {
            "groundtruths": groundtruths,
            "measurements": measurements,
            "run_failed": run_failed,
            "last_filter_states": last_filter_states,
            "runtimes": runtimes,
            "scenario_param": scenario_param,
            "filter_configs": filter_configs,
        },
        allow_pickle=True,
    )

    return (
        scenario_param,  # pylint: disable=R0801
        filter_configs,  # pylint: disable=R0801
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
    )
