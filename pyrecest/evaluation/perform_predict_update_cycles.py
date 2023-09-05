import numpy as np
import time
import warnings

from .configure_for_filter import configure_for_filter

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def perform_predict_update_cycles(scenario_param, filter_param, groundtruth,
                                  measurements, precalculated_params=None,
                                  extract_all_estimates=False,
                                  cumulated_updates_preferred=None):
    if extract_all_estimates:
        all_estimates = np.empty_like(groundtruth)
    else:
        all_estimates = None
    
    # Configure filter
    filter_obj, prediction_routine, _, meas_noise_for_filter = configure_for_filter(filter_param, scenario_param, precalculated_params)
    
    # Check conditions for cumulative updates
    perform_cumulative_updates = cumulated_updates_preferred and any(np.array(scenario_param['n_meas_at_individual_time_step']) > 1)
    if cumulated_updates_preferred and any(np.array(scenario_param['n_meas_at_individual_time_step']) > 1) and scenario_param.get('plot', False):
        warnings.warn("When plotting, measurements are fused sequentially.")
        perform_cumulative_updates = False
    
    # Start timer
    start_time = time.time()
    
    # Perform evaluation
    for t in range(scenario_param['timesteps']):
        # Update
        if 'MTT' in scenario_param['manifold_type']:
            raise NotImplementedError("MTT not implemented yet.")
        elif perform_cumulative_updates: # pylint: disable=no-else-raise
            raise NotImplementedError("Cumulative updates not implemented yet.")
        else: # pylint: disable=no-else-raise
            n_updates = scenario_param['n_meas_at_individual_time_step'][t]
            all_meas_curr_time_step = measurements[t]
            for m in range(n_updates):
                curr_meas = all_meas_curr_time_step[m]

                if not scenario_param["use_likelihood"]:
                    filter_obj.update_identity(measurement=curr_meas, meas_noise_cov=meas_noise_for_filter)

                # If plotting is required
                if scenario_param.get('plot', False):
                    raise NotImplementedError("Plotting is not implemented yet.")

        # Save results only if required (takes time)
        if extract_all_estimates:
            all_estimates[t, :] = filter_obj.get_point_estimate()
            
        # Predict
        if scenario_param['apply_sys_noise_times'][t]:
            if scenario_param.get('inputs') is None:
                prediction_routine()
            else:
                prediction_routine(scenario_param['inputs'][:, t])
            
            # If plotting is required
            if scenario_param.get('plot', False) and t != scenario_param['timesteps']:
                raise NotImplementedError("Plotting is not implemented yet.")
                
    # End timer
    time_elapsed = time.time() - start_time
    
    # Get the final filter state and estimate
    last_filter_state = filter_obj.filter_state
    if all_estimates is not None:
        last_estimate = all_estimates[-1, :]
    else:
        last_estimate = filter_obj.get_point_estimate()
    
    return time_elapsed, last_filter_state, last_estimate, all_estimates
