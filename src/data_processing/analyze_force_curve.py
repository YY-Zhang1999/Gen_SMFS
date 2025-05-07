# Function that analyzes force curve data and generates WLC model fit and parameters
# Based on original MATLAB code by Sivaraman R
import re

import numpy as np
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import os  # For file and directory operations
from igor.binarywave import load as ibw_load  # Using igor library to read .ibw files


class ForceCurveAnalyzer:
    def __init__(self, temperature_k=298.0, fit_threshold_percent=90.0, make_plots=True, diagnostics=False):
        """
        Initializes the ForceCurveAnalyzer.

        Parameters:
        temperature_k (float): Temperature in Kelvin.
        fit_threshold_percent (float): The minimum fit quality (0-100) to consider a WLC fit successful.
        make_plots (bool): If True, generates and displays plots for each curve.
        diagnostics (bool): If True, enables more detailed plotting and console output for debugging.
        """
        self.temperature_k = temperature_k
        self.fit_threshold_percent = fit_threshold_percent
        self.make_plots = make_plots
        self.diagnostics = diagnostics
        self.kb = 1.38064852e-23  # Boltzmann constant in J/K or m^2 kg s^-2 K^-1
        self.fig_no = 0

    def _get_force_wlc(self, lc, p, separation, temperature_k):
        """
        Calculates the force for a Worm-Like Chain (WLC) model.

        Parameters:
        lc (float): Contour length of the polymer (meters).
        p (float): Persistence length of the polymer (meters).
        separation (float or np.array): End-to-end separation (extension) of the polymer (meters).
        temperature_k (float): Temperature in Kelvin.

        Returns:
        float or np.array: Calculated force in Newtons (N).
        """
        if lc <= 0 or p <= 0:  # Basic validation
            if isinstance(separation, np.ndarray):
                return np.zeros_like(separation)
            return 0.0

        # Ensure separation does not exceed contour length to avoid division by zero or negative sqrt
        # Cap relative extension to avoid numerical instability near r=1
        # Max relative extension typically around 0.9 to 0.99 for stable calculation
        max_r = 0.99
        relative_extension = np.clip(np.abs(separation) / lc, 0, max_r)

        # Marko-Siggia WLC force equation (interpolated form for better behavior at low forces)
        # F = (kT/P) * [1/(4*(1-x/L)^2) - 1/4 + x/L]
        force_term = 0.25 * (1.0 / (1.0 - relative_extension) ** 2) - 0.25 + relative_extension
        force = (self.kb * temperature_k / p) * force_term
        return force

    def _get_cost_wlc(self, params_wlc, separation_data, temperature_k, force_data):
        """
        Cost function for WLC model fitting. Minimized to find best Lc and P.

        Parameters:
        params_wlc (tuple): A tuple (Lc, P) representing contour length and persistence length.
        separation_data (np.array): Array of separation values (m).
        temperature_k (float): Temperature in Kelvin.
        force_data (np.array): Array of corresponding experimental force values (N).

        Returns:
        float: The cost (sum of squared differences) for the given parameters.
        """
        lc, p = params_wlc

        # Add constraints to avoid non-physical values during optimization
        if lc <= 0 or p <= 0:
            return 1e12  # Return a large cost if parameters are invalid

        f_calculated = self._get_force_wlc(lc, p, separation_data, temperature_k)

        # Handle potential NaN or Inf values from force calculation if inputs are problematic
        if np.any(np.isnan(f_calculated)) or np.any(np.isinf(f_calculated)):
            return 1e12  # Large cost for invalid calculations

        f_diff = f_calculated - force_data
        cost = np.sum(f_diff ** 2)  # Sum of squared errors

        # Scale cost to make it more prominent for the optimizer, similar to MATLAB's 1e20 scaling
        # This can sometimes help with convergence for certain optimizers.
        return cost * 1e20

    def _extract_parameters_from_notes(self, wave_notes_str):
        """
        Extracts spring constant and retract velocity from the Igor wave notes.

        Parameters:
        wave_notes_str (str): The decoded string from the 'note' field of the IBW wave.

        Returns:
        tuple: (spring_constant_nm, retract_velocity_ms, k_read_success, v_read_success)
               spring_constant_nm (float): Spring constant in N/m. Default if not found.
               retract_velocity_ms (float): Retract velocity in m/s. Default if not found.
               k_read_success (bool): True if spring constant was successfully read.
               v_read_success (bool): True if velocity was successfully read.
        """
        spring_constant_nm = 0.03  # Default spring constant (N/m) - typical for AFM
        retract_velocity_ms = 1e-6  # Default velocity (m/s)
        k_read_success = False
        v_read_success = False

        # Extract Spring Constant
        k_spr_match = re.search(r"SpringConstant:\s*(-?\d+\.?\d*)", wave_notes_str, re.IGNORECASE)
        print(wave_notes_str)
        if k_spr_match:
            try:
                spring_constant_nm = float(k_spr_match.group(1))
                k_read_success = True
                if self.diagnostics: print(f"Read Spring Constant: {spring_constant_nm} N/m")
            except ValueError:
                if self.diagnostics: print(f"Could not parse Spring Constant value: {k_spr_match.group(1)}")
        elif self.diagnostics:
            print("SpringConstant not found in wave notes.")

        # Extract Retract Velocity
        v_match = re.search(r"RetractVelocity:\s*([0-9\.\-eE]+)", wave_notes_str, re.IGNORECASE)
        if v_match:
            try:
                retract_velocity_ms = float(v_match.group(1))
                v_read_success = True
                if self.diagnostics: print(f"Read Retract Velocity: {retract_velocity_ms} m/s")
            except ValueError:
                if self.diagnostics: print(f"Could not parse Retract Velocity value: {v_match.group(1)}")
        elif self.diagnostics:
            print("RetractVelocity not found in wave notes.")

        return spring_constant_nm, retract_velocity_ms, k_read_success, v_read_success

    def _process_raw_curve_data(self, y_data, k_spring):  # Renamed k_spr to k_spring
        """
        Processes raw y_data to extract approach and retract curves,
        corrects baseline, and converts to force vs. separation.
        This version is based on the user-provided snippet.
        """
        # Initial checks for y_data
        if y_data is None or not isinstance(y_data, np.ndarray):
            if self.diagnostics: print("Error: y_data is None or not a numpy array.")
            return None, None, False

        if y_data.ndim == 0:  # Handle 0-dimensional array
            if self.diagnostics: print("Error: y_data is 0-dimensional.")
            return None, None, False

        min_required_points = 10  # Minimum points to be considered valid data
        if y_data.shape[0] < min_required_points:
            if self.diagnostics: print(
                f"Error: Not enough data points in y_data (found {y_data.shape[0]}, need at least {min_required_points}).")
            return None, None, False

        # *************************************************************
        # Crop the retraction and approach curves
        # *************************************************************
        m, n = y_data.shape if y_data.ndim > 1 else (y_data.shape[0], 1)

        dfl = np.array([])  # Retract curve (deflection-like)
        apr = np.array([])  # Approach curve (deflection-like)

        if n == 1:
            # 1D data: [value]
            # We'll assume this single column is a force or deflection signal.
            # Separation will be a proxy based on index.
            force_col_1d = 0

            # Check for sufficient points after potential slicing
            if m < min_required_points:
                if self.diagnostics: print(f"Error: Not enough data points for 1D processing (found {m}).")
                return None, None, False

            y_col_data = y_data[:, force_col_1d] if y_data.ndim > 1 else y_data  # Handle truly 1D array

            m_max = np.max(y_col_data)
            m_min = y_col_data[-1] if m > 0 else 0  # Handle empty y_col_data case

            # Avoid issues if m_max equals m_min (flat line)
            if m_max == m_min:
                T = 0  # No change, so threshold won't trigger unless data is above m_min
            else:
                T = 0.6 * (m_max - m_min)

            # Extract retraction curve (dfl)
            dfl_list = []
            # Iterate safely, ensure 'i' doesn't go out of bounds
            for i in range(m - 1, -1, -1):
                if (y_col_data[i] - m_min) > T:
                    break  # Found start of "contact" or significant feature from the retract end
                # Using index/m as a proxy for position (extension-like value)
                # Storing [proxy_extension, value]
                dfl_list.append([i / float(m) if m > 0 else 0, y_col_data[i]])
            if dfl_list:
                dfl = np.array(dfl_list)
                if dfl.ndim == 1 and dfl.shape[0] > 0:
                    dfl = dfl.reshape(-1, 2)  # ensure 2D if only one point
                elif dfl.shape[0] == 0:
                    dfl = np.empty((0, 2))
            else:
                dfl = np.empty((0, 2))

            # Extract approach curve (apr)
            apr_list = []
            for i in range(m):
                if (y_col_data[i] - m_min) > T:
                    break  # Found end of "non-contact" or start of significant feature from approach start
                apr_list.append([i / float(m) if m > 0 else 0, y_col_data[i]])
            if apr_list:
                apr = np.array(apr_list)
                if apr.ndim == 1 and apr.shape[0] > 0:
                    apr = apr.reshape(-1, 2)
                elif apr.shape[0] == 0:
                    apr = np.empty((0, 2))
            else:
                apr = np.empty((0, 2))

        else:  # Multi-column data (n > 1)
            if n < 3:  # Need at least extension and force columns (assuming col 0 is index/time)
                if self.diagnostics: print(
                    f"Error: Multi-column data has {n} columns, expected at least 2 (ext, force) or 3 (e.g. Z, Defl, Height). Assuming ext=col 0, force=col 1.")
                if n < 2: return None, None, False  # Cannot proceed
                ext_col = 0
                force_col = 1
            else:  # Assuming specific columns as per the provided snippet
                force_col = 1  # Deflection/Force signal
                ext_col = 2  # Extension/Z-piezo signal

            # Threshold to detect retraction curve based on force_col
            force_signal_data = y_data[:, force_col]
            m_max = np.max(force_signal_data)
            m_min = force_signal_data[-1]  # Last point as minimum reference

            if m_max == m_min:
                M = 0  # Handle flat line case
            else:
                M = m_max - m_min

            T = 0.6 * M  # Ad hoc threshold

            # Extract retraction curve (dfl) from multi-column data
            # Moving from end to start of y_data
            breakpoint_idx_retract = m  # Start assuming all points are part of retract tail
            for i in range(m - 1, -1, -1):
                if (force_signal_data[i] - m_min) > T:
                    breakpoint_idx_retract = i  # This is the point where curve deviates from baseline
                    break

            # Points from breakpoint_idx_retract to m-1 form the initial part of dfl
            # The original code's `j` seems to count points from the end until the break.
            # So, dfl contains `m - breakpoint_idx_retract` points.
            # Let's adjust: if breakpoint_idx_retract is where deviation starts, then
            # points from breakpoint_idx_retract to end of array are part of the "tail".
            # The Matlab code seems to take points from `m-j` to `m`, where `j` is count from end.
            # This corresponds to taking points from `breakpoint_index` (inclusive) to `m-1`.

            num_pts_dfl = m - breakpoint_idx_retract
            if num_pts_dfl <= 0:  # No points identified for dfl, or only one point
                dfl = np.empty((0, 2))
            else:
                dfl = np.zeros((num_pts_dfl, 2))
                k_dfl = 0
                # Iterating from where the significant part of the retract curve starts (closer to contact)
                # The Matlab code: for i = m-j:1:m --> if breakpoint is at `idx`, then `j = m-idx`. So iterate from `idx` to `m`.
                # Python: range(breakpoint_idx_retract, m)
                for idx_orig in range(breakpoint_idx_retract, m):
                    dfl[k_dfl, 0] = y_data[idx_orig, ext_col]  # Extension value
                    dfl[k_dfl, 1] = y_data[idx_orig, force_col]  # Force/Deflection value
                    k_dfl += 1

            # Extract approach curve (apr) from multi-column data
            # Moving from start to end of y_data
            breakpoint_idx_approach = m  # Default if no break found (all data is "baseline")
            for i in range(m):
                if (force_signal_data[
                        i] - m_min) > T:  # Using same m_min, T for consistency. Could re-eval for approach.
                    breakpoint_idx_approach = i  # This is where approach curve starts to deviate
                    break

            num_pts_apr = breakpoint_idx_approach  # Points from 0 to breakpoint_idx_approach-1
            if num_pts_apr <= 0:
                apr = np.empty((0, 2))
            else:
                apr = np.zeros((num_pts_apr, 2))
                k_apr = 0
                for idx_orig in range(num_pts_apr):  # up to (but not including) breakpoint_idx_approach
                    apr[k_apr, 0] = y_data[idx_orig, ext_col]
                    apr[k_apr, 1] = y_data[idx_orig, force_col]
                    k_apr += 1

        # Check if dfl or apr are too short after extraction
        if dfl.shape[0] < min_required_points // 2 and apr.shape[0] < min_required_points // 2:  # Relaxed requirement
            if self.diagnostics: print(
                f"Warning: Retract ({dfl.shape[0]}) or Approach ({apr.shape[0]}) curve too short after cropping.")
            # Allow to proceed if at least one is somewhat valid, but might fail later.
            # If both are critically short, then fail.
            if dfl.shape[0] < 5 and apr.shape[0] < 5:
                return None, None, False

        # *************************************************************
        # Remove offsets to obtain true 0 - Primarily for Retraction (dfl)
        # *************************************************************
        retract_processed_final = None
        if dfl.shape[0] > 5:  # Need some points to process
            dfl_cur = np.flip(dfl[:, 1])  # Flipped deflection values from dfl
            dfl_cur = np.abs(dfl_cur)
            am_dfl = len(dfl_cur)

            # Initial estimate for baseline (yo) and noise (ystd) from the first half of flipped dfl
            idx_half_dfl = max(1, int(am_dfl / 2))  # Ensure at least 1 point
            yo_dfl_baseline = np.mean(dfl_cur[:idx_half_dfl])
            ystd_dfl_baseline = np.std(dfl_cur[:idx_half_dfl])

            # Find "flat region" start (xo_dfl_offset_ext) in dfl based on deviation from baseline
            # xo_dfl_offset_ext is the extension value from 'dfl' where significant force starts
            xo_dfl_offset_ext = dfl[0, 0]  # Default to first extension point if no flat region found
            flat_region_idx_dfl = 0  # Index in 'dfl_cur' (flipped, so corresponds to end of 'dfl')

            for i in range(am_dfl):
                if dfl_cur[i] >= (6 * ystd_dfl_baseline) + yo_dfl_baseline:
                    # dfl_cur is flipped dfl[:,1]. So index 'i' in dfl_cur corresponds to 'am_dfl-1-i' in dfl.
                    # dfl's original extension values are in dfl[:,0]
                    xo_dfl_offset_ext = dfl[am_dfl - 1 - i, 0]
                    flat_region_idx_dfl = i  # This 'i' is from start of dfl_cur
                    break

            # Refine baseline using the identified flat region (more accurate yo_dfl_true_offset)
            # The flat region in dfl_cur is from index 0 to flat_region_idx_dfl-1
            # This corresponds to the tail end of the original 'dfl' curve.
            if flat_region_idx_dfl > 0:
                yo_dfl_true_offset_calc_region = dfl_cur[:flat_region_idx_dfl]
            else:  # No clear flat region found, use initial estimate or last few points of dfl
                # Using the end of 'dfl' (which is start of 'dfl_cur')
                num_tail_pts = max(1, min(20, am_dfl // 2))  # Use up to 20 points or half, for baseline
                yo_dfl_true_offset_calc_region = dfl_cur[:num_tail_pts]

            if yo_dfl_true_offset_calc_region.size > 0:
                yo_dfl_true_offset = np.mean(yo_dfl_true_offset_calc_region)  # This is the force offset for dfl
            else:  # Fallback if region is empty
                yo_dfl_true_offset = yo_dfl_baseline

            # Bias corrected dfl data -> dfln
            # dfln will have [original_extension - xo_dfl_offset_ext, original_force - yo_dfl_true_offset]
            # However, the MATLAB code uses 'xo' from approach curve for dfl bias. Let's re-check snippet.
            # Snippet: `bias = np.column_stack((np.ones(md) * xo, np.ones(md) * yo))`
            # `xo` is from approach curve processing, `yo` seems to be from `dfl` processing here. This is a bit mixed.
            # Let's assume `xo_contact_point_ext` is determined from the approach curve for separation alignment.
            # And `yo_dfl_true_offset` is the force offset for the retract curve.

            # Find "knee point" (contact point extension) on approach curve (apr) for xo_contact_point_ext
            xo_contact_point_ext = apr[0, 0] if apr.shape[0] > 0 else 0.0  # Default
            yo_apr_baseline_offset = apr[0, 1] if apr.shape[0] > 0 else 0.0  # Default

            if apr.shape[0] > 5:  # Process approach curve for its contact point
                am_apr = apr.shape[0]
                idx_half_apr = max(1, int(am_apr / 2))
                yo_apr_init_baseline = np.mean(apr[:idx_half_apr, 1])
                ystd_apr_init_baseline = np.std(apr[:idx_half_apr, 1])

                flat_region_idx_apr = 0  # Index in 'apr'
                for i in range(am_apr):
                    # For approach, deviation is often a rise if force positive, or drop if negative.
                    # Assuming positive force means contact. The snippet uses `<= (3*ystd)+yo`.
                    # This means it looks for where the curve *enters* the baseline from contact.
                    # For finding first contact, we should look from start of approach where it *leaves* baseline.
                    # Let's adapt: look for force exceeding baseline + noise.
                    if apr[i, 1] >= yo_apr_init_baseline + (3 * ystd_apr_init_baseline):  # Leaves baseline
                        xo_contact_point_ext = apr[i, 0]
                        flat_region_idx_apr = i  # This is the point of contact
                        # The baseline force for approach should be taken before this point.
                        actual_baseline_apr_region = apr[:flat_region_idx_apr, 1]
                        if actual_baseline_apr_region.size > 0:
                            yo_apr_baseline_offset = np.mean(actual_baseline_apr_region)
                        else:  # No clear pre-contact baseline
                            yo_apr_baseline_offset = yo_apr_init_baseline
                        break
                else:  # Loop completed without break, no clear contact found above baseline
                    # Use initial part as baseline for force offset
                    yo_apr_baseline_offset = yo_apr_init_baseline
                    xo_contact_point_ext = apr[-1, 0] if apr.shape[0] > 0 else 0.0  # e.g. end of approach as 'contact'

            # Apply bias correction to dfl (retract curve)
            # Retract extension: dfl[:,0] - xo_contact_point_ext
            # Retract force: dfl[:,1] - yo_dfl_true_offset
            dfln_sep_component = dfl[:, 0] - xo_contact_point_ext
            dfln_force_component = dfl[:, 1] - yo_dfl_true_offset

            # *************************************************************
            # Convert the data to Final Force (N) vs Separation (m) for Retract
            # *************************************************************
            # Force is directly from the bias-corrected, k_spring-scaled deflection
            retract_force_final = dfln_force_component * k_spring

            # Separation calculation from snippet: `force_p[:, 0] = -dfln[:, 0] + dfln[:, 1]`
            # Here `dfln[:,0]` was `original_extension_from_dfl - xo_bias_from_apr`
            # and `dfln[:,1]` was `original_force_from_dfl - yo_bias_from_dfl`
            # This looks like: `-(ext_dfl - xo_apr) + (force_dfl - yo_dfl)`
            # This doesn't seem like a standard separation calculation.
            # Standard AFM: Separation = Z_piezo - Deflection_corrected_for_contact
            # If dfl[:,0] is Z_piezo_retract and dfl[:,1] is Deflection_retract
            # Then `dfln_sep_component` = Z_piezo_retract - Z_piezo_contact_from_approach
            # And `dfln_force_component` = Deflection_retract - Deflection_baseline_retract
            #
            # The separation for WLC is usually tip-sample distance, often made positive.
            # Let's use `retract_separation_final = dfln_sep_component`.
            # The MATLAB `forceP(:,1) = dfln(:,1)-dfln(:,2);` where dfln was [ext,defl_offset_corrected]
            # implied `separation = ext - defl_offset_corrected`.
            # The Python snippet's `force_p[:,0] = -dfln[:,0] + dfln[:,1]` is different.
            # Assuming `dfln_sep_component` (already Z_retract - Z_contact) is the extension-like part.
            # And `dfln_force_component` is the deflection-like part for force.
            #
            # For WLC, we want positive extension. If `dfln_sep_component` represents Z movement away from contact,
            # its sign might need flipping depending on coordinate system.
            # The snippet's final separation: `Separation = Deflection_bias_corrected - Extension_bias_corrected`
            # This is unusual. Let's assume `dfln_sep_component` represents the true tip-sample separation values.
            # WLC typically uses positive extension values. If `dfln_sep_component` decreases during pulling,
            # we might use `max(dfln_sep_component) - dfln_sep_component` or similar.
            #
            # Given the snippet `force_p[:, 0] = -dfln[:, 0] + dfln[:, 1]`, let's use it directly.
            # `dfln[:,0]` here refers to `dfln_sep_component` conceptually.
            # `dfln[:,1]` refers to `dfln_force_component` (before k_spring).
            # This `dfln[:,1]` (deflection) is used in separation.
            retract_separation_final = -dfln_sep_component + dfln_force_component  # As per snippet logic for separation component
            # This combines extension and deflection.

            retract_processed_temp = np.column_stack((retract_separation_final, retract_force_final))

            # Downsample
            if retract_processed_temp.shape[0] >= 2:
                retract_processed_final = retract_processed_temp[::2, :]
            elif retract_processed_temp.shape[0] == 1:  # Keep single point if only one
                retract_processed_final = retract_processed_temp
            else:  # Empty
                retract_processed_final = np.empty((0, 2))
        else:  # dfl too short
            retract_processed_final = np.empty((0, 2))

        # *************************************************************
        # Process Approach curve (apr) similarly for consistency
        # *************************************************************
        approach_processed_final = None
        if apr.shape[0] > 5:  # Need some points to process
            # apr_force_component uses its own baseline (yo_apr_baseline_offset)
            apr_force_component = apr[:, 1] - yo_apr_baseline_offset
            # apr_sep_component uses the same contact point extension from approach processing
            apr_sep_component = apr[:, 0] - xo_contact_point_ext

            approach_force_final = apr_force_component * k_spring
            # Applying similar logic for separation as per snippet's processing of dfln
            approach_separation_final = -apr_sep_component + apr_force_component

            approach_processed_temp = np.column_stack((approach_separation_final, approach_force_final))

            if approach_processed_temp.shape[0] >= 2:
                approach_processed_final = approach_processed_temp[::2, :]
            elif approach_processed_temp.shape[0] == 1:
                approach_processed_final = approach_processed_temp
            else:
                approach_processed_final = np.empty((0, 2))
        else:  # apr too short
            approach_processed_final = np.empty((0, 2))

        success_flag = True
        if (retract_processed_final is None or retract_processed_final.shape[0] == 0) and \
                (approach_processed_final is None or approach_processed_final.shape[0] == 0):
            success_flag = False  # Failed if both are empty
            if self.diagnostics: print("Processing resulted in empty retract and approach curves.")

        return retract_processed_final, approach_processed_final, success_flag

    def _process_test(self, y_data, k_spr):
        # *************************************************************
        # Crop the retraction and approach curves
        # *************************************************************
        # Get dimensions of data
        m, n = y_data.shape if len(y_data.shape) > 1 else (len(y_data), 1)

        # If data is 1D, we need to adjust our approach
        if n == 1:
            # Simplified approach for 1D data
            # We'll assume just a single force column
            force_col = 0

            # Threshold to detect retraction curve
            m_max = np.max(y_data[:, force_col])
            m_min = y_data[-1, force_col]
            T = 0.6 * (m_max - m_min)  # Ad hoc threshold

            # Extract retraction curve
            dfl = []
            for i in range(m - 1, -1, -1):
                if (y_data[i, force_col] - m_min) > T:
                    break
                dfl.append([i / m, y_data[i, force_col]])  # Using index/m as a proxy for position

            dfl = np.array(dfl)

            # Extract approach curve
            apr = []
            for i in range(m):
                if (y_data[i, force_col] - m_min) > T:
                    break
                apr.append([i / m, y_data[i, force_col]])

            apr = np.array(apr)
        else:
            # Multi-column data - we use specific columns for deflection and extension
            # In this case, columns should be:
            # Column 0: Time or index
            # Column 1: Deflection or force
            # Column 2: Extension or separation
            force_col = 1  # Deflection column
            ext_col = 2  # Extension column

            # Threshold to detect retraction curve
            m_max = np.max(y_data[:, force_col])
            m_min = y_data[-1, force_col]
            M = m_max - m_min
            T = 0.6 * M  # Ad hoc threshold

            # Extract retraction curve (moving from end to start)
            j = 0
            breakpoint_index = 0
            for i in range(m - 1, -1, -1):
                j += 1
                if (y_data[i, force_col] - m_min) > T:
                    breakpoint_index = i
                    break

            dfl = np.zeros((j, 2))
            k = 0
            for i in range(breakpoint_index, m):
                dfl[k, 0] = y_data[i, ext_col]  # Extension
                dfl[k, 1] = y_data[i, force_col]  # Force/Deflection
                k += 1

            # Extract approach curve (moving from start to end)
            j = 0
            for i in range(m):
                j += 1
                if (y_data[i, force_col] - m_min) > T:
                    breakpoint_index = i
                    break

            apr = np.zeros((j, 2))
            k = 0
            for i in range(j):
                apr[k, 0] = y_data[i, ext_col]  # Extension
                apr[k, 1] = y_data[i, force_col]  # Force/Deflection
                k += 1

        # *************************************************************
        # Remove offsets to obtain true 0
        # *************************************************************
        # This section handles baseline correction and identifies significant deflection points

        # Flip deflection for processing
        dfl_cur = np.flip(dfl[:, 1])
        dfl_cur = np.abs(dfl_cur)
        am = len(dfl_cur)

        # Calculate mean and std of initial portion (baseline)
        yo = np.mean(dfl_cur[:int(am / 2)])
        ystd = np.std(dfl_cur[:int(am / 2)])

        # Find "flat region" where deflection starts to be significant
        flat_region = 1
        xo = 0
        for i in range(am):
            if dfl_cur[i] >= (6 * ystd) + yo:
                xo = dfl[i, 0]
                flat_region = i
                break

        # Repeat with more accurate flat region
        yo = np.mean(dfl_cur[:flat_region])
        ystd = np.std(dfl_cur[:flat_region])

        flat_region = 1
        for i in range(am):
            if dfl_cur[i] >= (6 * ystd) + yo:
                xo = dfl[i, 0]
                flat_region = i
                break

        # Calculate final baseline values
        yo = np.mean(dfl[am - flat_region + 100:, 1]) if am - flat_region + 100 < am else np.mean(dfl[-20:, 1])
        ystd = np.std(dfl[am - flat_region + 100:, 1]) if am - flat_region + 100 < am else np.std(dfl[-20:, 1])

        yo_dfl = yo
        ystd_dfl = ystd

        # Find "knee point" on approach curve
        am, _ = apr.shape
        yo = np.mean(apr[:int(am / 2), 1])
        ystd = np.std(apr[:int(am / 2), 1])

        flat_region = 1
        for i in range(am):
            if apr[i, 1] <= (3 * ystd) + yo:
                xo = apr[i, 0]
                flat_region = i

        # Repeat with more accurate flat region
        yo = np.mean(apr[:flat_region, 1])
        ystd = np.std(apr[:flat_region, 1])

        flat_region = 1
        for i in range(am):
            if apr[i, 1] <= (3 * ystd) + yo:
                xo = apr[i, 0]
                flat_region = i

        # Bias corrected data
        md, _ = dfl.shape
        bias = np.column_stack((np.ones(md) * xo, np.ones(md) * yo))

        dfln = dfl - bias  # Data with true 0

        # *************************************************************
        # Convert the data to Force (N) vs Separation (m)
        # *************************************************************
        force_p = np.zeros_like(dfln)
        force_p[:, 1] = dfln[:, 1] * k_spr  # Convert deflection to force
        force_p[:, 0] = -dfln[:, 0] + dfln[:, 1]  # Calculate true separation

        # Downsample to reduce noise and computational load
        dfln = force_p[::2]  # Take every 4th point
        return dfln, None, 1

    def _find_regions_of_interest_matlab(self, force_curve_data, k_spring):
        """
        Identifies regions of interest (potential unfolding events) in the force curve.
        Implementation based on the MATLAB code for identifying regions with significant
        force changes that can be fit with WLC model.

        Parameters:
        force_curve_data (np.array): Processed force curve [separation (m), force (N)].
                                     Typically the retract curve.
        k_spring (float): Cantilever spring constant (N/m), used for threshold calculations.

        Returns:
        list: A list of np.arrays, where each array is an ROI [separation (m), force (N)].
        """
        # Check if we have enough data
        if force_curve_data is None or len(force_curve_data) < 20:
            return []

        # Extract separation and force
        dfln = force_curve_data

        # Apply median filter for initial peak detection
        dfln_f = np.zeros_like(dfln)
        dfln_f[:, 0] = dfln[:, 0]
        dfln_f[:, 1] = signal.medfilt(dfln[:, 1], kernel_size=7)

        # Find peaks (positive peaks) - these represent locations where force increases
        try:
            locs, _ = signal.find_peaks(dfln_f[:, 1], distance=10)
            if len(locs) > 0:
                # Calculate peak prominences
                p = signal.peak_prominences(dfln_f[:, 1], locs)[0]
            else:
                p = np.array([])
        except Exception as e:
            print(f"Error finding positive peaks: {e}")
            locs = np.array([])
            p = np.array([])

        # Find valleys (negative peaks) - these represent locations where force decreases
        try:
            neg_locs, _ = signal.find_peaks(-dfln_f[:, 1], distance=20)
            if len(neg_locs) > 0:
                # Calculate valley prominences
                np_vals = signal.peak_prominences(-dfln_f[:, 1], neg_locs)[0]
            else:
                np_vals = np.array([])
        except Exception as e:
            print(f"Error finding negative peaks: {e}")
            neg_locs = np.array([])
            np_vals = np.array([])

        if self.diagnostics:
            print(f"Found {len(locs)} positive peaks and {len(neg_locs)} negative peaks")

        # Apply a second, milder median filter for detailed analysis (like dflni_f in MATLAB)
        dflni = dfln.copy()  # Start with the original data
        dflni_f = np.zeros_like(dflni)
        dflni_f[:, 0] = dflni[:, 0]
        dflni_f[:, 1] = signal.medfilt(dflni[:, 1], kernel_size=3)

        # Calculate noise level from approach curve or baseline (assumed passed separately in MATLAB)
        # For now, estimate from initial part of the curve
        yfstd = np.std(dflni_f[:min(20, len(dflni_f)), 1])

        # Find important peaks based on prominence/median(prominences) like in MATLAB
        # Threshold values as in MATLAB
        t_pp = 7  # Threshold for positive peaks' prominence
        t_np = 7  # Threshold for negative peaks' prominence

        # Select important peaks based on prominence
        locs_imp = []
        if len(p) > 0:
            med_p = np.median(p)
            for i in range(len(p)):
                if p[i] > t_pp * med_p:
                    locs_imp.append(locs[i])

        # Select important valleys based on prominence
        nlocs_imp = []
        if len(np_vals) > 0:
            med_np = np.median(np_vals)
            for i in range(len(np_vals)):
                if np_vals[i] > t_np * med_np:
                    nlocs_imp.append(neg_locs[i])

        if self.diagnostics:
            print(f"Selected {len(locs_imp)} important positive peaks")
            print(f"Selected {len(nlocs_imp)} important negative peaks")

        # Interlace positive and negative important peaks to identify regions
        all_roi = []

        # Create a list of [location, type] where type=1 for positive peak, type=-1 for negative peak
        all_imp_locs = []
        for loc in locs_imp:
            all_imp_locs.append([loc, 1])
        for loc in nlocs_imp:
            all_imp_locs.append([loc, -1])

        # Sort by location
        all_imp_locs = sorted(all_imp_locs, key=lambda x: x[0])

        # Following MATLAB's logic for identifying ROIs
        roi = []
        for i in range(len(all_imp_locs) - 1):
            loc1 = all_imp_locs[i][0]
            loc2 = all_imp_locs[i + 1][0]
            check_diff = all_imp_locs[i][1] - all_imp_locs[i + 1][1]

            # MATLAB looks for checkDiff == 2 which means pos peak followed by neg peak
            if check_diff == 2:
                # Check if pk-pk distance is greater than threshold (like MATLAB)
                roi_data = dflni[loc1:loc2, :]
                roi_len = len(roi_data)

                if roi_len > 5:  # Ensure ROI has enough points
                    roi_pp = abs(roi_data[0, 1] - roi_data[-1, 1])
                    # MATLAB uses yfstd * 4 * Kspr for threshold
                    roi_thresh = yfstd * 4 * k_spring

                    if roi_pp > roi_thresh:
                        all_roi.append(roi_data)

        # Logic as in MATLAB for final roi selection
        if len(all_imp_locs) > 0 and all_imp_locs[0][1] == 1 and len(all_roi) >= 2:
            roi = all_roi[1:-1]
        else:
            roi = all_roi[:-1] if len(all_roi) > 0 else all_roi

        if self.diagnostics:
            print(f"Identified {len(roi)} regions of interest")
            if self.make_plots and len(roi) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(dflni[:, 0], dflni[:, 1], 'c-', label='Processed Force')
                plt.plot(dflni_f[:, 0], dflni_f[:, 1], 'b-', label='Filtered Force')

                if len(locs_imp) > 0:
                    plt.plot(dflni[locs_imp, 0], dflni[locs_imp, 1], "rs", label="Important Peaks")

                if len(nlocs_imp) > 0:
                    plt.plot(dflni[nlocs_imp, 0], dflni[nlocs_imp, 1], "bs", label="Important Valleys")

                # Plot all ROIs
                roi_combined = np.vstack(roi) if len(roi) > 0 else np.array([])
                if len(roi_combined) > 0:
                    plt.plot(roi_combined[:, 0], roi_combined[:, 1], "rx", label="ROI Points")

                plt.xlabel("Separation (m)")
                plt.ylabel("Force (N)")
                plt.title("Regions of Interest for WLC Fitting")
                plt.legend()
                plt.show()

        return roi

    def analyze_single_curve(self, ibw_file_path):
        """
        Analyzes a single force curve from an .ibw file.

        Parameters:
        ibw_file_path (str): Path to the .ibw file.

        Returns:
        tuple: (all_wlc_fits, all_wlc_params, processed_curve, fig_handle)
               all_wlc_fits (list): List of WLC fit data arrays for each successful ROI.
                                    Each array is [separation (m), fitted_force (N)].
               all_wlc_params (list): List of dictionaries, each containing parameters
                                      for a successful WLC fit (Lc, P, rupture_L, rupture_F, fit_quality_percent, velocity).
               processed_curve (list): First row is Separation, Second row is Force
               fig_handle (matplotlib.figure.Figure or None): Handle to the generated plot if make_plots is True.
        """
        all_wlc_fits = []
        all_wlc_params = []
        fig_handle = None

        try:
            igor_wave = ibw_load(ibw_file_path)
        except Exception as e:
            print(f"Error loading IBW file {ibw_file_path}: {e}")
            return all_wlc_fits, all_wlc_params, fig_handle

        wave_data = igor_wave['wave']['wData']
        try:
            wave_notes_bytes = igor_wave['wave']['note']
            wave_notes = str(wave_notes_bytes)
        except Exception as e:
            if self.diagnostics: print(f"Could not decode wave notes for {ibw_file_path}: {e}. Using empty notes.")
            wave_notes = ""

        # Extract parameters like spring constant and velocity
        k_spring, velocity_ms, _, _ = self._extract_parameters_from_notes(wave_notes)

        # Process raw data to get force-separation curves (retract and approach)
        # We primarily use the retract curve for WLC fitting of unfolding events.
        retract_curve_processed, approach_curve_processed, success = self._process_test(wave_data, k_spring)

        if not success or retract_curve_processed is None or len(retract_curve_processed) == 0:
            print(f"Failed to process raw data for {ibw_file_path}.")
            return all_wlc_fits, all_wlc_params, fig_handle

        # Identify ROIs from the processed retract curve
        rois = self._find_regions_of_interest_matlab(retract_curve_processed, k_spring)

        if not rois:
            if self.diagnostics: print(f"No regions of interest found in {ibw_file_path}.")
            # Still, we might want to plot the processed curve if make_plots is True
            if self.make_plots:
                fig_handle, _ = plt.subplots(figsize=(10, 6))
                plt.plot(retract_curve_processed[:, 0] * 1e9, retract_curve_processed[:, 1] * 1e12,
                         label='Processed Retract Curve')
                if approach_curve_processed is not None:
                    plt.plot(approach_curve_processed[:, 0] * 1e9, approach_curve_processed[:, 1] * 1e12,
                             label='Processed Approach Curve', linestyle='--')
                plt.xlabel("Separation (nm)")
                plt.ylabel("Force (pN)")
                plt.title(f"Processed Force Curve: {os.path.basename(ibw_file_path)} (No ROIs)")
                plt.legend()
                plt.grid(True)
            return all_wlc_fits, all_wlc_params, retract_curve_processed, fig_handle

        # Perform WLC fitting for each ROI
        for i, roi_data in enumerate(rois):
            sep_data = roi_data[:, 0]
            force_data = roi_data[:, 1]  # Force data should be positive for WLC fitting

            # Ensure separation is positive and generally increasing for WLC model
            # If sep_data is negative or decreasing, adjust it.
            # Assuming sep_data from _find_regions_of_interest is already appropriate (positive extension)
            if sep_data[0] > sep_data[-1] and np.mean(np.diff(sep_data)) < 0:  # If separation is decreasing
                sep_data = sep_data[::-1]  # Reverse to make it increasing
                force_data = force_data[::-1]

            # Ensure separation starts near zero for the ROI
            sep_data = sep_data - sep_data[0]

            # Initial guess for WLC parameters [Lc, P]
            # Lc: Max separation in ROI, or slightly larger.
            # P: Persistence length, typically 0.3 nm for dsDNA, ~1 nm for some proteins.
            #    Can estimate P ~ Lc / (number of expected Kuhn segments, e.g. 10-1000)
            initial_lc = np.max(sep_data) * 1.1
            initial_p = initial_lc / 100.0  # Heuristic, P << Lc
            if initial_p < 1e-10: initial_p = 1e-10  # Avoid zero persistence length
            if initial_lc < 1e-9: initial_lc = 1e-9  # Avoid too small contour length

            initial_params_wlc = [initial_lc, initial_p]

            if self.diagnostics:
                print(f"ROI {i + 1}: Initial Lc={initial_lc * 1e9:.2f} nm, P={initial_p * 1e9:.2f} nm")

            try:
                # Bounds for optimization to keep parameters physical
                # Lc bounds: e.g., from min_sep_in_roi to 2*max_sep_in_roi
                # P bounds: e.g., from 0.01 nm to 50 nm (or related to Lc)
                min_lc_bound = np.max(sep_data) * 0.8
                max_lc_bound = np.max(sep_data) * 2.0
                min_p_bound = 0.01e-9  # 0.01 nm
                max_p_bound = 50e-9  # 50 nm

                bounds = [(min_lc_bound, max_lc_bound), (min_p_bound, max_p_bound)]

                result = optimize.minimize(
                    self._get_cost_wlc,
                    initial_params_wlc,
                    args=(sep_data, self.temperature_k, force_data),
                    method='Nelder-Mead',  # Robust but can be slow. 'L-BFGS-B' for bounds.
                    # method='L-BFGS-B', # Supports bounds
                    # bounds=bounds,
                    options={'maxiter': 1000, 'maxfev': 2000, 'xtol': 1e-8, 'ftol': 1e-8, 'adaptive': True}
                    # Adjust options
                )

                if result.success:
                    lc_fit, p_fit = result.x

                    # Check if fitted parameters are within reasonable physical limits
                    if not (min_lc_bound * 0.8 < lc_fit < max_lc_bound * 1.2 and \
                            min_p_bound * 0.8 < p_fit < max_p_bound * 1.2):
                        if self.diagnostics: print(
                            f"ROI {i + 1} fit parameters out of expected range. Lc={lc_fit * 1e9:.2f}nm, P={p_fit * 1e9:.2f}nm. Skipping.")
                        continue

                    fitted_force = self._get_force_wlc(lc_fit, p_fit, sep_data, self.temperature_k)

                    # Calculate fit quality (e.g., R-squared or 1 - residual_variance/data_variance)
                    # Simpler: 1 - (norm(residual)/norm(data))
                    residuals = force_data - fitted_force
                    data_norm = np.linalg.norm(force_data - np.mean(force_data))  # norm relative to mean
                    if data_norm < 1e-15:  # Avoid division by zero for flat data
                        fit_quality_percent = 0.0 if np.linalg.norm(residuals) > 1e-15 else 100.0
                    else:
                        fit_quality_percent = max(0.0, 1.0 - (np.linalg.norm(residuals) / data_norm)) * 100.0

                    if self.diagnostics:
                        print(
                            f"ROI {i + 1}: Lc={lc_fit * 1e9:.2f} nm, P={p_fit * 1e9:.2f} nm, Fit Quality={fit_quality_percent:.1f}%")

                    if fit_quality_percent >= self.fit_threshold_percent:
                        rupture_length_m = np.max(sep_data)  # Max extension in ROI
                        rupture_force_n = np.max(force_data)  # Max force in ROI (at rupture)

                        wlc_fit_data = np.column_stack((sep_data, fitted_force))
                        all_wlc_fits.append(wlc_fit_data)

                        params_dict = {
                            'Lc_m': lc_fit,
                            'P_m': p_fit,
                            'rupture_L_m': rupture_length_m,
                            'rupture_F_N': rupture_force_n,
                            'fit_quality_percent': fit_quality_percent,
                            'velocity_ms': velocity_ms,
                            'roi_index': i
                        }
                        all_wlc_params.append(params_dict)
                else:
                    if self.diagnostics: print(f"WLC fitting failed for ROI {i + 1}: {result.message}")

            except Exception as e:
                print(f"Error during WLC fitting for ROI {i + 1} in {ibw_file_path}: {e}")
                if self.diagnostics:
                    import traceback
                    traceback.print_exc()

        # Plotting
        if self.make_plots:
            num_subplots = 1
            fig_handle, ax = plt.subplots(num_subplots, 1, figsize=(12, num_subplots * 5),
                                          sharex=False)  # sharex=True if all on same x-axis
            if num_subplots == 1:
                ax_list = [ax]
            else:
                ax_list = ax.ravel()

            # Plot 1: Full processed curve and ROIs
            ax1 = ax_list[0]
            if retract_curve_processed is not None:
                ax1.plot(retract_curve_processed[:, 0] * 1e9, retract_curve_processed[:, 1] * 1e12,
                         label='Processed Retract Curve', color='lightblue', linewidth=1)

            # Highlight identified ROIs
            for i, roi_data in enumerate(rois):
                ax1.plot(roi_data[:, 0] * 1e9, roi_data[:, 1] * 1e12,
                         label=f'ROI {i + 1}', linestyle='--', linewidth=1.5)

            # Plot successful WLC fits
            for i, fit_data in enumerate(all_wlc_fits):
                params = all_wlc_params[i]
                ax1.plot(fit_data[:, 0] * 1e9, fit_data[:, 1] * 1e12,
                         label=f'WLC Fit ROI {params["roi_index"] + 1} (Lc: {params["Lc_m"] * 1e9:.1f}nm, P: {params["P_m"] * 1e9:.1f}nm, Q: {params["fit_quality_percent"]:.0f}%)',
                         linewidth=2)

            ax1.set_xlabel("Separation (nm)")
            ax1.set_ylabel("Force (pN)")
            ax1.set_title(f"Force Curve Analysis: {os.path.basename(ibw_file_path)}")
            ax1.legend(fontsize='small', loc='best')
            ax1.grid(True)

            plt.tight_layout()
            # plt.show() # Show plot immediately, or let the calling function handle it.

        return all_wlc_fits, all_wlc_params, retract_curve_processed, fig_handle

    def batch_process_ibw_files(self, directory_path, output_summary_file="wlc_fit_summary.csv"):
        """
        Processes all .ibw files in a given directory.

        Parameters:
        directory_path (str): The path to the directory containing .ibw files.
        output_summary_file (str): Path to save the summary CSV file.
        """
        ibw_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.ibw')]
        if not ibw_files:
            print(f"No .ibw files found in directory: {directory_path}")
            return

        print(f"Found {len(ibw_files)} .ibw files to process in {directory_path}")

        all_results_for_summary = []

        for filename in ibw_files:
            file_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {filename}...")

            wlc_fits, wlc_params_list, processed_curve, fig = self.analyze_single_curve(file_path)

            if wlc_params_list:
                print(f"Successfully fitted {len(wlc_params_list)} WLC regions for {filename}.")
                for params in wlc_params_list:
                    result_row = {
                        'filename': filename,
                        'roi_index': params['roi_index'] + 1,
                        'Lc_nm': params['Lc_m'] * 1e9,
                        'P_nm': params['P_m'] * 1e9,
                        'rupture_L_nm': params['rupture_L_m'] * 1e9,
                        'rupture_F_pN': params['rupture_F_N'] * 1e12,
                        'fit_quality_percent': params['fit_quality_percent'],
                        'velocity_um_s': params['velocity_ms'] * 1e6 if params['velocity_ms'] is not None else 'N/A'
                    }
                    all_results_for_summary.append(result_row)
            else:
                print(f"No successful WLC fits for {filename}.")

            if self.make_plots and fig is not None:
                plot_filename = os.path.splitext(filename)[0] + "_analysis.png"
                output_plot_path = os.path.join(directory_path, "analysis_plots")
                if not os.path.exists(output_plot_path):
                    os.makedirs(output_plot_path)
                fig.savefig(os.path.join(output_plot_path, plot_filename))
                print(f"Saved plot to {os.path.join(output_plot_path, plot_filename)}")
                plt.close(fig)  # Close figure to free memory

        # Save summary to CSV
        if all_results_for_summary:
            try:
                import pandas as pd
                summary_df = pd.DataFrame(all_results_for_summary)
                output_csv_path = os.path.join(directory_path, output_summary_file)
                summary_df.to_csv(output_csv_path, index=False)
                print(f"\nBatch processing complete. Summary saved to: {output_csv_path}")
            except ImportError:
                print("\nPandas library not found. Cannot save summary to CSV. Please install pandas.")
                print("Summary data:")
                for row in all_results_for_summary:
                    print(row)
        else:
            print("\nBatch processing complete. No results to summarize.")


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    IBW_FILES_DIRECTORY = "../../data/raw"  # Current directory or specify a path e.g., "C:/data/ibw_files"
    TEMPERATURE_K = 298.15  # Experimental temperature in Kelvin (e.g., 25°C)
    FIT_THRESHOLD = 70.0  # Minimum fit quality percentage for a WLC fit to be considered good
    GENERATE_PLOTS = True  # Set to True to see and save plots for each curve
    ENABLE_DIAGNOSTICS = True  # Set to True for verbose output during processing

    # --- Create Analyzer Instance ---
    analyzer = ForceCurveAnalyzer(
        temperature_k=TEMPERATURE_K,
        fit_threshold_percent=FIT_THRESHOLD,
        make_plots=GENERATE_PLOTS,
        diagnostics=ENABLE_DIAGNOSTICS
    )

    # --- Perform Batch Processing ---
    # Check if the directory exists
    if not os.path.isdir(IBW_FILES_DIRECTORY):
        print(f"Error: Directory not found: {IBW_FILES_DIRECTORY}")
        # Example of how to run a single file if batch directory is not set:
        # print("Attempting to run analysis on a single example file (if present).")
        # example_file = "your_example_file.ibw" # Replace with an actual .ibw file name
        # if os.path.exists(example_file):
        #     fits, params, fig_handle = analyzer.analyze_single_curve(example_file)
        #     if GENERATE_PLOTS and fig_handle:
        #         plt.show() # Show the plot for the single file
        # else:
        #     print(f"Example file {example_file} not found. Please set IBW_FILES_DIRECTORY correctly.")
    else:
        analyzer.batch_process_ibw_files(IBW_FILES_DIRECTORY, output_summary_file="wlc_analysis_summary.csv")

    print("\n--- Analysis Finished ---")