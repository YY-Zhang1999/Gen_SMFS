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
        k_spr_match = re.search(r"SpringConstant\s*([0-9\.\-eE]+)", wave_notes_str, re.IGNORECASE)
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
        v_match = re.search(r"RetractVelocity\s*([0-9\.\-eE]+)", wave_notes_str, re.IGNORECASE)
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

    def _process_raw_curve_data(self, y_data, k_spring):
        """
        Processes raw y_data to extract approach and retract curves,
        corrects baseline, and converts to force vs. separation.

        Parameters:
        y_data (np.array): The raw data array from the IBW file. Expected to have
                           at least 3 columns (e.g., Z-sensor, Deflection, Height).
                           The specific columns used are based on typical AFM setups.
        k_spring (float): Cantilever spring constant (N/m).

        Returns:
        tuple: (dfln_fs_corrected, apr_corrected, success)
               dfln_fs_corrected (np.array): Processed retraction curve [separation (m), force (N)].
                                             Returns None if processing fails.
               apr_corrected (np.array): Processed approach curve [separation (m), force (N)].
                                         Returns None if processing fails.
               success (bool): True if processing was successful.
        """
        if y_data.ndim < 2 or y_data.shape[1] < 3:  # Assuming Height, Deflection, ZSensor (or similar)
            print(f"Error: IBW data has unexpected shape: {y_data.shape}. Expected at least 3 columns.")
            return None, None, False

        # Column indices based on common AFM data formats (e.g., Asylum AFM)
        # Channel 0: Height (Piezo position of the tip relative to a starting point)
        # Channel 1: Deflection (Cantilever deflection signal, proportional to force)
        # Channel 2: ZSensor (Calibrated Z position of the piezo)
        # Separation = ZSensor - Deflection (or Height - Deflection, depending on setup and calibration)
        # Force = Deflection * SpringConstant

        # For simplicity, let's assume:
        # Column 2 (index 2) is Z-position (stage or piezo) -> often 'ZSnsr' or 'Height'
        # Column 1 (index 1) is Deflection signal (Volts or nm) -> often 'Defl'

        # Heuristic to find the turning point (contact point / max force point)
        # This often corresponds to the maximum value in the Z-sensor or deflection channel
        # during the approach/retract cycle. For a standard force curve, the data points
        # are often ordered from approach start to retract end.

        # A more robust way is to look at the Z-sensor data (assuming it's monotonic for approach/retract segments)
        # Find the point where Z-sensor changes direction or reaches its extremum.
        # For now, let's assume data is [approach_points, retract_points] concatenated.
        # The midpoint is a common, though not always accurate, separator.

        num_points = y_data.shape[0]
        # A simple way to separate approach and retract is by looking for the point of maximum Z travel or deflection
        # Let's assume column 2 is the Z-sensor data which extends and then retracts.
        # The point of maximum extension is often the switch from approach to retract.
        if num_points < 10:  # Not enough data
            print("Error: Not enough data points in the curve.")
            return None, None, False

        # Find the approximate contact point or turning point
        # In many systems, the Z-sensor (col 2) or Height (col 0) shows the piezo movement.
        # The deflection (col 1) shows the cantilever bending.
        # We'll use the Z-sensor (y_data[:, 2]) to find the turning point, assuming it dictates the overall motion.
        # If Z-sensor data is not available or reliable, one might use the deflection signal.

        # Simplified split: find the index of the maximum value in the Z-sensor or Height column
        # This often marks the end of approach / start of retract.
        # This is a common simplification. A more sophisticated method would analyze velocities or use metadata if available.
        # Let's use column 2 (Z-sensor) as primary, fallback to column 0 (Height) if needed.

        z_sensor_col_idx = 2
        if y_data.shape[1] <= z_sensor_col_idx:
            z_sensor_col_idx = 0  # Fallback to Height if ZSensor column is not present
            if y_data.shape[1] <= z_sensor_col_idx:  # Still not enough columns
                print("Error: Not enough columns for Z-sensor or Height data.")
                return None, None, False

        # Find the turning point based on the Z-sensor (or Height) data
        # This point is often where the piezo reverses direction.
        # It's usually the point of maximum extension of the Z-piezo.
        # If data is ordered approach -> retract, this is the max Z value.
        # If data is ordered retract -> approach, this is the min Z value.
        # Assuming standard approach then retract:
        turn_point_idx = np.argmax(y_data[:, z_sensor_col_idx])

        if turn_point_idx == 0 or turn_point_idx == num_points - 1:
            # If max is at the start/end, it might be a monotonic curve or bad data.
            # As a fallback, use midpoint, but this is less reliable.
            if self.diagnostics: print("Warning: Z-sensor maximum at curve boundary. Using midpoint to split.")
            turn_point_idx = num_points // 2

        approach_raw = y_data[:turn_point_idx, :]
        retract_raw = y_data[turn_point_idx:, :]

        if approach_raw.shape[0] < 5 or retract_raw.shape[0] < 5:
            print("Error: Not enough data points in approach or retract segments after splitting.")
            return None, None, False

        # Extract Deflection (col 1) and Z-sensor/Height (col 2 or 0) for each segment
        # Retraction curve data (often plotted with Z decreasing)
        # We usually analyze the retract curve for unfolding events.
        # The MATLAB code flips the retract curve (dfl_cur = flip(dfl(:,2));)
        # Here, we'll keep the natural order but select the appropriate columns.

        # Retraction:
        # Separation_retract = ZSensor_retract - Deflection_retract
        # Force_retract = Deflection_retract * k_spring
        defl_retract = retract_raw[:, 1]
        zsens_retract = retract_raw[:, z_sensor_col_idx]
        sep_retract = zsens_retract - defl_retract  # True tip-sample separation
        force_retract = defl_retract * k_spring

        # Approach:
        defl_approach = approach_raw[:, 1]
        zsens_approach = approach_raw[:, z_sensor_col_idx]
        sep_approach = zsens_approach - defl_approach
        force_approach = defl_approach * k_spring

        # Baseline correction (similar to MATLAB script's offset removal)
        # Use the initial part of the approach curve (non-contact) to find force offset
        # Or the final part of the retract curve (non-contact)

        # For retract curve baseline: use the tail end (furthest from contact)
        if len(force_retract) > 20:  # Ensure enough points for baseline
            baseline_force_offset_retract = np.mean(force_retract[-int(len(force_retract) * 0.2):])  # Use last 20%
        elif len(force_retract) > 0:
            baseline_force_offset_retract = np.mean(force_retract)
        else:
            baseline_force_offset_retract = 0.0

        force_retract_corrected = force_retract - baseline_force_offset_retract

        # For approach curve baseline: use the initial part
        if len(force_approach) > 20:
            baseline_force_offset_approach = np.mean(force_approach[:int(len(force_approach) * 0.2)])  # Use first 20%
        elif len(force_approach) > 0:
            baseline_force_offset_approach = np.mean(force_approach)
        else:
            baseline_force_offset_approach = 0.0
        force_approach_corrected = force_approach - baseline_force_offset_approach

        # Contact point determination for separation offset
        # A common method: find where the force on approach starts to deviate from baseline
        # Or, find the point of maximum adhesion on retract (if present and significant)
        # The MATLAB code has a more complex 'xo' finding logic.
        # Simplified: use the point of maximum force on approach as contact.
        # Or use the point where force_approach significantly deviates from zero.

        contact_point_sep_offset = 0.0
        if len(force_approach_corrected) > 10:
            # Find where approach force rises above noise (e.g., 3*std of baseline)
            noise_level_approach = np.std(force_approach_corrected[:int(len(force_approach_corrected) * 0.2)])
            contact_indices = np.where(np.abs(force_approach_corrected) > 3 * noise_level_approach)[0]
            if len(contact_indices) > 0:
                # First significant contact point on approach
                contact_idx_approach = contact_indices[0]
                contact_point_sep_offset = sep_approach[contact_idx_approach]
            else:  # If no clear contact, use max force point on approach
                contact_idx_approach = np.argmax(np.abs(force_approach_corrected))
                contact_point_sep_offset = sep_approach[contact_idx_approach]

        sep_retract_corrected = sep_retract - contact_point_sep_offset
        sep_approach_corrected = sep_approach - contact_point_sep_offset

        # Combine into [separation, force] arrays
        # The MATLAB code processes `dfln` which is the retract curve.
        # And `apr` which is the approach curve.
        # We return the corrected retract and approach curves.
        # Typically, WLC fitting is done on the retract curve.

        # The MATLAB script uses `dfln` which is effectively the processed retract curve.
        # `dfln(:,1)` is separation, `dfln(:,2)` is force.
        # It also seems to use negative separation for plotting/fitting in some contexts.
        # For WLC, extension (separation) is usually positive.

        # We want the retract curve for WLC analysis (pulling events)
        # Ensure separation is generally increasing during "pulling" part of retract.
        # AFM retract curves often start at high Z (large separation) and move to lower Z (smaller separation).
        # We need to present extension (separation) as positive and increasing for WLC.
        # The MATLAB 'dfln' seems to have separation values that can be negative after offset.
        # Let's ensure our 'separation' for WLC is the extension from the contact point.
        # If sep_retract_corrected is decreasing (as Z moves from max to min), we might need to flip it or use absolute values carefully.

        # The MATLAB 'AnalyseForceCurves.m' does:
        # forceP(:,1) = dfln(:,1)-dfln(:,2); % This is separation Z - deflection (already done for sep_retract)
        # forceP(:,2) = dfln(:,2)*Kspr;    % This is force (already done for force_retract_corrected)
        # Then `dfln = downsample(forceP, 4);`
        # So, the `dfln` used for peak finding is [separation, force] from the retract curve.

        # Let's ensure our `retract_processed` has separation increasing for typical WLC fitting.
        # If sep_retract_corrected[0] > sep_retract_corrected[-1], it means separation decreases during retract.
        # This is typical if Z moves from far to near.
        # For WLC, we usually consider extension from a zero point.
        # The MATLAB 'dflni' (region of interest) has separation values.
        # And 'fData = -(roi{i}(:,2))', so force is made positive for WLC fitting.

        retract_processed = np.column_stack(
            (-sep_retract_corrected, force_retract_corrected))  # Make separation positive extension
        approach_processed = np.column_stack((-sep_approach_corrected, force_approach_corrected))

        # Downsample, similar to MATLAB's `downsample(forceP, 4)`
        # Downsampling factor - adjust as needed
        ds_factor = 2  # MATLAB uses 4 for 'forceP' then later medfilt with 3 or 7
        if len(retract_processed) > ds_factor * 10:  # Only downsample if enough points
            retract_processed = retract_processed[::ds_factor, :]
        if len(approach_processed) > ds_factor * 10:
            approach_processed = approach_processed[::ds_factor, :]

        return retract_processed, approach_processed, True

    def _find_regions_of_interest(self, force_curve_data, k_spring):
        """
        Identifies regions of interest (potential unfolding events) in the force curve.
        This is a complex part of the MATLAB script involving peak finding, prominence,
        and curvature analysis. This Python version will be a simplified adaptation.

        Parameters:
        force_curve_data (np.array): Processed force curve [separation (m), force (N)].
                                     Typically the retract curve.
        k_spring (float): Cantilever spring constant (N/m), used for context if needed.

        Returns:
        list: A list of np.arrays, where each array is an ROI [separation (m), force (N)].
        """
        rois = []
        if force_curve_data is None or len(force_curve_data) < 20:
            return rois

        separation = force_curve_data[:, 0]
        force = force_curve_data[:, 1]

        # Apply a median filter to smooth the force data, similar to MATLAB's medfilt1
        # The MATLAB script uses medfilt1(dfln(:,2), 7) for initial peak finding (dfln_f)
        # and then medfilt1(dflni(:,2), 3) for ROIs (dflni_f)
        force_filtered = signal.medfilt(force, kernel_size=5)  # Kernel size can be tuned

        # Find peaks (representing rupture events or feature detachments)
        # The MATLAB script looks for valleys in deflection (npksf), then uses their prominences.
        # Here, we'll look for peaks in force if force is positive during pulling.
        # If your force signal is negative during pulling, you'd find peaks in -force.
        # Assuming positive force for pulling events:

        # Parameters for peak finding (can be tuned)
        # min_peak_height = np.std(force_filtered) # Example: force must be at least 1 std dev
        # min_peak_prominence = np.std(force_filtered) * 0.5 # Example: prominence
        # min_peak_distance = int(0.05 * len(force_filtered)) # Min distance between peaks (e.g., 5% of curve length)

        # A simpler approach than the full MATLAB peak analysis:
        # Look for significant "sawtooth" patterns. A peak followed by a drop.
        # The MATLAB script uses `findpeaks` on force `pks, locs` and on negative force `npks, nlocs`
        # Then it combines these to find regions `allroi`.
        # `checkDiff == 2` implies a sequence of positive peak then negative peak (or vice-versa in their logic).

        # Let's try to find "unfolding" peaks directly in the force.
        # These are typically sharp drops after a rise in force.
        # So, we are looking for peaks in force, and the region before the peak is the WLC segment.

        # Find significant positive peaks in the filtered force data.
        # These peaks often mark the point of rupture or full extension of a domain.
        # The region leading up to the peak is what we fit with WLC.

        # Adjust these based on typical force values (e.g., tens to hundreds of pN)
        # height_threshold_N = 20e-12 # Minimum height of a peak in Newtons (e.g., 20 pN)
        # prominence_threshold_N = 15e-12 # Minimum prominence in Newtons

        # Use relative thresholds based on data noise/scale
        noise_level = np.std(force_filtered[:len(force_filtered) // 5])  # Estimate noise from initial part
        height_threshold_N = max(5e-12, 3 * noise_level)  # e.g., 5 pN or 3x noise
        prominence_threshold_N = max(5e-12, 2 * noise_level)
        min_dist_points = 10  # Minimum number of data points between peaks

        peak_indices, properties = signal.find_peaks(force_filtered,
                                                     height=height_threshold_N,
                                                     prominence=prominence_threshold_N,
                                                     distance=min_dist_points)

        if self.diagnostics and len(peak_indices) > 0:
            print(f"Found {len(peak_indices)} potential rupture peaks.")
            if self.make_plots:
                plt.figure(figsize=(10, 6))
                plt.plot(separation, force, 'c-', label='Raw Force')
                plt.plot(separation, force_filtered, 'b-', label='Filtered Force')
                plt.plot(separation[peak_indices], force_filtered[peak_indices], "rx", label="Detected Peaks")
                plt.xlabel("Separation (m)")
                plt.ylabel("Force (N)")
                plt.title("Peak Detection for ROI Identification")
                plt.legend()
                plt.show()

        # Define ROIs: from the valley before a peak up to the peak.
        # The MATLAB code uses `allImpLocs` with +1 and -1 to mark peak/valley types.
        # `checkDiff == 2` means `locs_imp` (positive peak) then `nlocs_imp` (negative peak/valley).
        # This corresponds to an unfolding event: force rises (to locs_imp), then drops (to nlocs_imp).
        # The ROI is dflni(loc1:loc2), where loc1 is the start of the rise, loc2 is the peak.

        # Simplified ROI: Segment before each peak.
        # We need to find the start of the rise for each peak.
        # This could be the previous valley or where force starts to rise significantly.

        # For each peak, find the preceding valley (local minimum).
        last_valley_idx = 0
        for pk_idx in peak_indices:
            # Search for a valley (local minimum) in the segment before this peak and after the last valley/ROI end.
            search_segment_force = force_filtered[last_valley_idx:pk_idx]
            if len(search_segment_force) < 5:  # Too short to find a distinct valley
                start_idx_roi = last_valley_idx
            else:
                # Find minima in the negative of the search segment
                valley_indices_in_segment, _ = signal.find_peaks(-search_segment_force,
                                                                 prominence=prominence_threshold_N * 0.5)
                if len(valley_indices_in_segment) > 0:
                    # The last valley before the peak in this segment
                    current_valley_idx_in_segment = valley_indices_in_segment[-1]
                    start_idx_roi = last_valley_idx + current_valley_idx_in_segment
                else:  # No clear valley, start from the end of the last ROI or a bit before the peak
                    start_idx_roi = max(last_valley_idx, pk_idx - int(min_dist_points * 1.5))  # Fallback

            end_idx_roi = pk_idx  # End ROI at the peak

            # Ensure ROI has a minimum length and shows a force increase
            if end_idx_roi > start_idx_roi + 5 and \
                    force_filtered[end_idx_roi] > force_filtered[start_idx_roi] + prominence_threshold_N * 0.5:

                current_roi_data = force_curve_data[start_idx_roi:end_idx_roi + 1, :]

                # Filter ROI by an approximate force threshold (e.g., > 5 pN) to remove baseline noise segments
                # This is a simple filter; the MATLAB one is more sophisticated.
                min_force_for_roi = 5e-12  # 5 pN
                if np.max(current_roi_data[:, 1]) > min_force_for_roi:
                    rois.append(current_roi_data)

            last_valley_idx = end_idx_roi  # Next search starts after this peak

        if self.diagnostics:
            print(f"Identified {len(rois)} ROIs.")
        return rois

    def analyze_single_curve(self, ibw_file_path):
        """
        Analyzes a single force curve from an .ibw file.

        Parameters:
        ibw_file_path (str): Path to the .ibw file.

        Returns:
        tuple: (all_wlc_fits, all_wlc_params, fig_handle)
               all_wlc_fits (list): List of WLC fit data arrays for each successful ROI.
                                    Each array is [separation (m), fitted_force (N)].
               all_wlc_params (list): List of dictionaries, each containing parameters
                                      for a successful WLC fit (Lc, P, rupture_L, rupture_F, fit_quality_percent, velocity).
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
            wave_notes = wave_notes_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            if self.diagnostics: print(f"Could not decode wave notes for {ibw_file_path}: {e}. Using empty notes.")
            wave_notes = ""

        # Extract parameters like spring constant and velocity
        k_spring, velocity_ms, _, _ = self._extract_parameters_from_notes(wave_notes)

        # Process raw data to get force-separation curves (retract and approach)
        # We primarily use the retract curve for WLC fitting of unfolding events.
        retract_curve_processed, approach_curve_processed, success = self._process_raw_curve_data(wave_data, k_spring)

        if not success or retract_curve_processed is None or len(retract_curve_processed) == 0:
            print(f"Failed to process raw data for {ibw_file_path}.")
            return all_wlc_fits, all_wlc_params, fig_handle

        # Identify ROIs from the processed retract curve
        rois = self._find_regions_of_interest(retract_curve_processed, k_spring)

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
            return all_wlc_fits, all_wlc_params, fig_handle

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

        return all_wlc_fits, all_wlc_params, fig_handle

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

            wlc_fits, wlc_params_list, fig = self.analyze_single_curve(file_path)

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
    ENABLE_DIAGNOSTICS = False  # Set to True for verbose output during processing

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