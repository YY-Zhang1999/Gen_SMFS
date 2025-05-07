# Function to extract force and extension data from .ibw files
import numpy as np
from igor.binarywave import load as ibw_load  # Using igor library to read .ibw files


def extract_force_extension(ibw_file_path):
    """
    Extracts only the force and extension data from an Igor Binary Wave (.ibw) file

    Parameters:
    ibw_file_path -- Path to the .ibw file

    Returns:
    extension -- Array of extension values (in meters)
    force -- Array of force values (in Newtons)
    """
    try:
        # Load the .ibw file
        ibw_data = ibw_load(ibw_file_path)

        # Extract wave data and notes
        wave_data = ibw_data['wave']['wData']
        wave_notes = str(ibw_data['wave']['note']) if 'note' in ibw_data['wave'] else ''
        print(wave_notes)

        # Get spring constant from wave notes
        k_spr = None
        if 'SpringConstant' in wave_notes:
            k_spr_loc = wave_notes.find('SpringConstant')
            k_spr_end = wave_notes.find('\\r', k_spr_loc)
            print(k_spr_loc, k_spr_end)
            if k_spr_end == -1:  # If no carriage return, try finding a newline
                k_spr_end = wave_notes.find('\\n', k_spr_loc)
            if k_spr_end == -1:  # If still no end, take a reasonable chunk
                k_spr_end = k_spr_loc + 30
            k_spr_str = wave_notes[k_spr_loc + 15:k_spr_end].strip()
            print(k_spr_str)
            try:
                k_spr = float(k_spr_str)
            except:
                print(f"Couldn't parse spring constant: '{k_spr_str}'")
                k_spr = 1.0  # Default value
        else:
            print("Spring constant not found in wave notes")
            k_spr = 1.0  # Default value

        # Extract raw data based on the dimensions
        if len(wave_data.shape) == 2:
            # Assuming the data is organized as [position, deflection]
            # Column indices may need adjustment based on your specific file format
            extension_col = 2  # Typically the 3rd column (index 2) is separation/extension
            deflection_col = 1  # Typically the 2nd column (index 1) is deflection

            # Extract raw deflection and extension
            deflection = wave_data[:, deflection_col]
            extension = wave_data[:, extension_col]

            # Convert deflection to force using spring constant
            force = deflection * k_spr

            return extension, force

        elif len(wave_data.shape) == 1:
            # For single-column data, may need additional processing
            print("1D data format detected. Need specific parsing rules.")
            return None, None

        else:
            print(f"Unexpected data shape: {wave_data.shape}")
            return None, None

    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return None, None


# Function that analyzes force curve data and generates WLC model fit and parameters
# Based on original MATLAB code by Sivaraman R
import numpy as np
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import re
from igor.binarywave import load as ibw_load  # Using igor library to read .ibw files


def analyze_force_curves(ibw_file_path, temp, fit_threshold, make_plots, path_str):
    """
    Analyzes force curve data and generates WLC model fit and parameters

    Parameters:
    ibw_file_path -- Path to the .ibw file
    temp -- Temperature in Kelvin
    fit_threshold -- Fitting threshold percentage
    make_plots -- Boolean flag to generate plots
    path_str -- Path string for saving results

    Returns:
    WLC -- WLC model data
    WLC_params -- WLC model parameters
    """
    # Load the .ibw file
    ig_curve = ibw_load(ibw_file_path)

    # Enable detailed diagnostics
    diagnostics = 0
    fig_no = 1

    # *************************************************************
    # Extract experiment parameters from igor file wave notes
    # *************************************************************
    # Get wave notes as string
    wave_notes = ig_curve['wave']['note'].decode('utf-8', errors='ignore') if isinstance(ig_curve['wave']['note'],
                                                                                         bytes) else ig_curve['wave'][
        'note']

    # Spring constant
    k_spr_loc = wave_notes.find('SpringConstant')
    if k_spr_loc != -1:
        k_spr_end = wave_notes.find('\\r', k_spr_loc)
        if k_spr_end == -1:
            k_spr_end = wave_notes.find('\\n', k_spr_loc)
        if k_spr_end == -1:
            k_spr_end = k_spr_loc + 22  # Default length if no line ending found

        k_spr = wave_notes[k_spr_loc + 15:k_spr_end].strip()
        try:
            k_spr = float(k_spr)  # N/m
            k_spr_read_success = 1
        except:
            print('Cannot read spring constant!')
            k_spr = 0.01  # Default value
            k_spr_read_success = 0
    else:
        print('Spring constant not found in wave notes')
        k_spr = 0.01  # Default value
        k_spr_read_success = 0

    # Velocity
    v_loc = wave_notes.find('RetractVelocity')
    v_loc_next = wave_notes.find('ForceScanRate')
    if v_loc != -1 and v_loc_next != -1:
        velocity = wave_notes[v_loc + 17:v_loc_next - 1].strip()
        try:
            velocity = float(velocity)  # m/s
            velocity_read_success = 1
        except:
            print('Cannot read velocity!')
            velocity = 1e-6  # Default value
            velocity_read_success = 0
    else:
        print('Velocity not found in wave notes')
        velocity = 1e-6  # Default value
        velocity_read_success = 0

    # *************************************************************
    # Extract wave data
    # *************************************************************
    # Get the data from the wave
    wave_data = ig_curve['wave']['wData']

    # Check data dimensions
    if len(wave_data.shape) == 2:
        # For a 2D dataset
        y_data = wave_data
    elif len(wave_data.shape) == 1:
        # For a 1D dataset, reshape as needed
        y_data = wave_data.reshape(-1, 1)
        print("1D data detected, may need specific format adjustments")
    else:
        print(f"Unexpected data shape: {wave_data.shape}")
        return [], []

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

    if diagnostics == 1:
        plt.figure(fig_no)
        fig_no += 1
        plt.plot(dfl[:, 0], dfl[:, 1])
        plt.plot(apr[:, 0], apr[:, 1])
        plt.show()

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

    if diagnostics == 1:
        plt.figure(fig_no)
        fig_no += 1
        plt.plot(apr[:, 0], apr[:, 1])
        plt.plot(apr[:, 0], np.ones(am) * (yo + (2 * ystd)), 'g')
        plt.plot(apr[:, 0], np.ones(am) * yo, 'r')
        plt.show()

    # Bias corrected data
    md, _ = dfl.shape
    bias = np.column_stack((np.ones(md) * xo, np.ones(md) * yo))

    dfln = dfl - bias  # Data with true 0

    if diagnostics == 1:
        plt.figure(fig_no)
        fig_no += 1
        plt.plot(dfl[:, 0], dfl[:, 1])
        plt.plot(dfln[:, 0], dfln[:, 1])
        plt.show()

    # *************************************************************
    # Convert the data to Force (N) vs Separation (m)
    # *************************************************************
    force_p = np.zeros_like(dfln)
    force_p[:, 1] = dfln[:, 1] * k_spr  # Convert deflection to force
    force_p[:, 0] = dfln[:, 0] - dfln[:, 1]  # Calculate true separation

    # Downsample to reduce noise and computational load
    dfln = force_p[::2]  # Take every 4th point

    # *************************************************************
    # Define WLC model functions
    # *************************************************************
    def get_force_wlc(lc, p, x, temp):
        """WLC model force function"""
        kb = 1.3806488e-23  # Boltzmann constant
        x = abs(x)  # Make sure extension is positive

        # Check if contour length is valid
        if lc <= 0:
            return 0

        r = x / lc  # Relative extension

        # Force calculation with WLC model
        if r < 0.9:
            force = (kb * temp / p) * (0.25 * (1 / (1 - r) ** 2 - 1 + 4 * r))
        else:
            force = (kb * temp / p) * (0.25 * (1 / (1 - 0.9) ** 2 - 1 + 4 * 0.9))

        return force

    def get_cost_wlc(params, sep_data, temp, f_data):
        """Cost function for WLC model fitting"""
        lc, p = params
        cost = 0

        for i in range(len(sep_data)):
            f_calc = get_force_wlc(lc, p, sep_data[i], temp)
            cost += (f_calc - f_data[i]) ** 2

        return cost

    # *************************************************************
    # Narrow the region of interest and apply filtering
    # *************************************************************
    # Apply median filter to smooth data
    dfln_f = np.zeros_like(dfln)
    dfln_f[:, 0] = dfln[:, 0]
    dfln_f[:, 1] = signal.medfilt(dfln[:, 1], 7)

    # Find peaks and valleys in the filtered data
    pks = []
    locs = []
    w = np.array([])
    p = np.array([])

    try:
        # Find peaks safely
        locs, properties = signal.find_peaks(dfln_f[:, 1])
        if len(locs) > 0:
            pks = dfln_f[locs, 1]
            # Calculate properties only if we have peaks
            w = signal.peak_widths(dfln_f[:, 1], locs)[0]
            p = signal.peak_prominences(dfln_f[:, 1], locs)[0]
    except Exception as e:
        print(f"Peak detection error: {e}")
        # Continue with empty arrays


    # Find valleys (negative peaks)
    neg_pks_f = []
    neg_locs_f = []
    nw_f = np.array([])
    np_f = np.array([])

    try:
        # Find valleys safely
        neg_locs_f, properties = signal.find_peaks(-dfln_f[:, 1])
        if len(neg_locs_f) > 0:
            neg_pks_f = -dfln_f[neg_locs_f, 1]
            # Calculate properties only if we have valleys
            nw_f = signal.peak_widths(-dfln_f[:, 1], neg_locs_f)[0]
            np_f = signal.peak_prominences(-dfln_f[:, 1], neg_locs_f)[0]
    except Exception as e:
        print(f"Valley detection error: {e}")

    # *************************************************************
    # If there are enough peaks for analysis, continue with WLC fitting
    # *************************************************************
    # Check if we have sufficient data for analysis
    if len(p) == 0 or len(np_f) == 0:
        print("Insufficient peak data for analysis")
        return [], []

    # Create the WLC fitting and analysis
    WLC = []
    WLC_params = []

    # Define a simplified ROI for analysis if peaks/valleys analysis didn't work
    # Just use the whole curve but filter out low-force regions
    force_threshold = np.std(dfln[:, 1]) * 3
    roi_indices = np.where(np.abs(dfln[:, 1]) > force_threshold)[0]

    if len(roi_indices) > 0:
        roi = [dfln[roi_indices]]
    else:
        # If no significant force regions found, just use middle segment
        mid_point = len(dfln) // 2
        segment_size = len(dfln) // 4
        roi = [dfln[mid_point - segment_size:mid_point + segment_size]]

    # Fitting a WLC model for each ROI
    brk_cnt = 1
    f_fit = np.zeros(len(roi))

    for i in range(len(roi)):
        sep_data = roi[i][:, 0]
        f_data = -roi[i][:, 1]  # Negative because the WLC model fits pulling forces

        # Initial guess: param_wlc = [Lc, P]
        # Contour length typically larger than max extension
        # Persistence length typically 0.1-1% of contour length for DNA/proteins
        param_wlc = [2 * np.max(np.abs(sep_data)), 0.01 * np.max(np.abs(sep_data))]

        # Optimize parameters
        result = optimize.minimize(
            lambda x: get_cost_wlc(x, sep_data, temp, f_data),
            param_wlc,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'maxfev': 20000}
        )
        param_wlc_calc = result.x

        # Calculate force using optimized parameters
        f_calc = np.zeros(len(sep_data))
        for j in range(len(sep_data)):
            f_calc[j] = get_force_wlc(param_wlc_calc[0], param_wlc_calc[1], sep_data[j], temp)

        # Computing percentage fit
        f_diff = f_calc - f_data
        f_ratio = f_diff / f_data
        fit_val = np.linalg.norm(f_ratio) / np.sqrt(len(f_diff))  # RMS value
        f_fit[i] = (1 - fit_val) * 100

        rup_len = np.max(np.abs(sep_data))
        rup_force = np.max(np.abs(f_data))

        # Ensuring the WLC model is not incorrectly fit
        f_wlc_comb = np.column_stack((-sep_data, f_calc))
        f_wlc_comb_srt = f_wlc_comb[f_wlc_comb[:, 0].argsort()]

        f_wlc_diff = f_wlc_comb_srt[0, 1] - f_wlc_comb_srt[-1, 1]

        if f_wlc_diff <= 0:
            f_fit[i] = -1  # This will reject the fit from further analysis

        wlc_data = np.column_stack((sep_data, -f_calc))
        f_wlc_original = np.column_stack((sep_data, -f_data))

        # WLC: [Sep vs calc force, original data, region of interest]
        WLC.append([wlc_data, f_wlc_original, dfln])

        # WLC_params: [Contour Length, persistence length, Rupture Length, Rupture Force, percent fit, break count, velocity]
        WLC_params.append(
            [[param_wlc_calc[0], param_wlc_calc[1], rup_len, rup_force, f_fit[i], brk_cnt, velocity], path_str])

        brk_cnt += 1

    # *************************************************************
    # Plot generation for review
    # *************************************************************
    if make_plots == 1:
        plt.figure(fig_no)
        fig_no += 1

        # Plot settings
        x_scale = 1e9  # nm
        y_scale = 1e12  # pN

        # First subplot - Full range
        plt.subplot(2, 1, 1)

        # Unfiltered plot
        plt.plot(-dfln[:, 0] * x_scale, dfln[:, 1] * y_scale)

        # Add title and labels
        plt.title('Force vs Extension - Full range')
        plt.xlabel('Separation (nm)')
        plt.ylabel('Force (pN)')
        plt.grid(True)

        # Second subplot - WLC model fit
        plt.subplot(2, 1, 2)

        # Plot original data
        for i in range(len(WLC)):
            plt.plot(-WLC[i][1][:, 0] * x_scale, WLC[i][1][:, 1] * y_scale, 'b-', linewidth=1)

            # Plot WLC fit
            wlc_params = WLC_params[i][0]
            if wlc_params[4] >= fit_threshold:  # Check fit quality
                plt.plot(-WLC[i][0][:, 0] * x_scale, WLC[i][0][:, 1] * y_scale, 'g-', linewidth=2)

                # Add annotation for contour length
                lc_nm = wlc_params[0] * 1e9  # Convert to nm
                p_nm = wlc_params[1] * 1e9  # Convert to nm
                plt.text(0.05, 0.9 - i * 0.1, f"Lc = {lc_nm:.1f} nm, P = {p_nm:.2f} nm",
                         transform=plt.gca().transAxes)
            else:
                plt.plot(-WLC[i][0][:, 0] * x_scale, WLC[i][0][:, 1] * y_scale, 'r-', linewidth=2)

        plt.title('Force vs Extension - WLC Model Fit')
        plt.xlabel('Separation (nm)')
        plt.ylabel('Force (pN)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return WLC, WLC_params


def plot_force_extension(extension, force, save_path=None):
    """
    Creates a simple plot of force vs. extension

    Parameters:
    extension -- Array of extension values
    force -- Array of force values
    save_path -- Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(-extension * 1e9, force * 1e12, '-')  # Convert to nm and pN for display
    plt.xlabel('Extension (nm)')
    plt.ylabel('Force (pN)')
    plt.title('Force-Extension Curve')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your .ibw file path
    file_path = "Image0000.ibw"

    # Extract data
    extension, force = extract_force_extension(file_path)

    if extension is not None and force is not None:
        print(f"Extracted {len(extension)} data points")

        # Plot the data
        plot_force_extension(extension, force)

        # Save to CSV if needed
        import pandas as pd

        df = pd.DataFrame({
            'Extension (m)': extension,
            'Force (N)': force
        })
        df.to_csv('force_extension_data.csv', index=False)
        print("Data saved to force_extension_data.csv")
    else:
        print("Failed to extract data")

    # Set parameters for analysis
    temperature = 298  # Kelvin (25°C)
    fit_threshold = 80  # Percentage threshold for good fit
    make_plots = True
    output_path = "./"

    # Run analysis
    wlc_data, wlc_params = analyze_force_curves(file_path, temperature, fit_threshold, make_plots, output_path)

    # Print results
    if len(wlc_params) > 0:
        print("\nWLC Model Parameters:")
        for i, params in enumerate(wlc_params):
            if params[0][4] >= fit_threshold:  # Good fit
                print(f"Region {i + 1} (Good Fit):")
                print(f"  Contour Length: {params[0][0] * 1e9:.2f} nm")
                print(f"  Persistence Length: {params[0][1] * 1e9:.2f} nm")
                print(f"  Rupture Length: {params[0][2] * 1e9:.2f} nm")
                print(f"  Rupture Force: {params[0][3] * 1e12:.2f} pN")
                print(f"  Fit Quality: {params[0][4]:.1f}%")
            else:
                print(f"Region {i + 1} (Poor Fit, skipped)")
    else:
        print("No valid WLC fits found")

