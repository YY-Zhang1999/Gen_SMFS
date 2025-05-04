# src/evaluation/visualizer.py

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Any

# Import analysis functions for plotting derived properties like peaks or fits
try:
    from analysis.mechanical_properties import find_force_peaks
    from analysis.curve_fitting import wlc_model, fit_wlc_to_unfolding_segments # If you want to visualize fits
except ImportError as e:
    logging.error(f"Failed to import analysis modules for visualization: {e}")
    logging.error("Visualization of peaks or fits will be unavailable.")
    # Define placeholder functions
    def find_force_peaks(*args, **kwargs): return (np.array([]), {})
    def wlc_model(*args, **kwargs): return np.full_like(args[0], np.nan) # Return NaNs
    def fit_wlc_to_unfolding_segments(*args, **kwargs): return []


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message')

def plot_fe_curve_comparison(
    true_curve: np.ndarray,
    generated_curve: np.ndarray,
    sample_idx: int = None,
    extension_axis: np.ndarray = None, # Optional: provide a physical extension axis for plotting
    title: str = "Generated vs. True Force-Extension Curve",
    show_peaks: bool = False, # Option to show detected peaks
    peak_params: Dict[str, Any] = None # Parameters for find_force_peaks if show_peaks is True
):
    """
    Plots a single generated F-E curve compared to its corresponding true curve.

    Args:
        true_curve (np.ndarray): The true F-E curve (1D force vector).
        generated_curve (np.ndarray): The generated F-E curve (1D force vector).
        sample_idx (int, optional): Index of the sample being plotted (for title/context). Defaults to None.
        extension_axis (np.ndarray, optional): A 1D array representing the physical
                                               extension values corresponding to the
                                               points in the force curves. If None,
                                               uses simple index-based axis.
        title (str): Title of the plot.
        show_peaks (bool): If True, attempts to find and plot peaks on both curves.
        peak_params (Dict[str, Any], optional): Parameters for find_force_peaks.
                                               Required if show_peaks is True.
    """
    if true_curve.shape != generated_curve.shape or true_curve.ndim != 1:
        logging.error(f"Shape mismatch or incorrect dimensions for plotting. Expected (length,), got {true_curve.shape} and {generated_curve.shape}")
        return

    fe_len = len(true_curve)
    if extension_axis is None:
        extension_axis = np.arange(fe_len)
        xlabel = "Extension (Index)"
    else:
        if len(extension_axis) != fe_len:
            logging.warning(f"Extension axis length ({len(extension_axis)}) does not match curve length ({fe_len}). Using index axis.")
            extension_axis = np.arange(fe_len)
            xlabel = "Extension (Index)"
        else:
            xlabel = "Extension" # Assume physical units from axis


    plt.figure(figsize=(10, 6))
    plt.plot(extension_axis, true_curve, label='True Curve', alpha=0.7)
    plt.plot(extension_axis, generated_curve, label='Generated Curve', alpha=0.7, linestyle='--')

    if show_peaks:
        if peak_params is None:
            logging.warning("show_peaks is True but peak_params are not provided. Skipping peak plotting.")
        else:
            try:
                # Find peaks on true curve
                true_peak_indices, _ = find_force_peaks(true_curve, **peak_params)
                plt.plot(extension_axis[true_peak_indices], true_curve[true_peak_indices], 'x', color='blue', label='True Peaks')

                # Find peaks on generated curve
                gen_peak_indices, _ = find_force_peaks(generated_curve, **peak_params)
                plt.plot(extension_axis[gen_peak_indices], generated_curve[gen_peak_indices], 'o', color='red', fillstyle='none', label='Generated Peaks')

            except Exception as e:
                logging.error(f"Error finding or plotting peaks: {e}")


    plot_title = title
    if sample_idx is not None:
        plot_title = f"{title} (Sample {sample_idx})"

    plt.xlabel(xlabel)
    plt.ylabel("Force")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multiple_generated_curves(
    generated_curves: np.ndarray,
    true_curve_avg: np.ndarray = None, # Optional: plot average true curve for comparison
    extension_axis: np.ndarray = None, # Optional physical extension axis
    title: str = "Multiple Generated Force-Extension Curves",
    # Add parameters for plotting variability (e.g., confidence interval)
):
    """
    Plots multiple generated F-E curves (presumably for the same input conditions)
    to visualize the variability in generation.

    Args:
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).
                                       Assumes a single channel.
        true_curve_avg (np.ndarray, optional): Average true curve for comparison (1D force vector). Defaults to None.
        extension_axis (np.ndarray, optional): Physical extension axis.
        title (str): Title of the plot.
    """
    if generated_curves.ndim != 3 or generated_curves.shape[-1] != 1:
        logging.error(f"Invalid shape for plotting multiple curves. Expected (samples, length, 1), got {generated_curves.shape}")
        return

    num_samples, fe_len, _ = generated_curves.shape

    if extension_axis is None:
        extension_axis = np.arange(fe_len)
        xlabel = "Extension (Index)"
    else:
         if len(extension_axis) != fe_len:
             logging.warning(f"Extension axis length ({len(extension_axis)}) does not match curve length ({fe_len}). Using index axis.")
             extension_axis = np.arange(fe_len)
             xlabel = "Extension (Index)"
         else:
             xlabel = "Extension"


    plt.figure(figsize=(10, 6))

    # Plot each generated curve
    for i in range(num_samples):
        plt.plot(extension_axis, generated_curves[i, :, 0], alpha=0.5, linewidth=1)

    # Plot average generated curve
    gen_avg = np.mean(generated_curves[:, :, 0], axis=0)
    plt.plot(extension_axis, gen_avg, color='red', linewidth=2, label='Average Generated')

    # Plot average true curve if provided
    if true_curve_avg is not None:
        if true_curve_avg.shape != (fe_len,):
            logging.warning(f"Average true curve shape {true_curve_avg.shape} does not match expected ({fe_len},). Skipping plotting average true.")
        else:
            plt.plot(extension_axis, true_curve_avg, color='blue', linewidth=2, linestyle='--', label='Average True')

    plt.xlabel(xlabel)
    plt.ylabel("Force")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_property_distributions(
    true_properties: Dict[str, List[float]],
    generated_properties: Dict[str, List[float]],
    property_name: str # The name of the property to plot
):
    """
    Plots histograms or kernel density estimates (KDEs) of the distributions
    of a specific mechanical property for true and generated curves.

    Args:
        true_properties (Dict[str, List[float]]): Dictionary of true property lists.
        generated_properties (Dict[str, List[float]]): Dictionary of generated property lists.
        property_name (str): The name of the property to plot (must be a key in both dicts).
    """
    if property_name not in true_properties or property_name not in generated_properties:
        logging.error(f"Property '{property_name}' not found in both true and generated property dictionaries.")
        return

    true_vals = np.array(true_properties[property_name])
    gen_vals = np.array(generated_properties[property_name])

    # Remove NaNs for plotting
    true_valid = true_vals[~np.isnan(true_vals)]
    gen_valid = gen_vals[~np.isnan(gen_vals)]

    if len(true_valid) == 0 and len(gen_valid) == 0:
        logging.warning(f"No valid data for property '{property_name}' to plot distribution.")
        return

    plt.figure(figsize=(8, 6))

    # Plot histograms (can adjust bins)
    plt.hist(true_valid, bins=30, alpha=0.7, label='True Distribution', density=True) # density=True for normalized histogram
    plt.hist(gen_valid, bins=30, alpha=0.7, label='Generated Distribution', density=True)

    # Optional: Plot KDEs (requires scipy or seaborn)
    # import seaborn as sns
    # if len(true_valid) > 1: sns.kdeplot(true_valid, label='True KDE')
    # if len(gen_valid) > 1: sns.kdeplot(gen_valid, label='Generated KDE')


    plt.xlabel(property_name.replace('_', ' ').title()) # Simple formatting for axis label
    plt.ylabel("Density / Frequency")
    plt.title(f"Distribution of {property_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example Usage
if __name__ == "__main__":
    print("--- Testing visualizer.py ---")

    # Create dummy true and generated curves (same as in metrics.py example)
    num_samples = 20
    fe_len = 200
    channels = 1

    # Dummy true curves (simulate some variation)
    true_curves = np.random.rand(num_samples, fe_len, channels) * 100
    # Add some peak-like features to true curves
    for i in range(num_samples):
        peak_idx = np.random.randint(50, 150)
        peak_force = np.random.rand() * 80 + 20
        true_curves[i, :peak_idx, 0] = np.linspace(0, peak_force, peak_idx)
        true_curves[i, peak_idx:, 0] = peak_force * np.exp(-(np.arange(fe_len - peak_idx)) / 50)
        true_curves[i, :, 0] += np.random.randn(fe_len) * 5 # Add noise

    # Dummy generated curves (simulate some similarity but with noise)
    generated_curves = true_curves + np.random.randn(*true_curves.shape) * 10

    # Create a dummy physical extension axis
    dummy_physical_extension = np.linspace(0, 150, fe_len) # 0 to 150 nm

    # --- Test plot_fe_curve_comparison ---
    print("\n--- Testing plot_fe_curve_comparison ---")
    sample_to_plot = 5 # Plot the 6th sample
    plot_fe_curve_comparison(
        true_curves[sample_to_plot, :, 0], # Pass 1D arrays
        generated_curves[sample_to_plot, :, 0],
        sample_idx=sample_to_plot,
        extension_axis=dummy_physical_extension, # Plot with physical extension
        show_peaks=True, # Also try plotting peaks
        peak_params={'height': 10, 'distance': 20, 'prominence': 5} # Peak finding parameters
    )


    # --- Test plot_multiple_generated_curves ---
    print("\n--- Testing plot_multiple_generated_curves ---")
    # Assume the first 10 generated curves are for the same input
    plot_multiple_generated_curves(
        generated_curves[:10],
        true_curve_avg=np.mean(true_curves[:, :, 0], axis=0), # Plot average true curve
        extension_axis=dummy_physical_extension # Plot with physical extension
    )

    # --- Test plot_property_distributions ---
    print("\n--- Testing plot_property_distributions ---")
    # First, extract properties from the dummy curves
    peak_params_for_plotting = {'height': 10, 'distance': 20, 'prominence': 5}
    true_properties_dict: Dict[str, List[float]] = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [], 'avg_unfolding_force': []}
    gen_properties_dict: Dict[str, List[float]] = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [], 'avg_unfolding_force': []}

    for i in range(num_samples):
         true_curve_1d = true_curves[i, :, 0]
         gen_curve_1d = generated_curves[i, :, 0]

         # Assume dummy extension step 1.0 for energy calculation in this test
         true_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(true_curve_1d, extension_step=1.0))
         true_properties_dict['max_force'].append(calculate_max_force(true_curve_1d))
         true_peak_indices, _ = find_force_peaks(true_curve_1d, **peak_params_for_plotting)
         true_properties_dict['num_peaks'].append(len(true_peak_indices))
         true_unfolding_forces = _.get('peak_heights', [])
         true_properties_dict['avg_unfolding_force'].append(np.mean(true_unfolding_forces) if len(true_unfolding_forces) > 0 else np.nan)


         gen_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(gen_curve_1d, extension_step=1.0))
         gen_properties_dict['max_force'].append(calculate_max_force(gen_curve_1d))
         gen_peak_indices, _ = find_force_peaks(gen_curve_1d, **peak_params_for_plotting)
         gen_properties_dict['num_peaks'].append(len(gen_peak_indices))
         gen_unfolding_forces = _.get('peak_heights', [])
         gen_properties_dict['avg_unfolding_force'].append(np.mean(gen_unfolding_forces) if len(gen_unfolding_forces) > 0 else np.nan)


    # Now plot distributions for specific properties
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'max_force')
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'num_peaks')
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'unfolding_energy')