# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import r2_score # scikit-learn for R^2
from dtw import dtw # dtw-python for Dynamic Time Warping
import logging
from typing import List, Tuple, Dict, Any

# Import analysis functions to evaluate derived properties
try:
    from analysis.mechanical_properties import calculate_unfolding_energy, calculate_max_force, find_force_peaks
    # Import other analysis functions if needed
except ImportError as e:
    logging.error(f"Failed to import analysis modules for property evaluation: {e}")
    logging.error("Property evaluation metrics will be unavailable.")
    # Define placeholder functions if necessary to allow the rest of the file to load
    def calculate_unfolding_energy(*args, **kwargs): return np.nan
    def calculate_max_force(*args, **kwargs): return np.nan
    def find_force_peaks(*args, **kwargs): return (np.array([]), {})


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message')

def calculate_r2(true_curves: np.ndarray, generated_curves: np.ndarray) -> float:
    """
    Calculates the R^2 (coefficient of determination) between true and generated curves.
    Treats each curve as a high-dimensional sample.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).

    Returns:
        float: The calculated R^2 score. Returns np.nan if shapes don't match or inputs are invalid.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for R^2 calculation: {true_curves.shape} vs {generated_curves.shape}")
        return np.nan
    if true_curves.ndim < 2 or generated_curves.ndim < 2:
        logging.error("Input arrays must have at least 2 dimensions (samples, length, [channels]).")
        return np.nan

    # Reshape curves to (num_samples, feature_dimension) for r2_score
    num_samples = true_curves.shape[0]
    feature_dim = np.prod(true_curves.shape[1:])
    true_flat = true_curves.reshape(num_samples, feature_dim)
    gen_flat = generated_curves.reshape(num_samples, feature_dim)

    try:
        r2 = r2_score(true_flat, gen_flat)
        return r2
    except Exception as e:
        logging.error(f"Error calculating R^2 score: {e}")
        return np.nan


def calculate_relative_l2_error(true_curves: np.ndarray, generated_curves: np.ndarray) -> float:
    """
    Calculates the average Relative L2 Error (RMSE / ||true_curve||_2) per curve in the batch.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).

    Returns:
        float: The average relative L2 error. Returns np.nan if shapes don't match or inputs are invalid.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for Relative L2 Error calculation: {true_curves.shape} vs {generated_curves.shape}")
        return np.nan
    if true_curves.ndim < 2 or generated_curves.ndim < 2:
        logging.error("Input arrays must have at least 2 dimensions (samples, length, [channels]).")
        return np.nan

    # Flatten the last dimensions to calculate L2 norm per sample
    true_flat = true_curves.reshape(true_curves.shape[0], -1)
    gen_flat = generated_curves.reshape(generated_curves.shape[0], -1)

    # Calculate L2 norm of the difference (RMSE for each sample)
    l2_diff = np.linalg.norm(true_flat - gen_flat, ord=2, axis=1)

    # Calculate L2 norm of the true curve for normalization
    true_norm = np.linalg.norm(true_flat, ord=2, axis=1)

    # Avoid division by zero for true curves with zero norm
    non_zero_norm_mask = true_norm > 1e-6 # Use a small threshold
    if not np.any(non_zero_norm_mask):
        logging.warning("All true curves have zero norm. Cannot calculate relative L2 error.")
        return np.nan

    # Calculate relative L2 error for curves with non-zero norm
    relative_l2_errors = np.zeros_like(l2_diff)
    relative_l2_errors[non_zero_norm_mask] = l2_diff[non_zero_norm_mask] / true_norm[non_zero_norm_mask]

    # Return the average relative L2 error over the batch
    return np.mean(relative_l2_errors[non_zero_norm_mask])


def calculate_dtw_distance(true_curves: np.ndarray, generated_curves: np.ndarray) -> float:
    """
    Calculates the average Dynamic Time Warping (DTW) distance between true and generated curves.
    DTW is useful for comparing sequences that may vary in the time/extension axis.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
                                  Assumes a single channel for DTW calculation.
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).
                                  Assumes a single channel for DTW calculation.

    Returns:
        float: The average DTW distance. Returns np.nan if shapes don't match, channels > 1, or inputs are invalid.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for DTW calculation: {true_curves.shape} vs {generated_curves.shape}")
        return np.nan
    if true_curves.shape[-1] > 1:
        logging.error("DTW calculation currently supports only a single channel (force).")
        return np.nan
    if true_curves.ndim < 2 or generated_curves.ndim < 2:
        logging.error("Input arrays must have at least 2 dimensions (samples, length, [channels]).")
        return np.nan


    num_samples = true_curves.shape[0]
    total_dtw_distance = 0.0
    successful_dtw_calculations = 0

    # DTW is typically calculated per pair of sequences
    for i in range(num_samples):
        # Ensure curves are 1D for DTW library
        true_curve_1d = true_curves[i, :, 0]
        gen_curve_1d = generated_curves[i, :, 0]

        try:
            # Calculate DTW distance
            # You might need to adjust distance_metric based on your data scale and preference
            # 'euclidean', 'sqeuclidean', 'cosine', etc.
            distance, _, _, _ = dtw(true_curve_1d, gen_curve_1d, dist_method='euclidean')
            total_dtw_distance += distance
            successful_dtw_calculations += 1
        except Exception as e:
            logging.warning(f"DTW calculation failed for sample {i}: {e}. Skipping this sample.")
            continue

    if successful_dtw_calculations == 0:
        logging.warning("No successful DTW calculations.")
        return np.nan

    return total_dtw_distance / successful_dtw_calculations

def evaluate_mechanical_properties(
    true_curves: np.ndarray,
    generated_curves: np.ndarray,
    property_extraction_params: Dict[str, Any] = None # Parameters for analysis functions
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the accuracy of inferred mechanical properties by comparing
    properties extracted from generated curves to those extracted from true curves.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).
        property_extraction_params (Dict[str, Any], optional): Dictionary of parameters
                                                               for analysis functions like find_force_peaks.
                                                               Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are property names
                                     ('unfolding_energy', 'max_force', 'num_peaks', 'avg_unfolding_force', etc.)
                                     and values are dictionaries containing evaluation metrics
                                     (e.g., {'r2': value, 'mean_abs_error': value}).
                                     Returns empty dict if shapes don't match or extraction fails.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for property evaluation: {true_curves.shape} vs {generated_curves.shape}")
        return {}

    num_samples = true_curves.shape[0]
    true_properties: Dict[str, List[float]] = {}
    gen_properties: Dict[str, List[float]] = {}

    # Default extraction parameters if none provided
    default_extraction_params = {
        'find_peaks': {'height': None, 'distance': None, 'prominence': None} # Add your typical values
        # Add default params for WLC fitting if you evaluate those
    }
    if property_extraction_params is None:
        property_extraction_params = default_extraction_params

    logging.info(f"Extracting and evaluating mechanical properties for {num_samples} samples...")

    for i in range(num_samples):
        true_curve_1d = true_curves[i, :, 0] # Assuming 1 channel (force)
        gen_curve_1d = generated_curves[i, :, 0]

        # --- Extract properties from True Curve ---
        try:
            true_energy = calculate_unfolding_energy(true_curve_1d, extension_step=1.0) # Use a dummy step if physical ext is unknown
            true_max_force = calculate_max_force(true_curve_1d)
            true_peak_indices, true_peak_props = find_force_peaks(true_curve_1d, **property_extraction_params.get('find_peaks', {}))
            true_num_peaks = len(true_peak_indices)
            true_unfolding_forces = true_peak_props.get('peak_heights', [])
            true_avg_unfolding_force = np.mean(true_unfolding_forces) if len(true_unfolding_forces) > 0 else np.nan

            # Store true properties
            if 'unfolding_energy' not in true_properties: true_properties['unfolding_energy'] = []
            if 'max_force' not in true_properties: true_properties['max_force'] = []
            if 'num_peaks' not in true_properties: true_properties['num_peaks'] = []
            if 'avg_unfolding_force' not in true_properties: true_properties['avg_unfolding_force'] = []

            true_properties['unfolding_energy'].append(true_energy)
            true_properties['max_force'].append(true_max_force)
            true_properties['num_peaks'].append(true_num_peaks)
            true_properties['avg_unfolding_force'].append(true_avg_unfolding_force)

        except Exception as e:
            logging.warning(f"Error extracting properties from true curve {i}: {e}. Skipping this sample for true properties.")
            continue # Skip sample if true property extraction fails (issue with data)


        # --- Extract properties from Generated Curve ---
        try:
            gen_energy = calculate_unfolding_energy(gen_curve_1d, extension_step=1.0) # Use the same dummy step
            gen_max_force = calculate_max_force(gen_curve_1d)
            gen_peak_indices, gen_peak_props = find_force_peaks(gen_curve_1d, **property_extraction_params.get('find_peaks', {}))
            gen_num_peaks = len(gen_peak_indices)
            gen_unfolding_forces = gen_peak_props.get('peak_heights', [])
            gen_avg_unfolding_force = np.mean(gen_unfolding_forces) if len(gen_unfolding_forces) > 0 else np.nan

            # Store generated properties
            if 'unfolding_energy' not in gen_properties: gen_properties['unfolding_energy'] = []
            if 'max_force' not in gen_properties: gen_properties['max_force'] = []
            if 'num_peaks' not in gen_properties: gen_properties['num_peaks'] = []
            if 'avg_unfolding_force' not in gen_properties: gen_properties['avg_unfolding_force'] = []

            gen_properties['unfolding_energy'].append(gen_energy)
            gen_properties['max_force'].append(gen_max_force)
            gen_properties['num_peaks'].append(gen_num_peaks)
            gen_properties['avg_unfolding_force'].append(gen_avg_unfolding_force)

        except Exception as e:
             logging.warning(f"Error extracting properties from generated curve {i}: {e}. Skipping this sample for generated properties.")
             # Add placeholder NaNs to keep lists aligned if some true properties were extracted
             for prop_name in true_properties.keys(): # Iterate based on properties successfully extracted from true curves
                  if prop_name not in gen_properties: gen_properties[prop_name] = [] # Initialize if first failure
                  gen_properties[prop_name].append(np.nan) # Add NaN for failed sample


    # --- Evaluate Metrics for Each Property ---
    evaluation_metrics: Dict[str, Dict[str, float]] = {}

    # Ensure that we only evaluate properties for which we have both true and generated values for at least some samples
    common_properties = set(true_properties.keys()) & set(gen_properties.keys())

    for prop_name in common_properties:
        true_vals = np.array(true_properties[prop_name])
        gen_vals = np.array(gen_properties[prop_name])

        # Remove NaNs to only compare valid pairs
        valid_mask = (~np.isnan(true_vals)) & (~np.isnan(gen_vals))
        if not np.any(valid_mask):
             logging.warning(f"No valid pairs for property '{prop_name}'. Skipping evaluation for this property.")
             continue

        true_valid = true_vals[valid_mask]
        gen_valid = gen_vals[valid_mask]

        if len(true_valid) < 2:
             logging.warning(f"Need at least 2 valid samples to calculate R^2 for property '{prop_name}'. Skipping R^2.")
             r2 = np.nan
        else:
             try:
                 r2 = r2_score(true_valid, gen_valid)
             except Exception as e:
                 logging.warning(f"Error calculating R^2 for property '{prop_name}': {e}. Setting to NaN.")
                 r2 = np.nan


        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(true_valid - gen_valid))

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((true_valid - gen_valid)**2))

        # Calculate Mean Relative Error (MRE) - be cautious with division by zero/small numbers
        # MRE = np.mean(np.abs(true_valid - gen_valid) / np.abs(true_valid)) # Avoid division by zero

        evaluation_metrics[prop_name] = {
            'r2': r2,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'num_valid_samples': len(true_valid)
        }

        logging.info(f"Property '{prop_name}' metrics: R^2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} (N={len(true_valid)})")


    logging.info("Mechanical property evaluation complete.")

    return evaluation_metrics


# Example Usage
if __name__ == "__main__":
    print("--- Testing metrics.py ---")

    # Create dummy true and generated curves
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

    # --- Test Curve Shape Metrics ---
    print("\n--- Testing Curve Shape Metrics ---")
    r2_score_curves = calculate_r2(true_curves, generated_curves)
    print(f"Overall R^2 for curve shapes: {r2_score_curves:.4f}")

    rel_l2_error = calculate_relative_l2_error(true_curves, generated_curves)
    print(f"Average Relative L2 Error: {rel_l2_error:.4f}")

    # DTW can be slow for many large curves
    # print("\n--- Testing DTW Distance (may take time) ---")
    # dtw_dist = calculate_dtw_distance(true_curves, generated_curves)
    # print(f"Average DTW Distance: {dtw_dist:.4f}")


    # --- Test Mechanical Property Evaluation ---
    print("\n--- Testing Mechanical Property Evaluation ---")
    # Need parameters for peak finding for 'num_peaks', 'avg_unfolding_force'
    peak_params = {'height': 10, 'distance': 20, 'prominence': 5}
    property_evaluation_metrics = evaluate_mechanical_properties(
        true_curves,
        generated_curves,
        property_extraction_params={'find_peaks': peak_params}
    )

    print("\nMechanical Property Evaluation Metrics:")
    for prop_name, metrics in property_evaluation_metrics.items():
        print(f"Property: {prop_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")