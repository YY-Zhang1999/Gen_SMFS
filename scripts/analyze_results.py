# scripts/analyze_results.py

import os
import yaml
import argparse
import logging
import numpy as np
import pandas as pd # Useful for organizing results

# Ensure src is in PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from data_processing.utils import load_raw_data # Use for loading .npy or .pkl if saved that way
    from analysis.mechanical_properties import calculate_unfolding_energy, calculate_max_force, find_force_peaks, analyze_unfolding_pathway
    from analysis.curve_fitting import fit_wlc_to_unfolding_segments # If evaluating WLC fits
    from evaluation.metrics import calculate_r2, calculate_relative_l2_error, calculate_dtw_distance, evaluate_mechanical_properties
    from evaluation.visualizer import plot_fe_curve_comparison, plot_multiple_generated_curves, plot_property_distributions
    # Import other analysis/evaluation functions as needed

except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure the 'src' directory is in your PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file {config_path}: {e}")
        sys.exit(1)


def main(data_config_path: str, true_data_path: str, generated_data_path: str, output_dir: str, plot_samples: int = 5):
    """
    Main function to load true and generated data, perform analysis and evaluation,
    and generate visualizations.
    """
    data_cfg = load_config(data_config_path)
    analysis_params = data_cfg.get('analysis_params', {})
    fe_curve_length = data_cfg.get('data_preprocessing', {}).get('fe_curve_length')
    fe_curve_channels = data_cfg.get('data_preprocessing', {}).get('fe_curve_channels', 1)

    if fe_curve_length is None:
        logging.error("fe_curve_length must be specified in data_config.data_preprocessing.")
        sys.exit(1)

    # --- Load True and Generated Data ---
    # Assuming true_data_path and generated_data_path point to files
    # containing the processed F-E curves (e.g., .npy or .pkl)
    logging.info("Loading true and generated F-E curves...")
    try:
        true_curves_np = load_raw_data(true_data_path) # Using load_raw_data which handles some formats
        generated_curves_np = load_raw_data(generated_data_path)

        # Ensure data is numpy arrays
        if not isinstance(true_curves_np, np.ndarray) or not isinstance(generated_curves_np, np.ndarray):
             logging.error("Loaded data is not in numpy array format. Please ensure processed data is saved as .npy or compatible.")
             sys.exit(1)

        # Ensure shapes are consistent (ignoring sample size initially)
        if true_curves_np.shape[1:] != generated_curves_np.shape[1:]:
             logging.error(f"Shape mismatch between true and generated curves (excluding sample size): {true_curves_np.shape[1:]} vs {generated_curves_np.shape[1:]}")
             sys.exit(1)

        # Reshape if necessary (e.g., if saved as (samples, length)) to (samples, length, channels)
        if true_curves_np.ndim == 2 and fe_curve_channels == 1:
             true_curves_np = true_curves_np.reshape(true_curves_np.shape[0], true_curves_np.shape[1], 1)
        if generated_curves_np.ndim == 2 and fe_curve_channels == 1:
             generated_curves_np = generated_curves_np.reshape(generated_curves_np.shape[0], generated_curves_np.shape[1], 1)


        if true_curves_np.shape[-1] != fe_curve_channels or generated_curves_np.shape[-1] != fe_curve_channels:
             logging.error(f"Channel mismatch. Expected {fe_curve_channels}, got {true_curves_np.shape[-1]} and {generated_curves_np.shape[-1]}.")
             sys.exit(1)

        if true_curves_np.shape[1] != fe_curve_length or generated_curves_np.shape[1] != fe_curve_length:
             logging.warning(f"Curve length mismatch. Expected {fe_curve_length}, got {true_curves_np.shape[1]} and {generated_curves_np.shape[1]}.")


        logging.info(f"Loaded {len(true_curves_np)} true and {len(generated_curves_np)} generated F-E curves.")

    except FileNotFoundError:
         logging.error(f"True or generated data file not found.")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading true or generated data: {e}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving analysis results to {output_dir}")

    # --- Evaluation Metrics ---
    logging.info("Calculating evaluation metrics...")

    # Curve Shape Metrics
    r2_curves = calculate_r2(true_curves_np, generated_curves_np)
    rel_l2_error = calculate_relative_l2_error(true_curves_np, generated_curves_np)
    # DTW can be computationally expensive - run optionally
    # dtw_distance = calculate_dtw_distance(true_curves_np, generated_curves_np)


    logging.info(f"Curve Shape Metrics:")
    logging.info(f"  Overall R^2: {r2_curves:.4f}")
    logging.info(f"  Average Relative L2 Error: {rel_l2_error:.4f}")
    # logging.info(f"  Average DTW Distance: {dtw_distance:.4f}")

    # Mechanical Property Evaluation
    property_evaluation_params = analysis_config.get('find_peaks', {}) # Get peak params for evaluation
    property_metrics = evaluate_mechanical_properties(
        true_curves_np,
        generated_curves_np,
        property_extraction_params={'find_peaks': property_evaluation_params}
    )

    logging.info("\nMechanical Property Evaluation Metrics:")
    metrics_summary = {}
    for prop_name, metrics in property_metrics.items():
        logging.info(f"Property: {prop_name}")
        metrics_summary[prop_name] = metrics
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")

    # Save metrics to a file (e.g., JSON or CSV)
    metrics_output_path = os.path.join(output_dir, 'evaluation_metrics.json')
    try:
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logging.info(f"Evaluation metrics saved to {metrics_output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {metrics_output_path}: {e}")


    # --- Visualization ---
    logging.info("Generating visualizations...")

    # Plot comparison for a few samples
    num_samples_to_plot = min(plot_samples, true_curves_np.shape[0], generated_curves_np.shape[0])
    logging.info(f"Plotting comparisons for {num_samples_to_plot} samples.")

    # You need an extension axis for plotting.
    # If your data is standardized on [0, 1], you might plot against index or [0, 1].
    # If you saved/can infer the original physical extension, use that.
    # Assuming standardized curves, let's plot against index for simplicity or infer a dummy axis.
    fe_len = true_curves_np.shape[1]
    extension_axis_for_plotting = np.arange(fe_len) # Default to index

    # If you have information about the original extension range, you could create a scaled axis
    # Example: Assuming standardization mapped original_max_ext to fe_curve_length indices
    # original_max_ext = data_cfg.get('original_max_extension') # Need to store this during preprocessing
    # if original_max_ext is not None and fe_len > 1:
    #      extension_axis_for_plotting = np.linspace(0, original_max_ext, fe_len)


    peak_params_for_plotting = analysis_config.get('find_peaks', {}) # Use the same peak params for plotting

    for i in range(num_samples_to_plot):
        plot_fe_curve_comparison(
            true_curves_np[i, :, 0], # Pass 1D force array
            generated_curves_np[i, :, 0],
            sample_idx=i,
            extension_axis=extension_axis_for_plotting,
            title="Generated vs. True F-E Curve",
            show_peaks=True,
            peak_params=peak_params_for_plotting
        )
        # Save the plot instead of showing directly
        plt.savefig(os.path.join(output_dir, f'comparison_plot_sample_{i:03d}.png'))
        plt.close() # Close plot to free memory

    # Plot multiple generated curves for the same input (if applicable)
    # This requires knowing which generated curves correspond to the same input.
    # If you generated N curves per input, your generated_data_path should reflect this.
    # E.g., if you generated 100 total curves from 10 distinct inputs (10 per input),
    # the first 10 generated curves might correspond to the first input in your generation input file.
    # This is a placeholder assuming the first `plot_samples` generated curves are for the same input conceptually.
    if generated_curves_np.shape[0] > 1:
        logging.info(f"Plotting multiple generated curves (showing first {min(plot_samples, generated_curves_np.shape[0])}).")
        # You might need the average true curve corresponding to these inputs
        # This requires aligning true data with the inputs used for generation.
        plot_multiple_generated_curves(
            generated_curves_np[:min(plot_samples, generated_curves_np.shape[0])], # Show a subset
            # true_curve_avg=... # Provide corresponding average true curve if available
            extension_axis=extension_axis_for_plotting,
            title="Multiple Generated F-E Curves (Sample Subset)"
        )
        plt.savefig(os.path.join(output_dir, 'multiple_generated_curves.png'))
        plt.close()


    # Plot property distributions
    logging.info("Plotting property distributions...")
    # Need the extracted property lists (true and generated)
    # These are calculated within evaluate_mechanical_properties, but not returned.
    # Modify evaluate_mechanical_properties to return the lists, or re-extract them here.
    # Re-extracting here for clarity, but less efficient.

    # Re-extract properties for plotting distributions
    true_props_for_plot = {}
    gen_props_for_plot = {}
    all_prop_names = ['unfolding_energy', 'max_force', 'num_peaks', 'avg_unfolding_force'] # List all properties you evaluate

    for prop_name in all_prop_names:
        true_props_for_plot[prop_name] = []
        gen_props_for_plot[prop_name] = []

    for i in range(true_curves_np.shape[0]): # Assuming same number of true/gen samples
         true_curve_1d = true_curves_np[i, :, 0]
         gen_curve_1d = generated_curves_np[i, :, 0]

         # Extract properties (using dummy extension step for energy if physical ext is unknown)
         true_props_for_plot['unfolding_energy'].append(calculate_unfolding_energy(true_curve_1d, extension_step=1.0))
         true_props_for_plot['max_force'].append(calculate_max_force(true_curve_1d))
         true_peak_indices, _ = find_force_peaks(true_curve_1d, **peak_params_for_plotting)
         true_props_for_plot['num_peaks'].append(len(true_peak_indices))
         true_unfolding_forces = _.get('peak_heights', [])
         true_props_for_plot['avg_unfolding_force'].append(np.mean(true_unfolding_forces) if len(true_unfolding_forces) > 0 else np.nan)


         gen_props_for_plot['unfolding_energy'].append(calculate_unfolding_energy(gen_curve_1d, extension_step=1.0))
         gen_props_for_plot['max_force'].append(calculate_max_force(gen_curve_1d))
         gen_peak_indices, _ = find_force_peaks(gen_curve_1d, **peak_params_for_plotting)
         gen_props_for_plot['num_peaks'].append(len(gen_peak_indices))
         gen_unfolding_forces = _.get('peak_heights', [])
         gen_props_for_plot['avg_unfolding_force'].append(np.mean(gen_unfolding_forces) if len(gen_unfolding_forces) > 0 else np.nan)


    # Plot distributions for each property
    for prop_name in all_prop_names:
        plot_property_distributions(true_props_for_plot, gen_props_for_plot, prop_name)
        plt.savefig(os.path.join(output_dir, f'distribution_plot_{prop_name}.png'))
        plt.close()


    logging.info("Analysis and visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and evaluate generated F-E curves.")
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to the data configuration YAML file (for analysis parameters).')
    parser.add_argument('--true_data', type=str, required=True,
                        help='Path to the file containing true F-E curves (processed format).')
    parser.add_argument('--generated_data', type=str, required=True,
                        help='Path to the file containing generated F-E curves.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save analysis results and plots.')
    parser.add_argument('--plot_samples', type=int, default=5,
                        help='Number of individual curve comparisons to plot.')

    args = parser.parse_args()

    main(args.data_config, args.true_data, args.generated_data, args.output_dir, args.plot_samples)