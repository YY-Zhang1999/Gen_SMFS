# scripts/preprocess_data.py

import os
from typing import Dict, Any

import yaml
import argparse
import logging
import pandas as pd
import numpy as np

# Ensure src is in PYTHONPATH if running script directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from Gen_SMFS.src.data_processing import read_simulation_data, standardize_fe_curve, preprocess_fe_curves, encode_protein_sequences, encode_conditions
from Gen_SMFS.src.data_processing import load_raw_data, save_processed_data, calculate_contour_length
from Gen_SMFS.src.data_processing import FEDataset # To check expected output format
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def main(config_path: str):
    """
    Main function to run the data preprocessing pipeline.
    """
    config = load_config(config_path)
    data_config = config.get('data_paths', {})
    preprocessing_config = config.get('data_preprocessing', {})
    analysis_config = config.get('analysis_params', {}) # Might need analysis params for property extraction during eval

    # --- Validate Configuration ---
    if not data_config.get('raw_data_dir') or not data_config.get('processed_data_dir'):
         logging.error("raw_data_dir and processed_data_dir must be specified in data_paths.")
         sys.exit(1)
    if not preprocessing_config.get('fe_curve_length') or not preprocessing_config.get('seq_encoding_type') or not preprocessing_config.get('condition_columns'):
         logging.error("fe_curve_length, seq_encoding_type, and condition_columns must be specified in data_preprocessing.")
         sys.exit(1)

    raw_data_dir = data_config['raw_data_dir']
    processed_data_dir = data_config['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)

    # --- Load Raw Data ---
    # This part is highly dependent on YOUR raw data file structure.
    # You might have one large file or many small files.
    # This is a placeholder assuming a single raw data file is specified in config.
    Fu_data_df, xp_data_df, WLC_data_df = read_simulation_data(raw_data_dir, 14, 500)

    raw_fe_data_list = []
    for col in range(len(Fu_data_df)):
        raw_fe_data_list.append({
            'extension': np.array(xp_data_df.iloc[col][:-4].values, dtype=np.float32),
            'force': np.array(Fu_data_df.iloc[col][:-4].values, dtype=np.float32)
        })

    # --- Preprocess F-E Curves ---
    target_curve_length = preprocessing_config['fe_curve_length']
    norm_strategy = preprocessing_config['fe_curve_normalization_strategy']
    global_max_force = WLC_data_df['Force'].max()
    force_unit_conversion_to_pN = preprocessing_config.get('force_unit_conversion_to_pN', 1.0)


    # Apply force unit conversion during standardization if specified
    # Need to modify preprocess_fe_curves or standardize_fe_curve to accept conversion factor

    # Let's update standardize_fe_curve definition to include the conversion factor if needed
    # (Assuming this change was made in data_processing/preprocessing.py)
    # In the meantime, you'd have to apply it manually here before calling preprocess_fe_curves
    # Example: raw_fe_data_list_converted = [{'extension': d['extension'], 'force': d['force'] * force_unit_conversion_to_pN} for d in raw_fe_data_list]
    # And then call preprocess_fe_curves with norm_strategy='max_force' and global_max_force in pN units

    # For now, proceed assuming force is already in desired units or handled by norm_factor
    # If force_unit_conversion_to_pN > 1.0, you should convert before normalization
    if force_unit_conversion_to_pN != 1.0:
         logging.warning(f"Applying force unit conversion factor: {force_unit_conversion_to_pN}")
         for item in raw_fe_data_list:
              item['force'] = item['force'] * force_unit_conversion_to_pN


    processed_fe_curves_list = preprocess_fe_curves(
        raw_fe_data_list,
        target_curve_length=target_curve_length,
        normalization_strategy=norm_strategy,
        global_max_force=global_max_force, # Should be in units after conversion if applicable
        # Add force_unit_conversion to standardize_fe_curve if implemented there
    )

    # Convert list of arrays to a single numpy array for saving
    processed_fe_curves_array = np.array(processed_fe_curves_list)
    processed_fe_curves_array = np.nan_to_num(processed_fe_curves_array, nan=0.0)

    if len(processed_fe_curves_array.shape) <= 2:
        processed_fe_curves_array = np.expand_dims(processed_fe_curves_array, -1)


    # --- Encode Conditions ---
    condition_cols = preprocessing_config['condition_columns']
    # Convert list of condition dicts to DataFrame to use encode_conditions
    conditions_df = WLC_data_df

    processed_conditions_array = encode_conditions(
        conditions_df,
        condition_columns=condition_cols
        # Add scaling parameters if implemented in encode_conditions
    )



    print(processed_conditions_array.shape, processed_fe_curves_array.shape)


    # --- Save Processed Data ---
    processed_fe_curves_path = os.path.join(processed_data_dir, data_config.get('processed_fe_curves_file', 'fe_curves.npy'))
    processed_conditions_path = os.path.join(processed_data_dir, data_config.get('processed_conditions_file', 'conditions.npy')) # Suggest .npy

    save_processed_data(processed_fe_curves_array, processed_fe_curves_path)
    save_processed_data(processed_conditions_array, processed_conditions_path)

    logging.info("Preprocessing complete. Processed data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw protein unfolding data.")
    parser.add_argument('--config', type=str, required=True,
                        default='../config/simulation_data_config.yaml',
                        help='Path to the data configuration YAML file.')
    args = parser.parse_args()

    main(args.config)