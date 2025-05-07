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

try:
    from data_processing.preprocessing import standardize_fe_curve, preprocess_fe_curves, encode_protein_sequences, encode_conditions
    from data_processing.utils import load_raw_data, save_processed_data, calculate_contour_length
    from data_processing.dataset import FEDataset # To check expected output format
except ImportError as e:
    logging.error(f"Failed to import data processing modules: {e}")
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
    raw_input_file = data_config.get('raw_input_file')
    if raw_input_file:
        raw_data_path = os.path.join(raw_data_dir, raw_input_file)
        raw_df = load_raw_data(raw_data_path)
        logging.info(f"Loaded {len(raw_df)} rows from {raw_data_path}")

        # Assuming raw_df contains columns for 'extension', 'force', 'sequence', and condition_columns
        # You will need to adapt column names to match your actual data.
        # Example column names:
        extension_col = 'extension'
        force_col = 'force'
        sequence_col = 'sequence'
        condition_cols = preprocessing_config['condition_columns']

        if not all(col in raw_df.columns for col in [extension_col, force_col, sequence_col] + condition_cols):
             logging.error(f"Raw data file missing required columns. Expected: {extension_col}, {force_col}, {sequence_col}, {condition_cols}")
             sys.exit(1)

        # Assuming each row is a data point within a single curve.
        # You need to group data points by curve (e.g., using a 'curve_id' column).
        # This requires significant adaptation based on your raw data format.
        # --- Placeholder: Assuming raw_df is already structured per curve for simplicity ---
        # Example: If each row is a full curve (less common for SMFS) - NOT TYPICAL
        # If raw_df has columns like 'curve_1_ext', 'curve_1_force', etc. - NOT TYPICAL

        # --- More Realistic Placeholder: Assuming raw_df contains all points, grouped by ID ---
        # Assuming raw_df has columns: 'curve_id', 'extension', 'force', 'sequence', condition_cols...
        curve_id_col = 'curve_id' # You need a column to identify individual curves
        if curve_id_col not in raw_df.columns:
             logging.error(f"Raw data file must contain a '{curve_id_col}' column to group data points into curves.")
             sys.exit(1)

        logging.info(f"Grouping raw data by '{curve_id_col}'...")
        grouped_curves = raw_df.groupby(curve_id_col)

        raw_fe_data_list = []
        protein_sequences_list = []
        conditions_list = [] # Store as list of dicts/series initially
        processed_curve_ids = []

        # Iterate through each curve group
        for curve_id, curve_data in grouped_curves:
             # Ensure curve data is sorted by extension if necessary
             curve_data = curve_data.sort_values(by=extension_col)

             raw_fe_data_list.append({
                 'extension': curve_data[extension_col].values,
                 'force': curve_data[force_col].values
             })

             # Assuming sequence and conditions are the same for all points in a curve
             # Take the first value for sequence and conditions for this curve
             protein_sequences_list.append(curve_data[sequence_col].iloc[0])
             conditions_list.append(curve_data[condition_cols].iloc[0].to_dict())
             processed_curve_ids.append(curve_id)


        logging.info(f"Extracted {len(raw_fe_data_list)} raw F-E curves.")

    else:
        logging.error("raw_input_file must be specified in data_paths to load raw data.")
        sys.exit(1)

    # --- Preprocess F-E Curves ---
    target_curve_length = preprocessing_config['fe_curve_length']
    norm_strategy = preprocessing_config['fe_curve_normalization_strategy']
    global_max_force = preprocessing_config.get('global_max_force')
    force_unit_conversion_to_pN = preprocessing_config.get('force_unit_conversion_to_pN', 1.0)


    # Pass protein sequences for contour length normalization if needed
    sequences_for_norm = protein_sequences_list if norm_strategy == 'contour_length' else None

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
        protein_sequences=sequences_for_norm
        # Add force_unit_conversion to standardize_fe_curve if implemented there
    )

    # Convert list of arrays to a single numpy array for saving
    processed_fe_curves_array = np.array(processed_fe_curves_list)

    # --- Encode Protein Sequences ---
    seq_encoding_type = preprocessing_config['seq_encoding_type']

    # If encoding_type is 'raw', encode_protein_sequences just returns the list of strings.
    # If it's 'onehot' or 'pretrained_embeddings', it returns a list/array of encoded features.
    # The actual PLM encoding for 'raw' might happen later in the ProteinEncoder module.
    # For saving purposes here, we save based on the returned type.

    # Check if input_dim is needed for the encoder based on encoding type
    protein_encoder_input_dim = None
    if seq_encoding_type == 'pretrained_embeddings':
        protein_encoder_input_dim = preprocessing_config.get('protein_input_dim')
        if protein_encoder_input_dim is None:
             logging.error("protein_input_dim must be specified in data_preprocessing for 'pretrained_embeddings' type.")
             sys.exit(1)
    elif seq_encoding_type == 'onehot':
         protein_encoder_input_dim = preprocessing_config.get('protein_input_dims_onehot')
         if protein_encoder_input_dim is None or not isinstance(protein_encoder_input_dim, list) or len(protein_encoder_input_dim) != 2:
              logging.error("protein_input_dims_onehot must be specified as a list [seq_len, alphabet_size] in data_preprocessing for 'onehot' type.")
              sys.exit(1)
         protein_encoder_input_dim = tuple(protein_encoder_input_dim)


    # Encode sequences - This function returns raw strings for 'raw' type, encoded data otherwise
    encoded_sequences_data = encode_protein_sequences(
        protein_sequences_list,
        encoding_type=seq_encoding_type
        # Add PLM-specific parameters if encoding_type is 'raw' and you implement PLM loading here
        # (though PLM encoding is often deferred to the model)
    )

    # For 'raw', encoded_sequences_data is still a list of strings.
    # For 'onehot' or 'pretrained_embeddings', it's a list/array of numpy arrays.
    # We need to store this in a format loadable by the Dataset.
    # Saving as CSV with one column 'sequence' for 'raw', or as numpy array/CSV for encoded features.
    if seq_encoding_type == 'raw':
         processed_sequences_df = pd.DataFrame({'sequence': encoded_sequences_data})
         sequences_output_file = data_config.get('processed_sequences_file', 'sequences.csv')
    else:
         # Assuming encoded_sequences_data is List[np.ndarray] or np.ndarray
         # Convert to numpy array if it's a list of arrays (and they have uniform shape)
         if isinstance(encoded_sequences_data, list) and len(encoded_sequences_data) > 0 and isinstance(encoded_sequences_data[0], np.ndarray):
              try:
                 encoded_sequences_data = np.array(encoded_sequences_data)
              except ValueError:
                  logging.error("Encoded sequences have inconsistent shapes and cannot be converted to a single numpy array. Please check preprocessing.")
                  sys.exit(1)

         if isinstance(encoded_sequences_data, np.ndarray):
              # Save numpy array
              processed_sequences_df = encoded_sequences_data # Keep as array for saving
              sequences_output_file = data_config.get('processed_sequences_file', 'sequences.npy') # Suggest .npy
         else:
              logging.error(f"Unsupported format for encoded sequences data ({type(encoded_sequences_data)}) after encoding.")
              sys.exit(1)


    # --- Encode Conditions ---
    condition_cols = preprocessing_config['condition_columns']
    # Convert list of condition dicts to DataFrame to use encode_conditions
    conditions_df = pd.DataFrame(conditions_list)

    processed_conditions_array = encode_conditions(
        conditions_df,
        condition_columns=condition_cols
        # Add scaling parameters if implemented in encode_conditions
    )

    # --- Save Processed Data ---
    processed_fe_curves_path = os.path.join(processed_data_dir, data_config.get('processed_fe_curves_file', 'fe_curves.npy'))
    processed_sequences_path = os.path.join(processed_data_dir, sequences_output_file)
    processed_conditions_path = os.path.join(processed_data_dir, data_config.get('processed_conditions_file', 'conditions.npy')) # Suggest .npy

    save_processed_data(processed_fe_curves_array, processed_fe_curves_path)
    save_processed_data(processed_sequences_df, processed_sequences_path) # df or array
    save_processed_data(processed_conditions_array, processed_conditions_path)

    logging.info("Preprocessing complete. Processed data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw protein unfolding data.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the data configuration YAML file.')
    args = parser.parse_args()

    main(args.config)