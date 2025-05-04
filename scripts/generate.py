# scripts/generate.py

import os
import yaml
import argparse
import logging
import torch
import pandas as pd
import numpy as np

# Ensure src is in PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from models.diffusion_model import ConditionalDiffusionModel
    from inference.generator import load_model_from_checkpoint, generate_fe_curves
    from data_processing.utils import load_raw_data, save_processed_data
    # You might need preprocessing functions if input data is raw
    from data_processing.preprocessing import encode_protein_sequences, encode_conditions
    # And the Dataset class if loading processed data using its logic
    from data_processing.dataset import FEDataset # For context/potential loading
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


def main(data_config_path: str, model_config_path: str, checkpoint_path: str, input_data_path: str, output_path: str, num_samples_per_input: int, device: str):
    """
    Main function to load a trained model and generate F-E curves.
    """
    data_cfg = load_config(data_config_path)
    model_cfg = load_config(model_config_path)

    # --- Load Trained Model ---
    device = torch.device(device)
    logging.info(f"Using device for generation: {device}")

    try:
        # Need to pass model_config (or relevant parts) to load_model_from_checkpoint
        # to correctly instantiate the model architecture.
        # You should save the model config alongside your checkpoint during training.
        # For now, we load the config from the file path.
        model = load_model_from_checkpoint(checkpoint_path, model_config=model_cfg, device=device)
    except Exception as e:
        logging.error(f"Failed to load trained model from {checkpoint_path}: {e}")
        sys.exit(1)

    # --- Load Input Data for Generation ---
    # This input data file should contain the protein sequences and conditions
    # for which you want to generate curves.
    # It could be a file with raw sequences or processed/encoded inputs,
    # depending on how your model's ProteinEncoder expects input during generate().

    # Assuming the input_data_path points to a CSV file containing:
    # - Protein sequences (in the format expected by your ProteinEncoder)
    # - Conditions (matching the columns specified in data_config)

    try:
        input_df = load_raw_data(input_data_path) # Use load_raw_data for flexibility
        logging.info(f"Loaded input data for generation from {input_data_path} ({len(input_df)} samples).")

        # Extract sequences and conditions based on data_config
        preprocessing_params = data_cfg.get('data_preprocessing', {})
        sequence_col = preprocessing_params.get('sequence_column', 'sequence') # Assuming default column name
        condition_cols = preprocessing_params.get('condition_columns')

        if sequence_col not in input_df.columns:
             logging.error(f"Input data file missing sequence column '{sequence_col}'.")
             sys.exit(1)
        if condition_cols is None or not all(col in input_df.columns for col in condition_cols):
             logging.error(f"Input data file missing required condition columns: {condition_cols}")
             sys.exit(1)


        # Prepare sequence data for the model's generate method
        seq_encoding_type = model_cfg.get('protein_encoding_type') # Get type from model config

        if seq_encoding_type == 'raw':
            # Model's generate method expects a list of raw strings
            sequence_data_for_gen = input_df[sequence_col].tolist()
            logging.info("Input sequences loaded as raw strings.")
        elif seq_encoding_type in ['onehot', 'pretrained_embeddings']:
            # Model's generate method expects a tensor of pre-encoded features.
            # This means the input_data_path should ideally point to a file
            # that *already* contains the encoded features, or you need to encode them here.

            # --- Placeholder: Encoding here for flexibility, but better if done offline ---
            logging.warning(f"Input sequence encoding type is '{seq_encoding_type}'. Assuming raw strings in input_data_path and encoding them now.")
            # You'd need to use encode_protein_sequences, which expects List[str]
            raw_sequences_from_input_file = input_df[sequence_col].tolist()

            # Need the *exact* same encoding process as preprocessing/training
            # This requires accessing preprocessing parameters like seq_len, alphabet size, PLM details etc.
            # which are in data_config.

            # For pre-trained embeddings, you'd typically load a *pre-encoded* feature file here.
            # For demonstration, let's assume the input file *is* the processed features file.
            # This is why having processed input data ready is cleaner for inference.

            # --- More Realistic Approach: Load from processed input file ---
            logging.info(f"Attempting to load pre-encoded sequences assuming '{input_data_path}' is the processed sequences file.")
            # Use save_processed_data and load_raw_data (which loads CSV/Excel)
            # or np.load if saved as .npy to load the tensor.
            # Assuming the input_data_path points to a CSV with numerical features for simplicity in this test
            try:
                 # Assuming input_df *is* the pre-encoded features DataFrame/array
                 # Need to identify the feature columns again, similar to Dataset loading
                 feature_columns = [col for col in input_df.columns if input_df[col].dtype in [np.number, np.bool_]]
                 if not feature_columns:
                      raise ValueError("No numeric/boolean columns found in input data file for encoded sequences.")
                 encoded_sequences_data = input_df[feature_columns].values
                 sequence_data_for_gen = torch.tensor(encoded_sequences_data, dtype=torch.float32)
                 logging.info(f"Loaded pre-encoded sequences with shape {sequence_data_for_gen.shape}.")

            except Exception as e:
                 logging.error(f"Error loading/interpreting input sequence data as encoded features: {e}")
                 logging.error("Please ensure input_data_path provides data in the format expected by the model's ProteinEncoder based on protein_encoding_type.")
                 sys.exit(1)

        else:
             logging.error(f"Unsupported protein_encoding_type '{seq_encoding_type}' specified in model config.")
             sys.exit(1)


        # Prepare condition data for the model's generate method
        conditions_data_for_gen = torch.tensor(input_df[condition_cols].values, dtype=torch.float32)
        if conditions_data_for_gen.size(-1) != model_cfg.get('condition_input_dim'):
             logging.error(f"Input condition dimension mismatch. Expected {model_cfg.get('condition_input_dim')}, got {conditions_data_for_gen.size(-1)}. Check condition_columns in data_config.")
             sys.exit(1)
        logging.info(f"Input conditions loaded with shape {conditions_data_for_gen.shape}.")


    except FileNotFoundError:
         logging.error(f"Input data file for generation not found at {input_data_path}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading or processing input data for generation: {e}")
        sys.exit(1)


    # --- Generate Curves ---
    logging.info(f"Generating {num_samples_per_input} curves for each of {len(input_df)} input samples.")
    try:
        generated_curves_tensor = generate_fe_curves(
            model=model,
            sequence_data=sequence_data_for_gen,
            conditions=conditions_data_for_gen,
            num_samples_per_input=num_samples_per_input,
            device=device
        )
        logging.info(f"Generation complete. Output tensor shape: {generated_curves_tensor.shape}")

    except Exception as e:
        logging.error(f"An error occurred during curve generation: {e}")
        sys.exit(1)

    # --- Save Generated Curves ---
    try:
        # Convert generated tensor to numpy array for saving
        generated_curves_np = generated_curves_tensor.cpu().numpy()

        # Suggest saving as .npy or .pkl
        if not output_path.endswith(('.npy', '.pkl', '.csv')):
             logging.warning(f"Output path '{output_path}' does not have a standard extension (.npy, .pkl, .csv). Saving as .npy.")
             output_path += '.npy' # Append .npy extension

        save_processed_data(generated_curves_np, output_path)
        logging.info(f"Generated curves saved to {output_path}")

    except Exception as e:
        logging.error(f"Error saving generated curves to {output_path}: {e}")
        sys.exit(1)

    logging.info("Generation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein unfolding F-E curves using a trained Diffusion Model.")
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to the data configuration YAML file.')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint file (.pt).')
    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to the file containing input sequences and conditions for generation (e.g., CSV).')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the generated F-E curves (e.g., .npy, .pkl).')
    parser.add_argument('--num_samples_per_input', type=int, default=1,
                        help='Number of curves to generate for each input sequence and condition pair.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for generation (e.g., "cuda", "cpu").')

    args = parser.parse_args()

    main(args.data_config, args.model_config, args.checkpoint, args.input_data, args.output, args.num_samples_per_input, args.device)