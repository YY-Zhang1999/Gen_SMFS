# scripts/generate.py

import os
from typing import Dict, Any

import yaml
import argparse
import logging
import torch
import pandas as pd
import numpy as np

# Ensure src is in PYTHONPATH
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import yaml
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from utils import load_config, creat_dataset

from Gen_SMFS.src.data_processing.dataset import FEDataset
from Gen_SMFS.src.training.trainer import DiffusionModelTrainer
from Gen_SMFS.src.models import create_diffusion, DiT1D
from Gen_SMFS.src.data_processing.utils import load_raw_data, save_processed_data
from Gen_SMFS.src.inference.generator import load_model_from_checkpoint, generate_fe_curves

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def main(data_config_path: str, model_config_path: str, checkpoint_path: str, output_path: str, num_samples_per_input: int, device: str):
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
        model = load_model_from_checkpoint(checkpoint_path, model_cfg, device=device)
    except Exception as e:
        logging.error(f"Failed to load trained model from {checkpoint_path}: {e}")
        sys.exit(1)

    # --- Data Loading and Preparation ---
    full_dataset = creat_dataset(data_cfg)

    conditions_data_for_gen = full_dataset.get_condition(1)

    # --- Generate Curves ---
    logging.info(f"Generating {num_samples_per_input} curves for each of {len(conditions_data_for_gen)} input samples.")
    try:
        generated_curves_tensor = generate_fe_curves(
            diffusion=create_diffusion(timestep_respacing=model_cfg["timestep_respacing"]),
            model=model,
            sequence_data=None,
            conditions=conditions_data_for_gen,
            num_samples_per_input=num_samples_per_input,
            device=device,
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

    fe_curve = full_dataset.__getitem__(0)['fe_curve']
    plt.plot(fe_curve.reshape(-1), label='original')
    for i in range(len(generated_curves_np)):
        curve = generated_curves_np[i]
        plt.plot(curve.reshape(-1))
    plt.legend()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein unfolding F-E curves using a trained Diffusion Model.")
    parser.add_argument('--data_config', type=str,
                        default='..\\config\\simulation_data_config.yaml',
                        help='Path to the data configuration YAML file.')
    parser.add_argument('--model_config', type=str,
                        default='..\\config\\diffusion_model_config.yaml',
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--checkpoint', type=str,
                        default='.\\checkpoints\\DiffusionModelTrainer\\best_model.pt',
                        help='Path to the trained model checkpoint file (.pt).')
    parser.add_argument('--output', type=str,
                        default='./output/output.npy',
                        help='Path to save the generated F-E curves (e.g., .npy, .pkl).')
    parser.add_argument('--num_samples_per_input', type=int, default=4,
                        help='Number of curves to generate for each input sequence and condition pair.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for generation (e.g., "cuda", "cpu").')

    args = parser.parse_args()

    main(args.data_config, args.model_config, args.checkpoint, args.output, args.num_samples_per_input, args.device)