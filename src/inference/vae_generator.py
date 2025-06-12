# src/inference/generator.py

import torch
import os
import logging
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

from Gen_SMFS.src.models.vae import create_conditiaonl_vae_model
from Gen_SMFS.src.data_processing.dataset import FEDataset

# Import necessary modules from your project
try:
    from models.diffusion_model import ConditionalDiffusionModel
    from data_processing.dataset import FEDataset # For understanding data format
    from data_processing.preprocessing import encode_protein_sequences, encode_conditions # Might be needed if inputs are raw
    from data_processing.utils import save_processed_data, load_raw_data # For loading/saving

    # You will also need to import or define your model's specific configuration
    # For example, if you saved config with your model:
    # from config.model_config import ModelConfig

except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure src directory is in your PYTHONPATH or run from the project root.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_from_checkpoint(checkpoint_path: str, model_config: Dict[str, Any], device: Union[str, torch.device]) -> ConditionalDiffusionModel:
    """
    Loads a trained ConditionalDiffusionModel from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the model checkpoint file (.pt).
        model_config (Dict[str, Any]): Dictionary containing the model configuration
                                       used during training. This is necessary to
                                       re-instantiate the model correctly.
        device (Union[str, torch.device]): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        ConditionalDiffusionModel: The loaded and trained Diffusion Model.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If there is an error loading the model state dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found at: {checkpoint_path}")

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    device = torch.device(device)

    # Instantiate the model using the provided configuration
    # You might need to pass specific args from model_config depending on your model's __init__
    try:
        model = create_conditiaonl_vae_model(
        input_feature_dim=1,
        sequence_len=model_config['fe_curve_length'],
        latent_feature_dim=4,
        conditional_dim=1,
        scale_factor=4,
        backbone_type='transformer',
        use_crossattention=True,
    )
    except KeyError as e:
        logging.error(f"Missing key in model_config: {e}. Cannot instantiate model.")
        raise
    except Exception as e:
        logging.error(f"Error instantiating model from config: {e}")
        raise


    # Load the state dictionary
    try:
        # Load checkpoint might contain more than just model state dict
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if 'model_state_dict' key exists, otherwise assume the file is the state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model state dict from 'model_state_dict' key in {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded model state dict directly from {checkpoint_path}")


        model.to(device)
        model.eval() # Set to evaluation mode
        logging.info("Model loaded successfully and set to evaluation mode.")
        return model

    except Exception as e:
        logging.error(f"Error loading model state dict from {checkpoint_path}: {e}")
        raise RuntimeError(f"Failed to load model state dict: {e}")


def generate_fe_curves(
    model: ConditionalDiffusionModel,
    sequence_data: Union[torch.Tensor, List[str]],
    conditions: torch.Tensor,
    num_samples_per_input: int = 1, # Number of curves to generate for each (sequence, condition) pair
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Generates Force-Extension curves using the trained Diffusion Model.

    Args:
        model (ConditionalDiffusionModel): The trained Diffusion Model instance.
        sequence_data (Union[torch.Tensor, List[str]]): Input protein sequence data.
                                                         If model protein_encoding_type is 'raw', expects a list of strings.
                                                         Otherwise, expects a tensor of pre-encoded features.
                                                         Shape/length determines the number of distinct inputs.
        conditions (torch.Tensor): Input experimental condition data.
                                   Shape (num_inputs, num_condition_features).
                                   Number of rows must match the number of sequences.
        num_samples_per_input (int): The number of F-E curves to generate for each
                                     provided (sequence, condition) pair. Defaults to 1.
        device (Union[str, torch.device]): The device to perform generation on.

    Returns:
        torch.Tensor: A tensor of generated F-E curves.
                      Shape (num_inputs * num_samples_per_input, fe_curve_length, fe_curve_channels).
    """
    model.eval() # Ensure model is in evaluation mode
    device = torch.device(device)
    model.to(device)

    if num_samples_per_input < 1:
        raise ValueError("num_samples_per_input must be at least 1.")

    # Determine the number of distinct inputs
    num_inputs = len(sequence_data) if isinstance(sequence_data, list) else sequence_data.size(0)
    if num_inputs != conditions.size(0):
         raise ValueError("Number of sequence data samples and condition samples must match.")

    if num_samples_per_input > 1:
        logging.info(f"Generating {num_samples_per_input} curves for each of the {num_inputs} inputs.")
        # Duplicate inputs if generating multiple samples per input
        # This requires careful handling depending on how your ProteinEncoder takes input
        # If it takes a list of strings, you need to duplicate the strings.
        # If it takes a tensor, you duplicate the tensor.

        if isinstance(sequence_data, list):
             duplicated_sequence_data = [item for item in sequence_data for _ in range(num_samples_per_input)]
        elif isinstance(sequence_data, torch.Tensor):
             duplicated_sequence_data = sequence_data.repeat_interleave(num_samples_per_input, dim=0)
        else:
             raise TypeError(f"Unsupported sequence_data type: {type(sequence_data)}")

        duplicated_conditions = conditions.repeat_interleave(num_samples_per_input, dim=0)
        logging.info(f"Duplicated inputs. Generating total of {len(duplicated_sequence_data)} curves.")

        seq_input_for_generation = duplicated_sequence_data
        cond_input_for_generation = duplicated_conditions

    else: # num_samples_per_input == 1
        logging.info(f"Generating 1 curve for each of the {num_inputs} inputs.")
        seq_input_for_generation = sequence_data
        cond_input_for_generation = conditions


    # Move inputs to the correct device if they are tensors
    if isinstance(seq_input_for_generation, torch.Tensor):
         seq_input_for_generation = seq_input_for_generation.to(device)
    cond_input_for_generation = cond_input_for_generation.to(device)


    with torch.no_grad():
        # Call the model's generate method
        generated_curves = model.predict(
            seq_input_for_generation, # Pass the potentially duplicated inputs
            cond_input_for_generation,
            num_samples=len(seq_input_for_generation), # The model's generate expects num_samples as batch size
            device=device
        )

    logging.info(f"Finished generating curves. Output shape: {generated_curves.shape}")

    return generated_curves

