# src/inference/generator.py

import torch
import os
import logging
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

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
        model = ConditionalDiffusionModel(
            fe_curve_length=model_config['fe_curve_length'],
            fe_curve_channels=model_config.get('fe_curve_channels', 1),
            protein_input_dim=model_config['protein_input_dim'],
            protein_embed_dim=model_config['protein_embed_dim'],
            protein_encoding_type=model_config['protein_encoding_type'],
            protein_plm_name=model_config.get('protein_plm_name'),
            protein_plm_embed_dim=model_config.get('protein_plm_embed_dim'),
            protein_freeze_plm=model_config.get('protein_freeze_plm', True),
            condition_input_dim=model_config['condition_input_dim'],
            condition_embed_dim=model_config['condition_embed_dim'],
            time_embed_dim=model_config['time_embed_dim'],
            model_channels=model_config['model_channels'],
            num_diffusion_steps=model_config['num_diffusion_steps'],
            beta_schedule=model_config.get('beta_schedule', 'linear')
            # Add any other specific model parameters from config
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
        generated_curves = model.generate(
            sequence_data=seq_input_for_generation, # Pass the potentially duplicated inputs
            conditions=cond_input_for_generation,
            num_samples=len(seq_input_for_generation), # The model's generate expects num_samples as batch size
            device=device
        )

    logging.info(f"Finished generating curves. Output shape: {generated_curves.shape}")

    return generated_curves


# Example Usage
if __name__ == "__main__":
    print("--- Testing generator.py ---")

    # This example requires:
    # 1. A dummy model configuration dictionary that matches the structure expected by load_model_from_checkpoint
    # 2. A dummy checkpoint file (which you'd normally get from training)
    # 3. Dummy processed sequence data and condition data (tensors or lists, matching the model's expected input)

    # --- Create Dummy Model Config and Checkpoint ---
    # In a real scenario, this config would be loaded from a file (e.g., config/model_config.yaml)
    # and the checkpoint path would point to your trained model file.

    # Define dummy model parameters that match the DummyDiffusionModel used in trainer.py example
    fe_len = 200
    seq_input_dim = 100 # Matches DummyProteinEncoder expected input
    seq_embed_dim = 128
    cond_input_dim = 2
    cond_embed_dim = 64
    time_embed_dim = 128
    model_channels = 256
    num_diff_steps = 100

    dummy_model_config = {
        'fe_curve_length': fe_len,
        'fe_curve_channels': 1,
        'protein_input_dim': seq_input_dim,
        'protein_embed_dim': seq_embed_dim,
        'protein_encoding_type': 'pretrained_embeddings', # Must match how data is prepared
        'condition_input_dim': cond_input_dim,
        'condition_embed_dim': cond_embed_dim,
        'time_embed_dim': time_embed_dim,
        'model_channels': model_channels,
        'num_diffusion_steps': num_diff_steps,
        'beta_schedule': 'linear'
    }

    # Create a dummy model and save a dummy checkpoint
    # This replaces loading a real trained model for demonstration purposes
    class DummyProteinEncoder(torch.nn.Module):
         def __init__(self, input_dim, output_dim, encoding_type, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
            self.encoding_type = encoding_type # Needed by the model's generate method
         def forward(self, x): return self.fc(x)

    class DummyConditionEncoder(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
        def forward(self, x): return self.fc(x)

    class DummyTimeEmbedding(torch.nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.fc = torch.nn.Linear(embed_dim, embed_dim)
        def forward(self, t): return self.fc(torch.randn(t.size(0), self.fc.in_features, device=t.device))

    class DummyDenoisingModel(torch.nn.Module):
         def __init__(self, fe_curve_length, fe_curve_channels, protein_embed_dim, condition_embed_dim, time_embed_dim, model_channels):
            super().__init__()
            self.fe_len = fe_curve_length
            self.fe_channels = fe_curve_channels
            # Simple linear layer that takes concatenated embeddings and returns output shape
            total_cond_dim = protein_embed_dim + condition_embed_dim + time_embed_dim
            self.fc = torch.nn.Linear(fe_curve_length * fe_curve_channels + total_cond_dim, fe_curve_length * fe_curve_channels)

         def forward(self, x_t, time_emb, protein_embedding, condition_embedding):
             # Flatten and concatenate for dummy
             batch_size = x_t.size(0)
             x_t_flat = x_t.view(batch_size, -1)
             protein_flat = protein_embedding.view(batch_size, -1)
             condition_flat = condition_embedding.view(batch_size, -1)
             time_flat = time_emb.view(batch_size, -1)
             combined_input = torch.cat([x_t_flat, protein_flat, condition_flat, time_flat], dim=-1)

             output_flat = self.fc(combined_input)
             # Reshape to output shape
             output = output_flat.view(batch_size, self.fe_len, self.fe_channels)
             return output # Simulating noise prediction output


    class DummyDiffusionModelForGenTest(torch.nn.Module):
         def __init__(self, fe_curve_length, fe_curve_channels, protein_input_dim, protein_embed_dim, protein_encoding_type, condition_input_dim, condition_embed_dim, time_embed_dim, model_channels, num_diffusion_steps, beta_schedule):
             super().__init__()
             self.fe_curve_length = fe_curve_length
             self.fe_curve_channels = fe_curve_channels
             self.num_diffusion_steps = num_diffusion_steps
             self.protein_encoding_type = protein_encoding_type # Store for generate method
             self.condition_input_dim = condition_input_dim # Store for input check

             # Dummy diffusion schedule terms (needed for generate)
             self.betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
             self.alphas = 1.0 - self.betas
             self.alphas_bar = torch.cumprod(self.alphas, dim=0)
             self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
             self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
             self.posterior_variance = self.betas * (1.0 - self.alphas_bar[:-1]) / (1.0 - self.alphas_bar)