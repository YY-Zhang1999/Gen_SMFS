# scripts/train.py

import os
import yaml
import argparse
import logging
import torch
from torch.utils.data import DataLoader

# Ensure src is in PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from data_processing.dataset import FEDataset
    from models.diffusion_model import ConditionalDiffusionModel
    from training.trainer import Trainer
    # Import specific collate functions if needed (e.g., for raw sequences)
    # from data_processing.collate_fn import collate_fn_raw_seq

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


def main(data_config_path: str, model_config_path: str, training_config_path: str, checkpoint_path: str = None):
    """
    Main function to set up and run the training process.
    """
    data_cfg = load_config(data_config_path)
    model_cfg = load_config(model_config_path)
    train_cfg = load_config(training_config_path)

    # --- Data Loading and Preparation ---
    data_paths = data_cfg.get('data_paths', {})
    preprocessing_params = data_cfg.get('data_preprocessing', {})

    processed_data_dir = data_paths.get('processed_data_dir')
    fe_curves_file = data_paths.get('processed_fe_curves_file', 'fe_curves.npy')
    sequences_file = data_paths.get('processed_sequences_file', 'sequences.csv') # Or .npy
    conditions_file = data_paths.get('processed_conditions_file', 'conditions.npy') # Or .csv

    fe_curves_path = os.path.join(processed_data_dir, fe_curves_file)
    sequences_path = os.path.join(processed_data_dir, sequences_file)
    conditions_path = os.path.join(processed_data_dir, conditions_file)

    if not all([os.path.exists(fe_curves_path), os.path.exists(sequences_path), os.path.exists(conditions_path)]):
         logging.error(f"Processed data files not found. Please run 'preprocess_data.py' first.")
         logging.error(f"Missing file(s): {[f for f in [fe_curves_path, sequences_path, conditions_path] if not os.path.exists(f)]}")
         sys.exit(1)


    # Determine condition_input_dim from data config
    condition_columns = preprocessing_params.get('condition_columns')
    if condition_columns is None:
         logging.error("condition_columns not specified in data_config.data_preprocessing.")
         sys.exit(1)
    model_cfg['condition_input_dim'] = len(condition_columns)

    # Determine protein_input_dim based on encoding type if not 'raw'
    seq_encoding_type = preprocessing_params.get('seq_encoding_type')
    if seq_encoding_type is None:
         logging.error("seq_encoding_type not specified in data_config.data_preprocessing.")
         sys.exit(1)

    if seq_encoding_type == 'pretrained_embeddings':
         model_cfg['protein_input_dim'] = preprocessing_params.get('protein_input_dim')
         if model_cfg['protein_input_dim'] is None:
              logging.error("protein_input_dim must be specified in data_config.data_preprocessing for 'pretrained_embeddings'.")
              sys.exit(1)
    elif seq_encoding_type == 'onehot':
         protein_input_dims_onehot = preprocessing_params.get('protein_input_dims_onehot')
         if protein_input_dims_onehot is None or not isinstance(protein_input_dims_onehot, list) or len(protein_input_dims_onehot) != 2:
              logging.error("protein_input_dims_onehot must be specified as a list [seq_len, alphabet_size] in data_config.data_preprocessing for 'onehot'.")
              sys.exit(1)
         # For a simple encoder that flattens one-hot, the input dim is the product
         model_cfg['protein_input_dim'] = protein_input_dims_onehot[0] * protein_input_dims_onehot[1]
         # Note: If your one-hot encoder is more complex (e.g., uses Conv/RNN),
         # the input_dim might need to be passed differently or handled within the model config.
         # For the current ProteinEncoder placeholder, it expects the flattened dim.
         logging.warning(f"Using flattened one-hot dim {model_cfg['protein_input_dim']} as protein_input_dim.")

    elif seq_encoding_type == 'raw':
         model_cfg['protein_input_dim'] = None # Handled by PLM/tokenizer within the model
         if model_cfg.get('protein_plm_name') is None or model_cfg.get('protein_plm_embed_dim') is None:
              logging.error("protein_plm_name and protein_plm_embed_dim must be specified in model_config for 'raw' protein encoding type.")
              sys.exit(1)
    else:
         logging.error(f"Unsupported seq_encoding_type: {seq_encoding_type}")
         sys.exit(1)

    # Need to ensure fe_curve_length and channels match between data and model configs
    if model_cfg['fe_curve_length'] != preprocessing_params['fe_curve_length']:
         logging.error("Model config 'fe_curve_length' must match data config 'fe_curve_length'.")
         sys.exit(1)
    if model_cfg['fe_curve_channels'] != preprocessing_params['fe_curve_channels']:
         logging.error("Model config 'fe_curve_channels' must match data config 'fe_curve_channels'.")
         sys.exit(1)


    # Create Dataset (assuming FEDataset can load the processed files)
    # Need to handle train/validation split. This example assumes all data is for training/validation.
    # You might need separate processed files for train/val or split within the script/dataset.
    # Simple approach: create one dataset and split it (less ideal for large data)
    # More robust: preprocess and save train/val data separately.

    # Assuming the processed files contain all data and FEDataset handles loading
    # You will likely need to modify FEDataset or add a split logic here.
    full_dataset = FEDataset(
        fe_curves_path=fe_curves_path,
        sequences_path=sequences_path,
        conditions_path=conditions_path,
        seq_encoding_type=seq_encoding_type, # Pass encoding type to Dataset
        fe_curve_length=preprocessing_params['fe_curve_length']
    )

    # Simple split: e.g., 80/20 train/val (seed for reproducibility)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    logging.info(f"Dataset split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")


    # Define collate_fn if needed (e.g., for raw sequence encoding)
    collate_fn = None
    if seq_encoding_type == 'raw':
         # Need a custom collate function to tokenize and pad raw sequences
         # from data_processing.collate_fn import collate_fn_raw_seq # You need to create this file
         # collate_fn = collate_fn_raw_seq
         logging.warning("Using 'raw' sequence encoding requires a custom collate_fn for DataLoader.")
         logging.warning("Default DataLoader collate will likely fail for list of strings.")
         pass # Use default collate_fn which will fail unless handled


    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() // 2 or 1 # Example num_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() // 2 or 1
    )

    # --- Model Initialization ---
    logging.info("Initializing model...")
    model = ConditionalDiffusionModel(
        fe_curve_length=model_cfg['fe_curve_length'],
        fe_curve_channels=model_cfg['fe_curve_channels'],
        protein_input_dim=model_cfg['protein_input_dim'], # Will be None for 'raw'
        protein_embed_dim=model_cfg['protein_embed_dim'],
        protein_encoding_type=model_cfg['protein_encoding_type'],
        protein_plm_name=model_cfg.get('protein_plm_name'),
        protein_plm_embed_dim=model_cfg.get('protein_plm_embed_dim'),
        protein_freeze_plm=model_cfg.get('protein_freeze_plm', True),
        condition_input_dim=model_cfg['condition_input_dim'],
        condition_embed_dim=model_cfg['condition_embed_dim'],
        time_embed_dim=model_cfg['time_embed_dim'],
        model_channels=model_cfg['model_channels'],
        num_diffusion_steps=model_cfg['num_diffusion_steps'],
        beta_schedule=model_cfg.get('beta_schedule', 'linear')
        # Pass other model config parameters as needed
    )
    logging.info("Model initialized.")

    # --- Trainer Setup ---
    logging.info("Setting up trainer...")
    device = torch.device(train_cfg.get('device', 'cpu'))

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer_config=train_cfg['optimizer'],
        scheduler_config=train_cfg['scheduler'],
        loss_config=train_cfg['loss'],
        epochs=train_cfg['epochs'],
        device=device,
        log_dir=train_cfg.get('log_dir', 'runs/'),
        checkpoint_dir=train_cfg.get('checkpoint_dir', 'checkpoints/'),
        save_interval=train_cfg.get('save_interval', 10),
        eval_interval=train_cfg.get('eval_interval', 5)
    )
    logging.info("Trainer setup complete.")

    # --- Resume Training if Checkpoint Provided ---
    start_epoch = 1
    if checkpoint_path:
        logging.info(f"Attempting to load checkpoint from {checkpoint_path}")
        try:
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            logging.info(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting training from epoch 1.")
            start_epoch = 1
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            sys.exit(1)

    # --- Start Training ---
    logging.info("Starting training process...")
    # Modify the trainer's train method to accept a starting epoch if implementing resume
    # For simplicity in this example, we'll assume train always starts from 1 or loaded epoch
    # trainer.train(start_epoch=start_epoch) # You might modify Trainer.train()
    trainer.train() # Current Trainer.train() starts from 1, load_checkpoint updates state

    logging.info("Training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Conditional Diffusion Model for F-E curve generation.")
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to the data configuration YAML file.')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--training_config', type=str, required=True,
                        help='Path to the training configuration YAML file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional path to a checkpoint file to resume training from.')
    args = parser.parse_args()

    main(args.data_config, args.model_config, args.training_config, args.checkpoint)