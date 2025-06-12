# scripts/train.py
import os
from typing import Dict, Any
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models')))

import yaml
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from Gen_SMFS.src.data_processing.dataset import FEDataset
from Gen_SMFS.src.training.trainer import VAETrainer
from Gen_SMFS.src.models.vae import create_conditiaonl_vae_model

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
    conditions_file = data_paths.get('processed_conditions_file', 'conditions.npy') # Or .csv

    fe_curves_path = os.path.join(processed_data_dir, fe_curves_file)
    conditions_path = os.path.join(processed_data_dir, conditions_file)


    # Determine condition_input_dim from data config
    condition_columns = preprocessing_params.get('condition_columns')
    if condition_columns is None:
         logging.error("condition_columns not specified in data_config.data_preprocessing.")
         sys.exit(1)
    model_cfg['condition_input_dim'] = len(condition_columns)


    # Create Dataset (assuming FEDataset can load the processed files)
    # Need to handle train/validation split. This example assumes all data is for training/validation.
    # You might need separate processed files for train/val or split within the script/dataset.
    # Simple approach: create one dataset and split it (less ideal for large data)
    # More robust: preprocess and save train/val data separately.

    # Assuming the processed files contain all data and FEDataset handles loading
    # You will likely need to modify FEDataset or add a split logic here.
    device = torch.device(train_cfg.get('device', 'cpu'))

    full_dataset = FEDataset(
        fe_curves_path=fe_curves_path,
        conditions_path=conditions_path,
        fe_curve_length=preprocessing_params['fe_curve_length']
    )

    # Simple split: e.g., 80/20 train/val (seed for reproducibility)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    logging.info(f"Dataset split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")


    # Define collate_fn if needed (e.g., for raw sequence encoding)
    collate_fn = None

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0 # Example num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --- Model Initialization ---
    logging.info("Initializing model...")
    model = create_conditiaonl_vae_model(
        input_feature_dim=1,
        sequence_len=preprocessing_params['fe_curve_length'],
        latent_feature_dim=4,
        conditional_dim=1,
        scale_factor=4,
        backbone_type='transformer',
        use_crossattention=True,
    )


    logging.info("Model initialized.")

    # --- Trainer Setup ---
    logging.info("Setting up trainer...")

    trainer = VAETrainer(
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
    trainer.train()  # Current Trainer.train() starts from 1, load_checkpoint updates state

    logging.info("Training script finished.")




if __name__ == "__main__":
    main('..\\config\\simulation_data_config.yaml',
         '..\\config\\model_config.yaml',
         '..\\config\\training_config.yaml',
         '..\\results')
