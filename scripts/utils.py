import logging
import os
import sys
from typing import Dict, Any

import yaml

from Gen_SMFS.src.data_processing import FEDataset


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

def creat_dataset(data_cfg: Dict[str, Any], feature_first : bool = True) -> FEDataset:
    # --- Data Loading and Preparation ---
    data_paths = data_cfg.get('data_paths', {})
    preprocessing_params = data_cfg.get('data_preprocessing', {})

    processed_data_dir = data_paths.get('processed_data_dir')
    fe_curves_file = data_paths.get('processed_fe_curves_file', 'fe_curves.npy')
    conditions_file = data_paths.get('processed_conditions_file', 'conditions.npy')  # Or .csv

    fe_curves_path = os.path.join(processed_data_dir, fe_curves_file)
    conditions_path = os.path.join(processed_data_dir, conditions_file)

    # Determine condition_input_dim from data config
    condition_columns = preprocessing_params.get('condition_columns')
    if condition_columns is None:
        logging.error("condition_columns not specified in data_config.data_preprocessing.")
        sys.exit(1)

    full_dataset = FEDataset(
        fe_curves_path=fe_curves_path,
        conditions_path=conditions_path,
        fe_curve_length=preprocessing_params['fe_curve_length'],
        feature_first=feature_first
    )

    return full_dataset

