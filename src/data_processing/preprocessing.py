# src/data_processing/preprocessing.py

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import utility function (assuming utils.py is in the same directory)
try:
    from .utils import calculate_contour_length, load_raw_data
except ImportError:
    # If running this file directly or in a different structure,
    # you might need to adjust the import or add src to your PYTHONPATH.
    logging.warning("Could not import utils.py directly. Ensure src.data_processing is in PYTHONPATH.")
    from utils import calculate_contour_length, load_raw_data

def read_simulation_data(df_save_path, clustering_no, speed=500):
    """
    Reads simulation data files generated from molecular simulations

    Args:
        df_save_path (str): Path where data files are stored
        clustering_no (str): Clustering identifier used in filenames
        speed (int): Speed in nm/s used in simulation

    Returns:
        tuple: (Fu_data_df, xp_data_df, WLC_data_df) - DataFrames containing force,
               extension, and worm-like chain parameter data
    """
    # Construct file paths
    fu_file = f"{df_save_path}clustering_Fu_{clustering_no}_sim_speed_{speed}_data.csv"
    xp_file = f"{df_save_path}clustering_xp_{clustering_no}_sim_speed_{speed}_data.csv"
    wlc_file = f"{df_save_path}clustering_WLC_data_{clustering_no}_sim_speed_{speed}_data.csv"

    # Read data files
    try:
        Fu_data_df = pd.read_pickle(fu_file)
        print(f"Successfully loaded force data from {fu_file}")
    except FileNotFoundError:
        print(f"Force data file not found: {fu_file}")
        Fu_data_df = None

    try:
        xp_data_df = pd.read_pickle(xp_file)
        print(f"Successfully loaded extension data from {xp_file}")
    except FileNotFoundError:
        print(f"Extension data file not found: {xp_file}")
        xp_data_df = None

    try:
        WLC_data_df = pd.read_pickle(wlc_file)
        print(f"Successfully loaded WLC parameter data from {wlc_file}")
    except FileNotFoundError:
        print(f"WLC data file not found: {wlc_file}")
        WLC_data_df = None

    return Fu_data_df, xp_data_df, WLC_data_df



def standardize_fe_curve(
    extension: np.ndarray,
    force: np.ndarray,
    target_length: int,
    normalization_factor: float = 1.0, # e.g., contour length or max force
    force_unit_conversion: float = 1.0 # e.g., pN to nN
) -> np.ndarray:
    """
    Standardizes a single Force-Extension (F-E) curve by normalizing force,
    converting extension to a standardized range (0 to 1 or 0 to contour length),
    and resampling to a fixed target length.

    Args:
        extension (np.ndarray): Array of extension values for a single curve.
        force (np.ndarray): Array of force values for a single curve.
        target_length (int): The desired fixed length of the output F-E curve vector.
        normalization_factor (float): Factor to normalize the force data (e.g., max expected force).
                                      Forces will be divided by this factor.
        force_unit_conversion (float): Factor to convert force units if necessary.
                                       Forces will be multiplied by this factor.

    Returns:
        np.ndarray: A standardized F-E curve vector of shape (target_length,).
                    Contains force values corresponding to resampled extensions.
    """
    if len(extension) != len(force) or len(extension) < 2:
        logging.warning(f"Invalid F-E curve data length: {len(extension)} extension, {len(force)} force. Returning empty array.")
        return np.zeros(target_length, dtype=np.float32)

    # Apply force unit conversion
    force = force * force_unit_conversion

    # 1. Standardize Extension: Map original extension range to a [0, 1] range
    # Or you might standardize based on contour length: [0, contour_length]
    # Here we map to [0, 1] as a general approach for resampling
    min_extension = np.min(extension)
    max_extension = np.max(extension)
    if max_extension - min_extension == 0:
         logging.warning(f"Zero extension range in curve. Cannot standardize extension.")
         # Return a zero curve or handle as an error case
         return np.zeros(target_length, dtype=np.float32)

    standardized_extension = (extension - min_extension) / (max_extension - min_extension)

    # Create an interpolation function
    # kind='linear' is a common choice, 'cubic' or 'nearest' are alternatives
    try:
        interp_func = interp1d(standardized_extension, force, kind='linear', bounds_error=False, fill_value="extrapolate")
    except ValueError as e:
         logging.warning(f"Interpolation failed for curve: {e}. Data might be invalid (e.g., non-monotonic extension). Returning zero array.")
         return np.zeros(target_length, dtype=np.float32)


    # 2. Resample to target length
    # Create the new standardized extension points
    resampled_standardized_extension = np.linspace(0, 1, target_length)

    # Get the force values at the new extension points
    resampled_force = interp_func(resampled_standardized_extension)

    # 3. Normalize Force
    if normalization_factor <= 0:
        logging.warning("Normalization factor is zero or negative. Skipping force normalization.")
    else:
        resampled_force = resampled_force / normalization_factor

    # Ensure the output is float32 as expected by PyTorch
    return resampled_force.astype(np.float32)

def standardize_fe_curve_length(
    extension: np.ndarray,
    force: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Standardizes a single Force-Extension (F-E) curve by normalizing force,
    converting extension to a standardized range (0 to 1 or 0 to contour length),
    and resampling to a fixed target length.

    Args:
        extension (np.ndarray): Array of extension values for a single curve.
        force (np.ndarray): Array of force values for a single curve.
        target_length (int): The desired fixed length of the output F-E curve vector.
        normalization_factor (float): Factor to normalize the force data (e.g., max expected force).
                                      Forces will be divided by this factor.
        force_unit_conversion (float): Factor to convert force units if necessary.
                                       Forces will be multiplied by this factor.

    Returns:
        np.ndarray: A standardized F-E curve vector of shape (target_length,).
                    Contains force values corresponding to resampled extensions.
    """
    if len(extension) != len(force) or len(extension) < 2:
        logging.warning(f"Invalid F-E curve data length: {len(extension)} extension, {len(force)} force. Returning empty array.")
        return np.zeros(target_length, dtype=np.float32)
    elif target_length < len(force):
        logging.warning(
            f"Invalid F-E curve data length: {len(extension)} extension, {target_length} target_length. Returning empty array.")
        return np.zeros(target_length, dtype=np.float32)

    diff_length = target_length - len(extension)

    new_arr = np.append(force, np.zeros(diff_length, dtype=np.float32))

    # Ensure the output is float32 as expected by PyTorch
    return new_arr.astype(np.float32)


def preprocess_fe_curves(
    raw_data_list: List[Dict[str, np.ndarray]],
    target_curve_length: int,
    normalization_strategy: str = 'max_force', # 'max_force', 'contour_length', 'none'
    global_max_force: float = None, # Required if normalization_strategy is 'max_force' and not per curve
    protein_sequences: List[str] = None # Required if normalization_strategy is 'contour_length'
) -> List[np.ndarray]:
    """
    Applies standardization and resampling to a list of raw F-E curves.

    Args:
        raw_data_list (List[Dict[str, np.ndarray]]): A list where each element is a dictionary
                                                    representing a raw F-E curve.
                                                    Expected keys: 'extension', 'force'.
                                                    Values are numpy arrays.
        target_curve_length (int): The desired fixed length for all processed curves.
        normalization_strategy (str): Strategy for force normalization:
                                      'max_force': Normalize by a predefined global max force.
                                      'contour_length': Normalize force by estimated contour length (less common, usually for extension).
                                      'none': No force normalization applied here.
        global_max_force (float): The global maximum force to use for normalization
                                  if normalization_strategy is 'max_force'.
        protein_sequences (List[str]): List of protein sequences, corresponding to
                                       raw_data_list, if normalization_strategy is 'contour_length'.

    Returns:
        List[np.ndarray]: A list of standardized and resampled F-E curve vectors.
    """
    processed_curves = []
    logging.info(f"Starting F-E curve preprocessing for {len(raw_data_list)} curves...")

    if normalization_strategy == 'max_force' and global_max_force is None:
         logging.error("Normalization strategy 'max_force' requires 'global_max_force' to be provided.")
         raise ValueError("global_max_force is required for 'max_force' normalization.")

    if normalization_strategy == 'contour_length' and protein_sequences is None:
         logging.error("Normalization strategy 'contour_length' requires 'protein_sequences' to be provided.")
         raise ValueError("protein_sequences is required for 'contour_length' normalization.")
    if normalization_strategy == 'contour_length' and len(protein_sequences) != len(raw_data_list):
         logging.error("Number of protein sequences does not match number of curves.")
         raise ValueError("Mismatch between number of curves and sequences.")


    for i, raw_curve in enumerate(raw_data_list):
        try:
            extension = raw_curve['extension']
            force = raw_curve['force']

            # Determine normalization factor based on strategy
            norm_factor = 1.0 # Default to no normalization
            if normalization_strategy == 'max_force':
                norm_factor = global_max_force
            elif normalization_strategy == 'contour_length':
                 # Normalizing force by contour length is less standard, typically extension is normalized
                 # If this strategy is truly intended for force, use contour length here.
                 # If it was meant for extension normalization, the standardize_fe_curve function needs modification.
                 # Assuming for now it's intended as a factor for force normalization.
                 contour_len = calculate_contour_length(protein_sequences[i])
                 if contour_len > 0:
                     norm_factor = contour_len
                 else:
                     logging.warning(f"Contour length is zero for curve {i}. Skipping normalization for this curve.")
                     norm_factor = 1.0 # Avoid division by zero

            # You might also consider normalizing force by the peak force *of that curve*
            # if capturing relative changes is more important than absolute force values.
            # This would require calculating max force per curve.
            processed_curve = standardize_fe_curve_length(
                extension,
                force,
                target_length=target_curve_length,
                # Add force_unit_conversion if needed
            )
            processed_curves.append(processed_curve)

        except Exception as e:
            logging.error(f"Error processing curve {i}: {e}. Skipping this curve.")
            # Depending on requirements, you might want to raise the exception
            # or store a placeholder for the failed curve.
            continue # Skip to the next curve


    logging.info(f"Finished F-E curve preprocessing. Processed {len(processed_curves)} curves.")
    return processed_curves


def encode_protein_sequences(
    sequences: List[str],
    encoding_type: str,
    # Add parameters for specific encoders if needed
    # e.g., plm_model_name: str = 'esm2_t6_8m_UR50D', plm_layer: int = 6
) -> List[np.ndarray] | List[str]:
    """
    Encodes a list of protein sequences based on the specified encoding type.

    Args:
        sequences (List[str]): List of protein sequence strings.
        encoding_type (str): Type of encoding to use ('onehot', 'pretrained_embeddings', 'raw').

    Returns:
        List[np.ndarray] or List[str]: A list of encoded sequence representations (numpy arrays)
                                       or the original list of strings if 'raw'.
    """
    logging.info(f"Encoding {len(sequences)} sequences using '{encoding_type}'...")

    if encoding_type == 'raw':
        logging.info("Returning raw sequences.")
        return sequences # Return original strings, dataset/model handles encoding

    elif encoding_type == 'onehot':
        # Simple one-hot encoding (example implementation)
        # This assumes a fixed alphabet and sequence length for padding or truncation
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY' # 20 standard amino acids
        aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
        # Determine max length for padding or use a fixed length
        # For simplicity, let's return variable length one-hot arrays for now.
        # Padding should ideally happen in a collate_fn or a dedicated padding step.
        encoded_list = []
        for seq in sequences:
            onehot_seq = np.zeros((len(seq), len(amino_acids)), dtype=np.float32)
            for i, aa in enumerate(seq.upper()):
                if aa in aa_to_int:
                    onehot_seq[i, aa_to_int[aa]] = 1.0
                else:
                    logging.warning(f"Unknown amino acid '{aa}' in sequence '{seq}'. Skipping.")
            encoded_list.append(onehot_seq)
        logging.info("One-hot encoding complete.")
        return encoded_list

    elif encoding_type == 'pretrained_embeddings':
        # This is a placeholder for using a PLM.
        # Using PLMs typically involves:
        # 1. Loading the PLM model and possibly tokenizer (requires e.g., transformers library)
        # 2. Tokenizing sequences (handling variable lengths, padding)
        # 3. Passing tokens through the PLM to get embeddings (requires torch)
        # 4. Selecting which layer's embeddings to use (often the last hidden state or average of layers)
        # 5. Pooling/reducing embeddings per sequence (e.g., mean pooling, using [CLS] token if available)

        logging.warning("Pretrained embedding encoding is a placeholder. Actual PLM integration is required.")
        # Example: return dummy embeddings of a fixed size
        embedding_dim = 1024 # Example embedding dimension from a PLM
        return [np.random.rand(embedding_dim).astype(np.float32) for _ in sequences]

    else:
        raise ValueError(f"Unsupported sequence encoding type: {encoding_type}")


def encode_conditions(
    conditions_df: pd.DataFrame,
    condition_columns: List[str],
    # Add parameters for specific encoding/scaling if needed
    # e.g., scaling_method: str = 'standard_scaler'
) -> np.ndarray:
    """
    Encodes/prepares experimental conditions from a DataFrame.

    Args:
        conditions_df (pd.DataFrame): DataFrame containing experimental conditions.
        condition_columns (List[str]): List of column names in conditions_df to use.

    Returns:
        np.ndarray: A numpy array where each row is the encoded conditions for a sample.
                    Shape: (num_samples, num_condition_features).
    """
    logging.info(f"Encoding conditions from columns: {condition_columns}")

    # Select the relevant columns
    if not all(col in conditions_df.columns for col in condition_columns):
        missing = [col for col in condition_columns if col not in conditions_df.columns]
        logging.error(f"Missing condition columns in DataFrame: {missing}")
        raise ValueError(f"Missing condition columns: {missing}")

    conditions_data = conditions_df[condition_columns]

    # --- Add scaling or further encoding here if needed ---
    # Example: using StandardScaler from scikit-learn (requires installation)
    # from sklearn.preprocessing import StandardScaler
    # ------------------------------------------------------

    logging.info(f"Conditions encoded with shape: {conditions_data.shape} and type {type(conditions_data)}")
    conditions_data = np.array(conditions_data.values)
    return conditions_data



