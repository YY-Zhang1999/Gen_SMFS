import numpy as np
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AMINO_ACID_MASSES = {
    'A': 71.0788, 'R': 156.1875, 'N': 114.1038, 'D': 115.0886, 'C': 103.1388,
    'E': 129.1155, 'Q': 128.1292, 'G': 57.0519, 'H': 137.1411, 'I': 113.1594,
    'L': 113.1594, 'K': 128.1741, 'M': 131.1926, 'F': 147.1766, 'P': 97.1167,
    'S': 87.0782, 'T': 101.1051, 'W': 186.2139, 'Y': 163.1760, 'V': 99.1326
}

# Approximate contour length per amino acid in unfolded state (in nm)
# This is a typical value used in WLC modeling
CONTOUR_LENGTH_PER_AA = 0.36 # nm

def calculate_contour_length(amino_acid_sequence: str) -> float:
    """
    Calculates the approximate contour length of a protein sequence
    in its fully extended (unfolded) state.

    Args:
        amino_acid_sequence (str): The protein sequence string.

    Returns:
        float: The approximate contour length in nanometers.
    """
    if not isinstance(amino_acid_sequence, str) or not amino_acid_sequence:
        logging.warning("Invalid or empty amino acid sequence provided for contour length calculation.")
        return 0.0
    # Basic check for valid amino acids (case-insensitive)
    valid_aa_sequence = ''.join(c for c in amino_acid_sequence.upper() if c in AMINO_ACID_MASSES)
    if len(valid_aa_sequence) != len(amino_acid_sequence):
         logging.warning(f"Sequence contains unexpected characters. Using valid AA count. Original: {amino_acid_sequence}, Valid: {valid_aa_sequence}")

    num_residues = len(valid_aa_sequence)
    return num_residues * CONTOUR_LENGTH_PER_AA

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from a specified file path using pandas.
    Assumes data is in a tabular format (CSV, Excel, etc.).

    Args:
        file_path (str): Path to the raw data file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
        Exception: For other loading errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        # Add support for other formats as needed
        else:
            raise ValueError(f"Unsupported raw data file format: {os.path.basename(file_path)}")
        logging.info(f"Successfully loaded raw data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading raw data from {file_path}: {e}")
        raise

def save_processed_data(data, file_path: str):
    """
    Saves processed data to a specified file path.
    Supports saving pandas DataFrames to CSV and numpy arrays to .npy.

    Args:
        data: The processed data (pandas DataFrame or numpy array).
        file_path (str): Path to save the processed data.

    Raises:
        ValueError: If the data type or file format is unsupported.
        Exception: For other saving errors.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    print(type(data), isinstance(data, np.ndarray))
    try:
        if isinstance(data, pd.DataFrame) and file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif isinstance(data, np.ndarray) and file_path.endswith('.npy'):
             np.save(file_path, data)
        elif isinstance(data, (list, np.ndarray)) and file_path.endswith('.pkl'):
             # Saving list/array to pickle can be convenient for arbitrary structures
             pd.to_pickle(data, file_path)
        # Add support for other formats as needed
        else:
            raise ValueError(f"Unsupported data type ({type(data)}) or file format ({os.path.basename(file_path)}) for saving processed data.")
        logging.info(f"Successfully saved processed data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving processed data to {file_path}: {e}")
        raise


# Example Usage
if __name__ == "__main__":
    print("--- Testing utils.py ---")

    # Test calculate_contour_length
    sequence = "AGARSDG"
    contour_len = calculate_contour_length(sequence)
    print(f"Contour length of '{sequence}': {contour_len:.2f} nm")

    sequence_invalid = "AGZXRSDG"
    contour_len_invalid = calculate_contour_length(sequence_invalid)
    print(f"Contour length of '{sequence_invalid}': {contour_len_invalid:.2f} nm (with warning)")

    # Test file loading and saving (requires creating dummy files)
    dummy_raw_csv_path = 'data/raw/dummy_raw_data.csv'
    dummy_processed_csv_path = 'data/processed/dummy_processed_data.csv'
    dummy_processed_npy_path = 'data/processed/dummy_processed_array.npy'
    dummy_processed_pkl_path = 'data/processed/dummy_processed_list.pkl'

    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Create dummy raw data
    dummy_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['A', 'B', 'C']
    })
    dummy_df.to_csv(dummy_raw_csv_path, index=False)

    # Load dummy raw data
    try:
        loaded_df = load_raw_data(dummy_raw_csv_path)
        print("\nLoaded raw data:")
        print(loaded_df)
    except FileNotFoundError as e:
         print(f"Skipping load test: {e}")
    except Exception as e:
        print(f"An error occurred during load test: {e}")


    # Save dummy processed data (DataFrame)
    try:
        save_processed_data(loaded_df, dummy_processed_csv_path)
    except Exception as e:
         print(f"An error occurred during save CSV test: {e}")


    # Save dummy processed data (Numpy array)
    dummy_array = np.array([[1.1, 2.2], [3.3, 4.4]])
    try:
        save_processed_data(dummy_array, dummy_processed_npy_path)
    except Exception as e:
         print(f"An error occurred during save NPY test: {e}")

    # Save dummy processed data (List to pickle)
    dummy_list = [np.random.rand(10) for _ in range(5)]
    try:
        save_processed_data(dummy_list, dummy_processed_pkl_path)
    except Exception as e:
         print(f"An error occurred during save PKL test: {e}")


    # Clean up dummy files
    # os.remove(dummy_raw_csv_path)
    # os.remove(dummy_processed_csv_path)
    # os.remove(dummy_processed_npy_path)
    # os.remove(dummy_processed_pkl_path)
    # os.rmdir('data/raw') # Only if empty
    # os.rmdir('data/processed') # Only if empty