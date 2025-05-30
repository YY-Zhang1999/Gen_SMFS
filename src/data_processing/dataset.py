import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FEDataset(Dataset):
    """
    Custom Dataset class for loading and providing processed Force-Extension (F-E)
    curves, protein sequences, and experimental conditions for model training.

    Assumes data has been preprocessed and stored in specified file formats.
    """
    def __init__(self,
                 fe_curves_path: str,
                 device: torch.device,
                 sequences_path: str = None,
                 conditions_path: str = None,
                 seq_encoding_type: str = 'onehot',
                 fe_curve_length: int = 1000,
                 ):
        """
        Initializes the FEDataset.

        Args:
            fe_curves_path (str): Path to the processed F-E curve data file (e.g., .pkl, .csv).
                                  Assumed to be a structure where curves can be indexed.
            sequences_path (str): Path to the processed protein sequences data file (e.g., .csv).
                                  Assumed to contain sequence strings or pre-encoded features.
            conditions_path (str): Path to the processed experimental conditions data file (e.g., .csv).
                                   Assumed to contain numerical conditions like pulling speed.
            seq_encoding_type (str): Type of sequence encoding used in the sequences_path file.
                                     Currently supports 'onehot' or 'pretrained_embeddings'.
                                     (Future: add support for loading raw sequences for PLM encoding)
            fe_curve_length (int): The fixed length of the F-E curve vectors after preprocessing.
        """
        self.device = device

        if not os.path.exists(fe_curves_path):
            raise FileNotFoundError("Force-extension data files do not exist.")
        if sequences_path and not os.path.exists(sequences_path):
            raise FileNotFoundError("Sequences data files do not exist.")
        if conditions_path and not os.path.exists(conditions_path):
            raise FileNotFoundError("Conditions data files do not exist.")

        logging.info(f"Loading data from: {fe_curves_path}, {sequences_path}, {conditions_path}")

        # Load F-E curves
        # Assuming fe_curves_path points to a file that loads into a list/array
        # where each element is a preprocessed F-E curve vector of fe_curve_length.
        # Example using numpy or pandas:
        try:
            if fe_curves_path.endswith('.pkl'):
                 self.fe_curves = pd.read_pickle(fe_curves_path)
            elif fe_curves_path.endswith('.csv'):
                # Assuming CSV is loaded and potentially converted to numpy array/list
                 self.fe_curves = pd.read_csv(fe_curves_path).values # Adjust if necessary
            elif fe_curves_path.endswith('.npy'):
                 self.fe_curves = np.load(fe_curves_path)
            else:
                 raise ValueError(f"Unsupported F-E curves file format: {fe_curves_path}")

            if not isinstance(self.fe_curves, (list, np.ndarray)):
                 raise TypeError("F-E curves data not loaded into list or numpy array.")
            if isinstance(self.fe_curves, np.ndarray) and self.fe_curves.shape[1] != fe_curve_length:
                 logging.warning(f"Loaded F-E curves have length {self.fe_curves.shape[1]}, expected {fe_curve_length}. Please check preprocessing.")
            elif isinstance(self.fe_curves, list) and len(self.fe_curves) > 0 and len(self.fe_curves[0]) != fe_curve_length:
                 logging.warning(f"Loaded F-E curves have length {len(self.fe_curves[0])}, expected {fe_curve_length}. Please check preprocessing.")

            # Convert numpy array to tensor
            self.fe_curves = torch.as_tensor(self.fe_curves, dtype=torch.float32, device=device)


        except Exception as e:
            logging.error(f"Error loading F-E curves from {fe_curves_path}: {e}")
            raise

        # Load sequences
        # Assuming sequences_path points to a CSV file where each row corresponds
        # to a data sample and contains either the raw sequence string or
        # pre-extracted embeddings.
        self.has_sequences = True
        if not sequences_path:
            self.has_sequences = False
            self.raw_sequences = None
            self.encoded_sequences = None
        else:
            try:
                self.sequences_df = pd.read_csv(sequences_path)
                self.seq_encoding_type = seq_encoding_type.lower()

                if self.seq_encoding_type not in ['onehot', 'pretrained_embeddings', 'raw']:
                    logging.warning(
                        f"Unsupported sequence encoding type '{seq_encoding_type}'. Expected 'onehot', 'pretrained_embeddings', or 'raw'.")
                    # Still load data, but user should handle encoding appropriately in model
                    # If 'raw', the model's protein encoder should handle tokenization/embedding.
                    # Assuming a column named 'sequence' for raw sequences.
                    if self.seq_encoding_type == 'raw' and 'sequence' not in self.sequences_df.columns:
                        raise ValueError(
                            f"Sequence encoding type 'raw' specified, but no 'sequence' column found in {sequences_path}.")

                # If sequences are already encoded (e.g., one-hot or PLM embeddings)
                if self.seq_encoding_type in ['onehot', 'pretrained_embeddings']:
                    # Assuming feature columns are numeric and represent the encoding
                    # Need a way to identify sequence feature columns. This is a placeholder.
                    # In a real scenario, you'd need to know which columns contain the encoding.
                    # For now, let's assume all columns except maybe 'protein_id' or similar are features.
                    # A more robust approach would be needed based on actual data format.
                    feature_columns = [col for col in self.sequences_df.columns if
                                       self.sequences_df[col].dtype in [np.number, np.bool_]]
                    if not feature_columns:
                        raise ValueError(f"No numeric/boolean columns found in {sequences_path} for encoded sequences.")
                    self.encoded_sequences = self.sequences_df[feature_columns].values
                    logging.info(
                        f"Loaded {self.seq_encoding_type} sequences with shape: {self.encoded_sequences.shape}")

                elif self.seq_encoding_type == 'raw':
                    self.raw_sequences = self.sequences_df['sequence'].tolist()
                    logging.info(
                        f"Loaded raw sequences ({len(self.raw_sequences)}). Model's protein encoder should handle embedding.")

            except Exception as e:
                logging.error(f"Error loading sequences from {sequences_path}: {e}")
                raise

        # Load conditions
        # Assuming conditions_path points to a CSV file where each row corresponds
        # to a data sample and contains numerical experimental conditions.
        self.has_conditions = True
        if conditions_path:
            try:
                if conditions_path.endswith('.csv'):
                    self.conditions_df = pd.read_csv(conditions_path)
                    # Assuming all columns in the conditions file are numerical conditions
                    self.conditions = np.array(self.conditions_df.values)
                else:
                    self.conditions = np.load(conditions_path)
                logging.info(f"Loaded conditions with shape: {self.conditions.shape}")

                self.conditions = torch.as_tensor(self.conditions, dtype=torch.float32, device=device)

            except Exception as e:
                logging.error(f"Error loading conditions from {conditions_path}: {e}")
                raise
        else:
            self.has_conditions = False
            self.conditions = None

        # Ensure all data sources have the same number of samples
        if self.has_sequences:
            if not (len(self.fe_curves) == len(self.sequences_df) == len(self.conditions)):
                raise ValueError(f"Mismatch in number of samples across data files: "
                                 f"F-E curves ({len(self.fe_curves)}), "
                                 f"Sequences ({len(self.sequences_df)}), "
                                 f"Conditions ({len(self.conditions)})")
        else:
            if not (len(self.fe_curves) == len(self.conditions)):
                raise ValueError(f"Mismatch in number of samples across data files: "
                                 f"F-E curves ({len(self.fe_curves)}), "
                                 f"Conditions ({len(self.conditions)})")

        self.num_samples = len(self.fe_curves)
        self.fe_curve_length = fe_curve_length  # Store the expected length

        logging.info(f"Successfully loaded {self.num_samples} data samples.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves a single data sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the data for the sample:
                  'fe_curve': Tensor of the F-E curve.
                  'sequence_features': Tensor of sequence encoding (or raw sequence if encoding_type is 'raw').
                  'conditions': Tensor of experimental/simulation conditions.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Dataset index out of range.")

        fe_curve = self.fe_curves[idx]
        if isinstance(fe_curve, list): # Handle case where loaded as list of lists
            fe_curve = np.array(fe_curve)

        # Ensure the F-E curve is a numpy array of the expected shape (fe_curve_length,)
        if fe_curve.shape[0] != self.fe_curve_length:
             # Attempt to reshape if necessary, but this indicates a preprocessing issue
             logging.warning(f"F-E curve at index {idx} has shape {fe_curve.shape}, "
                             f"expected ({self.fe_curve_length},). Attempting reshape.")
             try:
                 fe_curve = fe_curve.reshape((self.fe_curve_length, 1))
             except ValueError:
                 raise ValueError(f"Cannot reshape F-E curve at index {idx} with shape {fe_curve.shape} "
                                  f"to ({self.fe_curve_length},). Check preprocessing.")

        if self.has_sequences:
            if self.seq_encoding_type == 'raw':
                sequence_data = self.raw_sequences[idx]
                # Return raw sequence string. The model's protein encoder will handle tokenization/embedding.
                # Note: This will require a custom collate_fn in the DataLoader
                # or handling variable-length strings/token lists in the model's input layer.
                # A common approach is to pad sequences in a collate_fn.
                # For simplicity here, we return the raw string.
                sequence_tensor = sequence_data  # Returning string directly
            else:  # 'onehot' or 'pretrained_embeddings'
                sequence_features = self.encoded_sequences[idx]
                sequence_tensor = torch.as_tensor(sequence_features, dtype=torch.float32)
        else:
            sequence_tensor = torch.ones(1)

        if self.has_conditions:
            conditions_tensor = self.conditions[idx]
        else:
            conditions_tensor = torch.ones(1, device=self.device)


        sample = {
            'fe_curve': fe_curve,
            'sequence_data': sequence_tensor, # Renamed to be more general for raw strings
            'conditions': conditions_tensor
        }

        return sample

# Example Usage (assuming dummy data files exist for demonstration)
if __name__ == "__main__":
    # Create dummy data files for demonstration
    dummy_fe_curves = np.random.rand(100, 500).astype(np.float32) * 100 # 100 curves of length 500, max force ~100
    dummy_sequences_onehot = np.random.randint(0, 2, size=(100, 20, 20)) # 100 sequences, 20 residues, 20 amino acids one-hot
    dummy_sequences_raw = [f"AAAAA{i}" for i in range(100)] # Dummy raw sequences
    dummy_sequences_raw_df = pd.DataFrame({'sequence': dummy_sequences_raw})

    dummy_conditions = np.random.rand(100, 1).astype(np.float32) * 1000 # 100 samples, 1 condition (pulling speed)
    dummy_conditions_df = pd.DataFrame({'pulling_speed': dummy_conditions[:, 0]})


    dummy_fe_curves_path = 'data/processed/dummy_fe_curves.npy'
    # dummy_sequences_onehot_path = 'data/processed/dummy_sequences_onehot.npy' # Example for numpy array seq features
    dummy_sequences_raw_path = 'data/processed/dummy_sequences_raw.csv' # Example for raw sequences in CSV
    dummy_conditions_path = 'data/processed/dummy_conditions.csv'

    # Ensure data directory exists
    os.makedirs('data/processed', exist_ok=True)

    np.save(dummy_fe_curves_path, dummy_fe_curves)
    # np.save(dummy_sequences_onehot_path, dummy_sequences_onehot) # Save dummy encoded sequences
    dummy_sequences_raw_df.to_csv(dummy_sequences_raw_path, index=False)
    dummy_conditions_df.to_csv(dummy_conditions_path, index=False)


    print("\n--- Testing FEDataset with raw sequences ---")
    try:
        dataset_raw_seq = FEDataset(
            fe_curves_path=dummy_fe_curves_path,
            sequences_path=dummy_sequences_raw_path,
            conditions_path=dummy_conditions_path,
            seq_encoding_type='raw',
            fe_curve_length=500
        )

        print(f"Dataset size: {len(dataset_raw_seq)}")
        sample = dataset_raw_seq[0]
        print("Sample 0:")
        print(f"  FE Curve shape: {sample['fe_curve'].shape}")
        print(f"  Sequence data type: {type(sample['sequence_data'])}") # Should be string
        print(f"  Sequence data: {sample['sequence_data']}")
        print(f"  Conditions shape: {sample['conditions'].shape}")

        # Example with DataLoader (requires a collate_fn for raw strings)
        from torch.utils.data import DataLoader

        def collate_fn_raw_seq(batch):
            # A simple collate function for raw sequences (padding or just returning list)
            fe_curves = torch.stack([item['fe_curve'] for item in batch])
            raw_sequences = [item['sequence_data'] for item in batch] # Keep as list of strings
            conditions = torch.stack([item['conditions'] for item in batch])
            # In a real scenario, you'd tokenize and pad raw_sequences here
            return {'fe_curve': fe_curves, 'sequence_data': raw_sequences, 'conditions': conditions}


        dataloader_raw_seq = DataLoader(dataset_raw_seq, batch_size=16, shuffle=True, collate_fn=collate_fn_raw_seq)

        first_batch = next(iter(dataloader_raw_seq))
        print("\nFirst batch from DataLoader (raw sequences):")
        print(f"  FE Curve batch shape: {first_batch['fe_curve'].shape}")
        print(f"  Sequence data batch type: {type(first_batch['sequence_data'])}") # Should be list
        print(f"  Sequence data batch (first item): {first_batch['sequence_data'][0]}")
        print(f"  Conditions batch shape: {first_batch['conditions'].shape}")

    except FileNotFoundError as e:
         print(f"Skipping raw sequence test: {e}")
    except Exception as e:
        print(f"An error occurred during raw sequence test: {e}")


    # Clean up dummy files
    # os.remove(dummy_fe_curves_path)
    # os.remove(dummy_sequences_raw_path)
    # os.remove(dummy_conditions_path)
    # os.rmdir('data/processed') # Only if empty and you want to remove the dir