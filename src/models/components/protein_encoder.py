# src/models/protein_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProteinEncoder(nn.Module):
    """
    Encodes protein sequences into a fixed-size embedding vector.
    Can handle pre-encoded features or serve as a placeholder for integrating a PLM.
    """
    def __init__(self,
                 input_dim: Union[int, Tuple[int, int], None], # Input dimension(s) from dataset
                 output_dim: int,
                 encoding_type: str = 'pretrained_embeddings', # 'onehot', 'pretrained_embeddings', 'raw'
                 # Add parameters for PLM if encoding_type is 'raw'
                 plm_model_name: str = None,
                 plm_embedding_dim: int = None,
                 freeze_plm: bool = True):
        """
        Initializes the ProteinEncoder.

        Args:
            input_dim (Union[int, Tuple[int, int], None]): The dimensionality of the input sequence data
                                                         from the dataset.
                                                         - If 'onehot': Tuple (seq_len, alphabet_size).
                                                         - If 'pretrained_embeddings': int (embedding_dim).
                                                         - If 'raw': None (PLM handles dimension).
            output_dim (int): The desired dimensionality of the final protein embedding vector.
            encoding_type (str): Type of encoding expected from the dataset ('onehot', 'pretrained_embeddings', 'raw').
                                 This dictates how the input is handled.
            plm_model_name (str, optional): Name of the pretrained protein language model
                                            if encoding_type is 'raw'. Defaults to None.
            plm_embedding_dim (int, optional): Dimensionality of the PLM output embeddings per residue.
                                               Required if encoding_type is 'raw'. Defaults to None.
            freeze_plm (bool): Whether to freeze the weights of the PLM if integrated. Defaults to True.
        """
        super().__init__()
        self.encoding_type = encoding_type
        self.output_dim = output_dim

        if encoding_type == 'pretrained_embeddings':
            if not isinstance(input_dim, int):
                 raise ValueError("For 'pretrained_embeddings', input_dim must be an integer representing the embedding dimension.")
            self.input_dim = input_dim
            # Simple linear layer to project pre-computed embeddings to output_dim
            self.encoder = nn.Linear(self.input_dim, output_dim)
            logging.info(f"ProteinEncoder initialized for '{encoding_type}' with input_dim={self.input_dim}, output_dim={output_dim}.")

        elif encoding_type == 'onehot':
            if not isinstance(input_dim, tuple) or len(input_dim) != 2:
                 raise ValueError("For 'onehot', input_dim must be a tuple (seq_len, alphabet_size).")
            self.input_dim = input_dim
            # Simple encoder for one-hot: e.g., Flatten and Linear or use a CNN/RNN
            # For simplicity, flatten and use a linear layer. Might need more complex layers
            # depending on how padding is handled and desired complexity.
            flat_input_dim = input_dim[0] * input_dim[1]
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_input_dim, output_dim)
            )
            logging.info(f"ProteinEncoder initialized for '{encoding_type}' with input_dim={self.input_dim}, flat_input_dim={flat_input_dim}, output_dim={output_dim}.")


        elif encoding_type == 'raw':
            # Placeholder for integrating a PLM (e.g., ESM-2)
            # This would typically involve using the `transformers` library.
            # The PLM would take raw sequence tokens and produce residue embeddings.
            # A pooling step (e.g., mean or using CLS token) would reduce
            # residue embeddings to a fixed-size sequence embedding.

            if plm_model_name is None or plm_embedding_dim is None:
                raise ValueError("For 'raw' encoding, plm_model_name and plm_embedding_dim must be provided.")

            self.plm_model_name = plm_model_name
            self.plm_embedding_dim = plm_embedding_dim
            self.freeze_plm = freeze_plm

            logging.warning(f"ProteinEncoder initialized for 'raw' encoding. PLM integration ({plm_model_name}) is a placeholder.")
            logging.warning("Requires implementation using e.g., huggingface/transformers library.")

            # Placeholder: In reality, load and configure the PLM here
            # from transformers import AutoModel, AutoTokenizer
            # self.plm = AutoModel.from_pretrained(plm_model_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(plm_model_name)
            # if freeze_plm:
            #     for param in self.plm.parameters():
            #         param.requires_grad = False

            # Placeholder for projection layer after pooling PLM embeddings
            # Assuming mean pooling results in a vector of shape (plm_embedding_dim,)
            self.projection_layer = nn.Linear(self.plm_embedding_dim, output_dim)

        else:
            raise ValueError(f"Unsupported protein encoding type: {encoding_type}. Choose from 'onehot', 'pretrained_embeddings', 'raw'.")

    def forward(self, sequence_data: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        """
        Forward pass for the ProteinEncoder.

        Args:
            sequence_data (Union[torch.Tensor, List[str]]): Input protein data.
                                                            - If 'onehot' or 'pretrained_embeddings': Tensor
                                                              of shape (batch_size, ...) as defined by input_dim.
                                                            - If 'raw': List of raw sequence strings (batch_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, output_dim) representing
                          the protein sequence embeddings.
        """
        if self.encoding_type in ['onehot', 'pretrained_embeddings']:
            if not isinstance(sequence_data, torch.Tensor):
                 raise TypeError(f"Expected torch.Tensor input for encoding type '{self.encoding_type}', but got {type(sequence_data)}")
            # Pass pre-encoded tensor through the defined encoder layers
            embeddings = self.encoder(sequence_data)

        elif self.encoding_type == 'raw':
            if not isinstance(sequence_data, list):
                 raise TypeError(f"Expected list of strings input for encoding type '{self.encoding_type}', but got {type(sequence_data)}")

            # --- PLM Integration Placeholder ---
            logging.warning("PLM forward pass is a placeholder.")
            batch_size = len(sequence_data)
            # In reality, you would:
            # 1. Tokenize and pad `sequence_data` using self.tokenizer
            # 2. Pass tokens through self.plm
            # 3. Get residue embeddings (e.g., last_hidden_state)
            # 4. Pool residue embeddings to get sequence embedding (e.g., mean over sequence length)
            # Example placeholder: create dummy embeddings
            plm_output_embedding = torch.randn(batch_size, self.plm_embedding_dim) # Shape after pooling

            # Project the PLM pooled embeddings to the desired output dimension
            embeddings = self.projection_layer(plm_output_embedding)
            # ------------------------------------

        else:
             # Should not happen if __init__ is correct
             raise NotImplementedError(f"Forward pass not implemented for encoding type: {self.encoding_type}")


        # Ensure the output shape is correct
        if embeddings.shape[1] != self.output_dim:
            logging.error(f"Protein encoder output shape mismatch. Expected (batch_size, {self.output_dim}), got {embeddings.shape}")
            # Handle error or attempt reshape/projection
            raise RuntimeError("Protein encoder output shape mismatch.")


        return embeddings

# Example Usage
if __name__ == "__main__":
    print("--- Testing ProteinEncoder ---")

    batch_size = 4
    output_dim = 128

    # Test with pretrained_embeddings (assuming pre-computed embeddings)
    print("\n--- Testing with 'pretrained_embeddings' ---")
    plm_embedding_dim_input = 1024
    dummy_pretrained_embeddings = torch.randn(batch_size, plm_embedding_dim_input)

    try:
        encoder_plm_precomputed = ProteinEncoder(
            input_dim=plm_embedding_dim_input,
            output_dim=output_dim,
            encoding_type='pretrained_embeddings'
        )
        output_plm_precomputed = encoder_plm_precomputed(dummy_pretrained_embeddings)
        print(f"Input shape: {dummy_pretrained_embeddings.shape}")
        print(f"Output shape ('pretrained_embeddings'): {output_plm_precomputed.shape}")
        output_plm_precomputed.sum().backward()
        print("Backward pass successful.")

    except Exception as e:
        print(f"An error occurred during 'pretrained_embeddings' test: {e}")


    # Test with onehot (assuming fixed length one-hot input)
    print("\n--- Testing with 'onehot' ---")
    seq_len_input = 50
    alphabet_size_input = 20
    dummy_onehot_input = torch.randn(batch_size, seq_len_input, alphabet_size_input) # Simplified: usually binary values

    try:
        encoder_onehot = ProteinEncoder(
            input_dim=(seq_len_input, alphabet_size_input),
            output_dim=output_dim,
            encoding_type='onehot'
        )
        output_onehot = encoder_onehot(dummy_onehot_input)
        print(f"Input shape: {dummy_onehot_input.shape}")
        print(f"Output shape ('onehot'): {output_onehot.shape}")
        output_onehot.sum().backward()
        print("Backward pass successful.")

    except Exception as e:
        print(f"An error occurred during 'onehot' test: {e}")


    # Test with raw sequences (placeholder test)
    print("\n--- Testing with 'raw' (PLM placeholder) ---")
    dummy_raw_sequences = ["SEQUENCEA", "SEQB", "LONGERSEQUENCE"] * (batch_size // 3 + 1)
    dummy_raw_sequences = dummy_raw_sequences[:batch_size] # Ensure correct batch size

    try:
        encoder_raw_plm = ProteinEncoder(
            input_dim=None, # Input dim is handled by PLM/tokenizer
            output_dim=output_dim,
            encoding_type='raw',
            plm_model_name='dummy_plm', # Placeholder name
            plm_embedding_dim=768 # Placeholder PLM dimension
        )
        output_raw_plm = encoder_raw_plm(dummy_raw_sequences)
        print(f"Input (list of strings): {dummy_raw_sequences}")
        print(f"Output shape ('raw' placeholder): {output_raw_plm.shape}")
        output_raw_plm.sum().backward()
        print("Backward pass successful.")

    except Exception as e:
        print(f"An error occurred during 'raw' test: {e}")