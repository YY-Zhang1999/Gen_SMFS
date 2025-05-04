# src/models/condition_encoder.py

import torch
import torch.nn as nn
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConditionEncoder(nn.Module):
    """
    Encodes experimental/simulation conditions (e.g., pulling speed, temperature)
    into a fixed-size embedding vector.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initializes the ConditionEncoder.

        Args:
            input_dim (int): The dimensionality of the input condition vector
                             (number of condition features).
            output_dim (int): The desired dimensionality of the final condition embedding vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Simple feed-forward network to encode conditions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2), # Example: Expand then project
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        logging.info(f"ConditionEncoder initialized with input_dim={input_dim}, output_dim={output_dim}.")

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConditionEncoder.

        Args:
            conditions (torch.Tensor): Input condition tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, output_dim) representing
                          the condition embeddings.
        """
        if conditions.shape[-1] != self.input_dim:
            raise ValueError(f"Condition encoder expected input dimension {self.input_dim}, but got {conditions.shape[-1]}.")

        embeddings = self.encoder(conditions)

        # Ensure the output shape is correct
        if embeddings.shape[-1] != self.output_dim:
            logging.error(f"Condition encoder output shape mismatch. Expected (batch_size, {self.output_dim}), got {embeddings.shape}")
            raise RuntimeError("Condition encoder output shape mismatch.")

        return embeddings

# Example Usage
if __name__ == "__main__":
    print("--- Testing ConditionEncoder ---")

    batch_size = 4
    input_dim = 2 # e.g., pulling speed, temperature
    output_dim = 64 # Condition embedding dimension

    # Create a dummy input tensor
    dummy_conditions = torch.randn(batch_size, input_dim)
    print(f"Input shape: {dummy_conditions.shape}")

    # Create a ConditionEncoder instance
    condition_encoder = ConditionEncoder(input_dim, output_dim)

    # Perform forward pass
    output = condition_encoder(dummy_conditions)
    print(f"Output shape: {output.shape}")

    # Check if gradients flow
    output.sum().backward()
    print("Backward pass successful.")