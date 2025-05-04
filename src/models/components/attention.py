# src/models/components/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module as used in Transformer models.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initializes the MultiHeadSelfAttention layer.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads. Must divide embed_dim.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for queries, keys, and values
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Self-Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Optional mask tensor to prevent attention
                                           to certain positions. Shape (batch_size, 1, 1, seq_len).
                                           Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.size()

        # Project input to queries, keys, and values, and split into heads
        # Shape becomes (batch_size, num_heads, seq_len, head_dim)
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (queries dot keys)
        # (batch_size, num_heads, seq_len, head_dim) x (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply mask if provided
        if mask is not None:
            # Mask has shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention probabilities to values
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attended_values = torch.matmul(attention_probs, values)

        # Concatenate heads and apply final linear layer
        # Shape becomes (batch_size, seq_len, embed_dim)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.fc_out(attended_values)

        return output

# Example Usage
if __name__ == "__main__":
    print("--- Testing MultiHeadSelfAttention ---")
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {dummy_input.shape}")

    # Create a MultiHeadSelfAttention instance
    attention_layer = MultiHeadSelfAttention(embed_dim, num_heads)

    # Perform forward pass
    output = attention_layer(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test with a mask (e.g., simulating padding)
    # Create a dummy mask: Batch 0 attends everywhere, Batch 1 attends only to first 5 tokens
    dummy_mask = torch.ones(batch_size, seq_len) # (batch_size, seq_len)
    dummy_mask[1, 5:] = 0
    # Expand mask to attention scores shape (batch_size, 1, 1, seq_len) for broadcasting
    dummy_mask = dummy_mask.unsqueeze(1).unsqueeze(2)
    print(f"\nInput mask shape: {dummy_mask.shape}")

    output_masked = attention_layer(dummy_input, mask=dummy_mask)
    print(f"Output shape with mask: {output_masked.shape}")

    # Check if gradients flow
    output.sum().backward()
    print("\nBackward pass successful.")