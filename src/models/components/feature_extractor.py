import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any, List
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding using sine and cosine functions."""

    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class CNNFeatureExtractor(nn.Module):
    """Convolutional feature extractor for time series data."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channels: List[int] = [32, 64, 128],
                 kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        # Create CNN layers with increasing channel sizes
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim

        for out_channels in channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Final projection to desired output dimension
        self.projection = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features using CNN layers.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional attention mask (not used in CNN but kept for API consistency)

        Returns:
            Processed tensor of shape [batch_size, seq_length, output_dim]
        """
        # Transpose for CNN (batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)

        # Apply CNN layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Transpose back (batch_size, seq_length, channels[-1])
        x = x.transpose(1, 2)

        # Project to output dimension
        return self.projection(x)


class LSTMFeatureExtractor(nn.Module):
    """LSTM feature extractor for temporal data."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 representation_dim: int = 8,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        # Input projection
        self.representation_layer = nn.Linear(input_dim, representation_dim)

        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=representation_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output projection
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.projection = nn.Linear(lstm_output_size, output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process sequence with LSTM.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask for packed sequence

        Returns:
            Processed tensor of shape [batch_size, seq_length, output_dim]
        """
        # Project the input into a high-dimensional representation space
        x = self.representation_layer(x)

        if mask is not None:
            # Pack padded sequence if mask is provided
            lengths = mask.sum(dim=1).cpu()
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed_sequence)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, _ = self.lstm(x)

        # Project to output dimension
        return self.projection(lstm_out)


class CNNLSTMFeatureExtractor(nn.Module):
    """Combined CNN and LSTM feature extractor for multi-scale temporal features."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 cnn_channels: List[int] = [32, 64, 128],
                 cnn_kernel_size: int = 5,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()

        # CNN for local feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = 1  # Process each dimension separately

        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=cnn_kernel_size, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Bi-directional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=sum(cnn_channels) * input_dim,  # Concatenated CNN features
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Instance normalization for stable training
        self.instance_norm = nn.InstanceNorm1d(lstm_hidden_size * (2 if bidirectional else 1))

        # Output projection
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.projection = nn.Linear(lstm_output_size, output_dim)

        self.input_dim = input_dim

    def _extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features using CNN layers.

        Args:
            x: Input tensor of shape [batch_size, input_dim, seq_length]

        Returns:
            CNN features of shape [batch_size, sum(cnn_channels), seq_length]
        """
        # Process each dimension separately
        batch_size, seq_length, input_dim = x.shape
        cnn_features = []

        for dim in range(input_dim):
            # Extract single dimension and add channel dimension
            dim_input = x[:, :, dim].unsqueeze(1)

            # Apply CNN layers and collect features
            features = []
            current_features = dim_input
            for cnn_layer in self.cnn_layers:
                current_features = cnn_layer(current_features)
                features.append(current_features)

            # Concatenate features from all CNN layers
            cnn_features.append(torch.cat(features, dim=1))

        # Combine features from all dimensions
        return torch.cat(cnn_features, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process sequence with CNN-LSTM architecture.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask for packed sequence

        Returns:
            Processed tensor of shape [batch_size, seq_length, output_dim]
        """
        # Extract CNN features
        cnn_features = self._extract_cnn_features(x)

        # Transpose back to sequence format for LSTM
        cnn_features = cnn_features.transpose(1, 2)  # [batch_size, seq_length, channels]

        # Apply LSTM with optional packing
        if mask is not None:
            # Pack padded sequence if mask is provided
            lengths = mask.sum(dim=1).cpu()
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                cnn_features, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed_sequence)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=cnn_features.size(1)
            )
        else:
            lstm_out, _ = self.lstm(cnn_features)

        # Apply instance normalization
        normalized = self.instance_norm(lstm_out.transpose(1, 2)).transpose(1, 2)

        # Project to output dimension
        return self.projection(normalized)


class FeatureEmbedding(nn.Module):
    """
    Enhanced feature embedding module with multiple embedding strategies.
    Supports simple linear projection, CNN, LSTM, and combined CNN-LSTM architectures.
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 max_seq_length: int = 5000,
                 embedding_mode: Optional[Union[str, nn.Module]] = None,
                 dropout: float = 0.1,
                 config: Optional[dict] = None):
        """
        Initialize the feature embedding module.

        Args:
            input_dim: Dimension of input features
            d_model: Target embedding dimension
            max_seq_length: Maximum sequence length for positional encoding
            embedding_mode: Embedding strategy ('mlp', 'cnn', 'lstm', 'cnn-lstm') or custom module
            dropout: Dropout rate
            config: Additional configuration parameters for embedding modules
        """
        super().__init__()

        # Set default config if not provided
        if config is None:
            config = {}

        # Configure embedding module based on mode
        if embedding_mode is None or embedding_mode == 'mlp':
            # Simple linear projection with non-linearity
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
        elif isinstance(embedding_mode, nn.Module):
            # Use provided custom module
            self.input_projection = embedding_mode
        elif embedding_mode == 'cnn':
            # CNN-based feature extractor
            self.input_projection = CNNFeatureExtractor(
                input_dim=input_dim,
                output_dim=d_model,
                channels=config.get('cnn_channels', [32, 64, 128]),
                kernel_size=config.get('cnn_kernel_size', 5),
                dropout=dropout
            )
        elif embedding_mode == 'lstm':
            # LSTM-based feature extractor
            self.input_projection = LSTMFeatureExtractor(
                input_dim=input_dim,
                output_dim=d_model,
                hidden_size=config.get('lstm_hidden_size', 128),
                num_layers=config.get('lstm_num_layers', 2),
                dropout=dropout,
                bidirectional=config.get('bidirectional', True)
            )
        elif embedding_mode == 'cnn-lstm':
            # Combined CNN-LSTM feature extractor
            self.input_projection = CNNLSTMFeatureExtractor(
                input_dim=input_dim,
                output_dim=d_model,
                cnn_channels=config.get('cnn_channels', [32, 64, 128]),
                cnn_kernel_size=config.get('cnn_kernel_size', 5),
                lstm_hidden_size=config.get('lstm_hidden_size', 128),
                lstm_num_layers=config.get('lstm_num_layers', 2),
                dropout=dropout,
                bidirectional=config.get('bidirectional', True)
            )
        else:
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}")

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the feature embedding module.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask tensor for padded sequences

        Returns:
            Enhanced position-encoded sequence of shape [batch_size, seq_length, d_model]
        """
        # Apply input projection
        features = self.input_projection(x, mask) if hasattr(self.input_projection, 'forward') and \
                                                     'mask' in self.input_projection.forward.__code__.co_varnames else \
            self.input_projection(x)

        # Apply layer normalization
        features = self.layer_norm(features)

        # Apply dropout and positional encoding
        output = self.pos_encoder(self.dropout(features))

        return output