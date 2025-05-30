import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple, List
from .feature_extractor import FeatureEmbedding

class OutputUnit(nn.Module):
    def __init__(self,
                 input_dim: int = 2,  # Input dimension (1D/2D/3D)
                 output_dim: Optional[int] = None,  # Output dimension (if None, automatically determined)
                 d_model: int = 512,  # Model dimension
                 nhead: int = 8,  # Number of attention heads
                 num_layers: int = 6,  # Number of transformer layers
                 dim_feedforward: int = 2048,  # Feedforward network dimension
                 dropout: float = 0.1,  # Dropout rate
                 activation: str = "gelu",  # Activation function
                 encoder_type: Union[str, nn.Module] = "mlp",  # Encoder type
                 use_decoder: bool = False,  # Whether to use decoder
                 pooling_method: str = "mean",  # How to aggregate sequence features
                 output_method: str = "conv",  # How to project to output space
                 layer_norm_eps: float = 1e-5,  # Layer norm epsilon
                 use_memory_efficient_attention: bool = False,  # Whether to use memory efficient attention
                 config: Optional[Dict[str, Any]] = None  # Additional configuration
                 ):
        super().__init__()
        if output_method == "linear":
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.LayerNorm(dim_feedforward // 2),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward // 2, output_dim)
            )

        elif output_method == "gru":
            # Use GRU for each output parameter independently
            self.output_projection = nn.Sequential(
                nn.GRU(d_model, dim_feedforward, 2, batch_first=True),
                nn.LayerNorm(dim_feedforward),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.LayerNorm(dim_feedforward // 2),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward // 2, output_dim)
            )
        elif output_method == "conv":
            # Use 1D convolutions for parameter prediction
            self.output_projection = nn.Sequential(
                nn.Conv1d(d_model, dim_feedforward, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim_feedforward),
                nn.Conv1d(dim_feedforward, dim_feedforward // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim_feedforward // 2),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(dim_feedforward // 2, output_dim)
            )
        else:
            raise ValueError(f"Unsupported output method: {output_method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_projection(x)

class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time series data
    """
    def __init__(self,
                 input_dim: int = 1,  # Dimension of input features
                 output_dim: int = 4,  # Output dimension (if None, automatically determined)
                 max_seq_len: int = 10000, # Maximum input sequence length
                 d_model: int = 512,  # Model dimension
                 nhead: int = 8,  # Number of attention heads
                 num_layers: int = 6,  # Number of transformer layers
                 dim_feedforward: int = 2048,  # Feedforward network dimension
                 dropout: float = 0.1,  # Dropout rate
                 activation: str = "gelu",  # Activation function
                 encoder_type: Union[str, nn.Module] = "mlp",  # Encoder type
                 use_decoder: bool = False,  # Whether to use decoder
                 output_method: str = "linear",  # How to project to output space
                 layer_norm_eps: float = 1e-5,  # Layer norm epsilon
                 use_memory_efficient_attention: bool = False,  # Whether to use memory efficient attention
                 config: Optional[Dict[str, Any]] = None  # Additional configuration
                 ):
        super().__init__()

        # Set default config if not provided
        self.config = {} if config is None else config

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.use_decoder = use_decoder
        self.output_method = output_method

        # Input embedding and feature extraction
        self.input_projection = FeatureEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            max_seq_length=max_seq_len,
            embedding_mode=encoder_type,
            dropout=dropout,
            config=self.config
        )

        # Configure transformer encoder with advanced options
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps
        )

        # Memory efficient attention for reduced VRAM usage
        if use_memory_efficient_attention and hasattr(encoder_layers, 'use_memory_efficient_attention'):
            encoder_layers.use_memory_efficient_attention = True

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Optional transformer decoder for sequence-to-sequence tasks
        if use_decoder:
            self.decoder_projection = nn.Linear(output_dim, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
                layer_norm_eps=layer_norm_eps
            )

            if use_memory_efficient_attention and hasattr(decoder_layer, 'use_memory_efficient_attention'):
                decoder_layer.use_memory_efficient_attention = True

            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers
            )

            # Optional query embedding for decoder
            self.query_embed = nn.Embedding(output_dim, d_model)

        # Output projection with different methods
        if output_method == "linear":
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(d_model // 2, output_dim)
            )
        elif output_method == "gru":
            # Use GRU for each output parameter independently
            self.output_projection = nn.Sequential(
                nn.GRU(d_model, dim_feedforward, 2, batch_first=True),
                nn.LayerNorm(dim_feedforward),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.LayerNorm(dim_feedforward // 2),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(dim_feedforward // 2, output_dim)
            )
        elif output_method == "conv":
            # Use 1D convolutions for parameter prediction
            self.output_projection = nn.Sequential(
                nn.Conv1d(d_model, dim_feedforward, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim_feedforward),
                nn.Conv1d(dim_feedforward, dim_feedforward // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(dim_feedforward // 2),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(dim_feedforward // 2, output_dim)
            )
        else:
            raise ValueError(f"Unsupported output method: {output_method}")

    def forward(self,
                src: torch.Tensor,
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                return_features: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the force field transformer model.

        Args:
            src: Input trajectory of shape [batch_size, seq_length, input_dim]
            tgt: Optional target sequence for decoder of shape [batch_size, tgt_length]
            src_mask: Optional mask for transformer encoder
            src_key_padding_mask: Optional key padding mask for encoder
            tgt_mask: Optional mask for decoder
            memory_mask: Optional mask for encoder-decoder attention
            return_features: Whether to return encoder features along with outputs

        Returns:
            Either:
                - Predicted parameters of shape [batch_size, output_dim]
                - Tuple of (parameters, encoder_features) if return_features=True
        """
        # Input embedding and feature extraction
        x = self.input_projection(src, src_key_padding_mask)

        # Transformer encoder
        memory = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Apply transformer decoder if enabled
        if self.use_decoder:
            if tgt is not None:
                if len(tgt.shape) < 3:
                    tgt = self.decoder_projection(tgt.unsqueeze(-1))

                # Use provided target sequence
                decoder_output = self.transformer_decoder(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask
                )
                features = decoder_output
            else:
                # Generate queries using learned embeddings
                batch_size = src.size(0)
                queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

                decoder_output = self.transformer_decoder(
                    queries,
                    memory,
                    memory_mask=memory_mask
                )
                features = decoder_output
        else:
            # Use encoder output directly
            features = memory
        features = self.output_projection(features)

        if return_features:
            return features, memory
        else:
            return features




