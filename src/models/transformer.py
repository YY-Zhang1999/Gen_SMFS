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
        ...

class DiffusionTransformer(nn.Module):
    """
    Transformer model for force field reconstruction with enhanced architecture.

    Features:
    - Flexible input embedding with various encoding strategies
    - Configurable transformer architecture with encoder-only or encoder-decoder modes
    - Advanced output projection methods for force field parameters
    - Support for sequence modeling and feature pooling
    """

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
                 normalizer: Optional[TrajectoryNormalizer] = None,
                 config: Optional[Dict[str, Any]] = None  # Additional configuration
                 ):
        super().__init__()

        # Set default config if not provided
        self.config = {} if config is None else config

        # Auto-determine output dimension based on input dimension if not specified
        output_dims = {1: 1, 2: 3, 3: 6}  # Maps input dimension to Jacobian matrix elements
        if output_dim is None:
            output_dim = output_dims.get(input_dim, input_dim * input_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.use_decoder = use_decoder
        self.pooling_method = pooling_method
        self.output_method = output_method
        self.normalizer = normalizer

        # Input embedding and feature extraction
        self.input_projection = FeatureEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            max_seq_length=self.config.get("max_seq_length", 10000),
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

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence features into a single vector.

        Args:
            x: Sequence features of shape [batch_size, seq_length, d_model]

        Returns:
            Pooled features of shape [batch_size, d_model]
        """
        if self.pooling_method == "entire":
            # Use entire token features
            return x
        if self.pooling_method == "last":
            # Use last token features
            return x[:, -1]
        elif self.pooling_method == "mean":
            # Average pooling across sequence
            return torch.mean(x, dim=1)
        elif self.pooling_method == "max":
            # Max pooling across sequence
            return torch.max(x, dim=1)[0]
        elif self.pooling_method == "cls":
            # Use first token features (like BERT's [CLS] token)
            return x[:, 0]
        elif self.pooling_method == "attention":
            # Learnable attention-weighted pooling
            scores = self.attention_pool(x)
            weighted = scores.unsqueeze(-1) * x
            return torch.sum(weighted, dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

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

        # Apply output projection based on method
        if self.output_method == "conv":
            # Apply convolutional projection (needs channel-first format)
            features_channels = features.transpose(1, 2)  # [batch, d_model, seq_len]
            parameters = self.output_projection(features_channels)
        else:
            # Apply linear projection to pooled features
            pooled_features = self._pool_features(features)
            parameters = self.output_projection(pooled_features)

        if return_features:
            return parameters, memory
        else:
            return parameters

    def inverse_transform(self,
                          trajectories: torch.Tensor = None,
                          forces: torch.Tensor = None,
                          params: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
                Convert normalized data back to original scale.

                Args:
                    trajectories: Normalized positions
                    forces: Normalized forces
                    parmas: Normalized parmas matrices

                Returns:
                    Tuple of arrays in original scale
        """
        if self.normalizer:
            return self.normalizer.inverse_transform(trajectories, forces, params)
        else:
            raise 'No avaliable normalization method'

    def transform(self,
                  trajectories: torch.Tensor = None,
                  forces: torch.Tensor = None,
                  params: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
        Normalize the data using computed statistics.

        Args:
            trajectories: Array of positions
            forces: Array of forces
            params: Array of parameters

        Returns:
            Tuple of normalized arrays
        """
        if self.normalizer:
            return self.normalizer.transform(trajectories, forces, params)
        else:
            raise 'No avaliable normalization method'


class ForceFieldTransformerWithPhysics(nn.Module):
    """Extension of ForceFieldTransformer with explicit physics constraints."""

    def __init__(self,
                 model: ForceFieldTransformer,
                 config: Dict,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.physics_layer = PhysicsLayer(temperature=config['physical']['temperature'],
                                          viscosity=config['physical']['viscosity'],
                                          particle_radius=config['physical']['particle_radius'],
                                          dt=config['physical']['dt'],
                                          dim=config['physical']['dim'])
        self.config = config

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with physics constraints.

        Args:
            src: Input trajectory (batch_size, seq_length, input_dim)
            src_mask: Mask for transformer (optional)

        Returns:
            jacobian: Predicted Jacobian matrices
            physical_params: Tuple of (k1, k2, theta, omega)
            forces: Predicted forces
        """
        # Get Jacobian and parameters from base model
        params = self.model.forward(src, src_mask)

        raw_positions, _, raw_params = self.model.inverse_transform(trajectories=src, params=params)

        jacobian = self.parameters_to_jacobian(raw_params)

        # Apply physics layer to get forces
        forces = self.physics_layer(raw_positions, jacobian)

        return params, forces

    def parameters_to_jacobian(self, params: torch.Tensor) -> torch.Tensor:
        return self.physics_layer.parameters_to_jacobian(params)

