# Modified from https://github.com/hkproj/pytorch-stable-diffusion/
import math

import torch
from torch import nn
from torch.nn import functional as F
from Gen_SMFS.src.models.components import TimeSeriesTransformer
from typing import Optional, Union, Dict, Any, Tuple, List

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # GroupNorm for 1D data will now operate on (Batch_Size, Channels, Length)
        self.groupnorm = nn.GroupNorm(32, channels)
        # SelfAttention takes (Batch_Size, Sequence_Length, Features)
        # Here, Features = channels, Sequence_Length = Length
        self.attention = SelfAttention(1, channels)  # 1 head, channels as embed_dim

    def forward(self, x):
        # x: (Batch_Size, Features, Length) for 1D data

        residue = x

        # (Batch_Size, Features, Length) -> (Batch_Size, Features, Length)
        x = self.groupnorm(x)

        n, c, l = x.shape  # n: Batch_Size, c: Features, l: Length

        # (Batch_Size, Features, Length) -> (Batch_Size, Length, Features).
        # Each point in the time series becomes a feature of size "Features", the sequence length is "Length".
        x = x.transpose(-1, -2)  # This is crucial for attention

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Length, Features) -> (Batch_Size, Length, Features)
        x = self.attention(x)

        # (Batch_Size, Length, Features) -> (Batch_Size, Features, Length)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Length) + (Batch_Size, Features, Length) -> (Batch_Size, Features, Length)
        x += residue

        # (Batch_Size, Features, Length)
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # Replace Conv2d with Conv1d
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # Replace Conv2d with Conv1d
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # Replace Conv2d with Conv1d for the residual connection
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (Batch_Size, In_Channels, Length)

        residue = x

        # (Batch_Size, In_Channels, Length) -> (Batch_Size, In_Channels, Length)
        x = self.groupnorm_1(x)

        # (Batch_Size, In_Channels, Length) -> (Batch_Size, In_Channels, Length)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Length) -> (Batch_Size, Out_Channels, Length)
        x = self.conv_1(x)

        # (Batch_Size, Out_Channels, Length) -> (Batch_Size, Out_Channels, Length)
        x = self.groupnorm_2(x)

        # (Batch_Size, Out_Channels, Length) -> (Batch_Size, Out_Channels, Length)
        x = F.silu(x)

        # (Batch_Size, Out_Channels, Length) -> (Batch_Size, Out_Channels, Length)
        x = self.conv_2(x)

        # (Batch_Size, Out_Channels, Length) -> (Batch_Size, Out_Channels, Length)
        return x + self.residual_layer(residue)


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

            # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output


class UnetEncoder(nn.Sequential):
    def __init__(self, input_channels: int = 1, output_channels: int = 4):
        super().__init__(
            # (Batch_Size, Channel, Length) -> (Batch_Size, 128, Length)
            # Assuming input channel is 1 for Force-Extension curve
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),  # Change to Conv1d (input 1 channel)

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length / 2)
            # Stride 2 for downsampling in 1D
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),  # Change to Conv1d

            # (Batch_Size, 128, Length / 2) -> (Batch_Size, 256, Length / 2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Length / 2) -> (Batch_Size, 256, Length / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Length / 2) -> (Batch_Size, 256, Length / 4)
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),  # Change to Conv1d

            # (Batch_Size, 256, Length / 4) -> (Batch_Size, 512, Length / 4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 8)
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),  # Change to Conv1d

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            nn.SiLU(),

            # Output has 8 channels for mean and log_variance (4 for each)
            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 8, Length / 8).
            nn.Conv1d(512, output_channels * 2, kernel_size=3, padding=1),  # Change to Conv1d

            # (Batch_Size, 8, Length / 8) -> (Batch_Size, 8, Length / 8)
            nn.Conv1d(output_channels * 2, output_channels * 2, kernel_size=1, padding=0),  # Change to Conv1d
        )

    def forward(self, x):
        # x: (Batch_Size, Channel, Length) - assuming 1 input channel for F-E curve
        # noise: (Batch_Size, 4, Length / 8) - latent space noise

        for module in self:
            # Padding at downsampling should be asymmetric for 1D (see #8)
            if getattr(module, 'stride', None) == (2,):  # Check for 1D stride tuple
                # Pad: (Padding_Left, Padding_Right).
                # Pad with zeros on the right.
                # (Batch_Size, Channel, Length) -> (Batch_Size, Channel, Length + 1)
                x = F.pad(x, (0, 1))  # Change to 1D padding

            x = module(x)

        return x


class UnetDecoder(nn.Sequential):
    def __init__(self, input_channels: int = 4, output_channels: int = 1):
        super().__init__(
            # (Batch_Size, 4, Length / 8) -> (Batch_Size, 4, Length / 8)
            nn.Conv1d(input_channels, input_channels, kernel_size=1, padding=0),  # Change to Conv1d

            # (Batch_Size, 4, Length / 8) -> (Batch_Size, 512, Length / 8)
            nn.Conv1d(input_channels, 512, kernel_size=3, padding=1),  # Change to Conv1d

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 8)
            VAE_ResidualBlock(512, 512),

            # Repeats the elements of the data by scale_factor.
            # (Batch_Size, 512, Length / 8) -> (Batch_Size, 512, Length / 4)
            # For 1D, use mode='linear' for smooth upsampling
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 4)
            nn.Conv1d(512, 512, kernel_size=3, padding=1),  # Change to Conv1d

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Length / 4) -> (Batch_Size, 512, Length / 2)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # Change to linear for 1D

            # (Batch_Size, 512, Length / 2) -> (Batch_Size, 512, Length / 2)
            nn.Conv1d(512, 512, kernel_size=3, padding=1),  # Change to Conv1d

            # (Batch_Size, 512, Length / 2) -> (Batch_Size, 256, Length / 2)
            VAE_ResidualBlock(512, 256),

            # (Batch_Size, 256, Length / 2) -> (Batch_Size, 256, Length / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Length / 2) -> (Batch_Size, 256, Length / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Length / 2) -> (Batch_Size, 256, Length)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # Change to linear for 1D

            # (Batch_Size, 256, Length) -> (Batch_Size, 256, Length)
            nn.Conv1d(256, 256, kernel_size=3, padding=1),  # Change to Conv1d

            # (Batch_Size, 256, Length) -> (Batch_Size, 128, Length)
            VAE_ResidualBlock(256, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            nn.GroupNorm(32, 128),

            # (Batch_Size, 128, Length) -> (Batch_Size, 128, Length)
            nn.SiLU(),

            # (Batch_Size, 128, Length) -> (Batch_Size, 3, Length) (Output channels for F-E curves, e.g., 1 for Force)
            # The original decoder outputs 3 channels, typically for RGB images.
            # For F-E curves, you usually have 1 channel (Force).
            # This should be configurable or set to 1.
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),  # Change to Conv1d, output 1 channel
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Length / 8)

        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 1, Length) if output channels are 1
        # The output shape will be (Batch_Size, 1, Original_Length)
        return x


class TransformerEncoder(TimeSeriesTransformer):
    def __init__(self,
                 input_dim: int = 1,
                 output_dim: int = 4,  # This is the final output dimension of this specialized encoder
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 encoder_type: Union[str, nn.Module] = "mlp",
                 # use_decoder should be False for an encoder
                 layer_norm_eps: float = 1e-5,
                 use_memory_efficient_attention: bool = False,
                 config: Optional[Dict[str, Any]] = None,
                 scale_factor: int = 4
                 ):
        super().__init__(
            input_dim=input_dim,
            # The parent TimeSeriesTransformer will use this output_dim for its own output_projection.
            # So, the output of super().forward() will have `output_dim` as its feature dimension.
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            encoder_type=encoder_type,
            use_decoder=False,  # Explicitly set to False for an encoder
            layer_norm_eps=layer_norm_eps,
            use_memory_efficient_attention=use_memory_efficient_attention,
            config=config
        )
        self.scale_factor = scale_factor
        # The output of super().forward() will have `output_dim` features (e.g., 4 in your default).
        # After reshaping, the features will be `scale_factor * output_dim`.
        # This projection layer will project it back to `output_dim` or a desired new dimension.
        # If the final output dimension of this TransformerEncoder should still be `output_dim`,
        # then the projection is from `scale_factor * output_dim` to `output_dim`.
        self.final_projection = nn.Linear(self.scale_factor * self.output_dim, self.output_dim)

    def forward(self,
                src: torch.Tensor,
                # tgt is not typically used by a standalone encoder's forward, but keep signature for consistency if parent needs it
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                # Remove decoder-specific args not used by encoder part of parent if use_decoder=False
                # tgt_mask: Optional[torch.Tensor] = None, # Not used if use_decoder=False
                # memory_mask: Optional[torch.Tensor] = None, # Not used if use_decoder=False
                return_features: bool = False  # This custom arg needs handling if different from parent
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the specialized TransformerEncoder model.
        It first uses the parent's transformer encoding and then rescales the sequence length.

        Args:
            src: Input trajectory of shape [batch_size, seq_length, input_dim]
            tgt: Optional target sequence (likely None or ignored for encoder).
            src_mask: Optional mask for transformer encoder.
            src_key_padding_mask: Optional key padding mask for encoder.
            return_features: Custom argument, if True, may need to return intermediate features.

        Returns:
            Predicted parameters of shape [batch_size, seq_length / scale_factor, output_dim]
            (or tuple if return_features is True and implemented)
        """
        # x will have shape [batch_size, seq_length, self.output_dim]
        # because super().__init__ was called with self.output_dim, and parent TimeSeriesTransformer
        # has self.output_projection that maps d_model to output_dim.
        x = super().forward(src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        batch_size, seq_len, feature_dim = x.shape  # feature_dim is self.output_dim

        if seq_len % self.scale_factor != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by scale_factor ({self.scale_factor})")

        target_len = seq_len // self.scale_factor

        # Reshape: (batch_size, seq_len, feature_dim) -> (batch_size, target_len, scale_factor * feature_dim)
        # .contiguous() is important before .view() if tensor is not already contiguous.
        x_reshaped = x.contiguous().view(batch_size, target_len, self.scale_factor * feature_dim)

        # Project to the final desired dimension for this encoder module
        # Input: (batch_size, target_len, scale_factor * feature_dim)
        # Output: (batch_size, target_len, self.output_dim)
        output = self.final_projection(x_reshaped)

        if return_features:  # Assuming x from super().forward could be considered as features
            return output, x
        return output


class TransformerDecoder(TimeSeriesTransformer):
    def __init__(self,
                 input_dim: int = 1,
                 output_dim: int = 4,  # This is the final output dimension of this specialized decoder
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 encoder_type: Union[str, nn.Module] = "mlp",
                 # use_decoder should be True for a decoder
                 layer_norm_eps: float = 1e-5,
                 use_memory_efficient_attention: bool = False,
                 config: Optional[Dict[str, Any]] = None,
                 scale_factor: int = 4
                 ):
        super().__init__(
            input_dim=input_dim,
            # The parent TimeSeriesTransformer will use this output_dim for its own output_projection.
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            encoder_type=encoder_type,
            use_decoder=False,  # Explicitly set to True for a decoder
            layer_norm_eps=layer_norm_eps,
            use_memory_efficient_attention=use_memory_efficient_attention,
            config=config
        )
        self.scale_factor = scale_factor
        # nn.Upsample for 1D sequence. Mode 'linear' needs 3D input: (N, C, L_in)
        # It will upsample the L_in dimension.
        self.upsample_layer = nn.Upsample(scale_factor=self.scale_factor, mode='linear', align_corners=False)
        # No additional projection is defined here, assuming self.output_dim is the final feature dim.

    def forward(self,
                src: torch.Tensor,  # Source sequence for the encoder part of the parent transformer
                tgt: Optional[torch.Tensor] = None,  # Target sequence for the decoder part
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                # Added these based on typical nn.TransformerDecoder arguments
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_features: bool = False  # This custom arg needs handling
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the specialized TransformerDecoder model.
        It uses the parent's transformer decoding and then upsamples the sequence length.

        Args:
            src: Input sequence for the encoder, shape [batch_size, src_seq_length, input_dim]
            tgt: Target sequence for the decoder, shape [batch_size, tgt_seq_length, input_dim]
            src_mask: Mask for encoder self-attention.
            src_key_padding_mask: Padding mask for encoder input.
            tgt_mask: Mask for decoder self-attention.
            memory_mask: Mask for decoder encoder-attention.
            tgt_key_padding_mask: Padding mask for decoder input.
            memory_key_padding_mask: Padding mask for memory (encoder output).
            return_features: Custom argument.

        Returns:
            Upsampled output of shape [batch_size, tgt_seq_length * scale_factor, output_dim]
            (or tuple if return_features is True and implemented)
        """

        # x will have shape [batch_size, tgt_seq_length, self.output_dim]
        # This is because super().__init__ set use_decoder=True, so parent's forward
        # will perform full encoding of `src` and decoding of `tgt` with `src` as memory.
        # The final self.output_projection in parent class ensures last dim is self.output_dim.
        x = super().forward(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        batch_size, seq_len, feature_dim = x.shape  # feature_dim is self.output_dim

        # Upsample the sequence length
        # nn.Upsample expects (N, C, L_in) for mode='linear' (1D upsampling)
        # Current x is (batch_size, seq_len, feature_dim)
        x_permuted = x.permute(0, 2, 1)  # -> (batch_size, feature_dim, seq_len)

        # Upsamples the last dimension (seq_len) by scale_factor
        x_upsampled_permuted = self.upsample_layer(x_permuted)  # -> (batch_size, feature_dim, seq_len * scale_factor)

        # Permute back to (batch_size, new_seq_len, feature_dim)
        output = x_upsampled_permuted.permute(0, 2, 1)
        # output shape: (batch_size, seq_len * scale_factor, feature_dim)

        if return_features:  # Assuming x (before upsampling) could be features
            return output, x
        return output

if __name__ == '__main__':
    input = torch.rand(64, 100, 1)
    latent_dim = 4
    scale_factor = 2
    encoder = TransformerEncoder(1, latent_dim, scale_factor=scale_factor)
    #encoder = UnetEncoder()

    latent = encoder(input)
    print(latent.shape)

    decoder = TransformerDecoder(latent_dim, 1, scale_factor=scale_factor)
    reconstruction = decoder(latent)
    print(reconstruction.shape)






