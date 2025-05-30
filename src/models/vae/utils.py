import torch

from .units import UnetEncoder, UnetDecoder, TransformerEncoder, TransformerDecoder
from .vae_base import VariationalAutoencoder
from .conditional import ConditionalVAE, ConditionalVAE_CrossAttention

# --- Factory Function ---
def create_vae_model(
    input_feature_dim: int,    # Number of features per time step in input
    sequence_len: int,         # Length of the input sequence.
    latent_feature_dim: int,   # Number of features per time step in the latent space
    scale_factor: int,         # Common for VAEs that reduce spatial/temporal dim
    backbone_type: str,        # 'unet' or 'transformer'
    use_kluber_scaling: bool = True,
    unet_config: dict = None,      # Config for UNet backbone if type is 'unet'
    transformer_config: dict = None # Config for Transformer backbone if type is 'transformer'
) -> VariationalAutoencoder:
    """
    Factory function to create a VariationalAutoencoder model with a choice of backbone.

    Args:
        input_feature_dim (int): Dimension of features at each time step of input.
        sequence_len (int): Length of the input sequence.
        latent_feature_dim (int): Dimension of features at each time step of latent space.
        backbone_type (str): Type of backbone model ('unet' or 'transformer').
        use_kluber_scaling (bool): Whether to use the 0.18215 scaling.
        unet_config (dict, optional): Configuration for UNet backbone.
            Expected keys: e.g., 'seq_len_divisor' (int, e.g., 8),
                           'depth' (int), 'start_channels' (int), etc.
        transformer_config (dict, optional): Configuration for Transformer backbone.
            Expected keys: e.g., 'seq_len_divisor' (int, e.g., 8),
                           'embed_dim' (int), 'n_heads' (int), 'depth' (int), etc.

    Returns:
        VariationalAutoencoder: The instantiated VAE model.
    """
    encoder_backbone = None
    decoder_backbone = None

    if backbone_type.lower() == 'transformer':
        if transformer_config is None:
            transformer_config = {}

        # Encoder backbone output features = 2 * latent_feature_dim (for mean and log_var)
        # Transformer usually takes (Batch, Seq, Features)
        encoder_backbone = TransformerEncoder(
            input_dim=input_feature_dim,
            output_dim=latent_feature_dim * 2,
            scale_factor=scale_factor,
            **transformer_config
        )
        decoder_backbone = TransformerDecoder(
            input_dim=latent_feature_dim,
            output_dim=input_feature_dim,
            scale_factor=scale_factor,
            **transformer_config
        )
        # VAE_Encoder_Wrapper will expect chunking on dim=-1 for (B,S,F) format

    elif backbone_type.lower() == 'unet':
        if unet_config is None:
            unet_config = {}

        # UNets often take (Batch, Channels, SeqLen)
        # Encoder output channels = 2 * latent_feature_dim
        # Decoder input channels = latent_feature_dim
        # Note: The DummyUnet assumes input_channels is feature_dim for (B,C,S)
        # If your VAE processes (B,S,F), you might need to permute before/after Unet
        # or design Unet for (B,S,F) directly (less common for standard Unet convs)

        # For UNet, latent_feature_dim is often interpreted as number of channels in latent space
        encoder_backbone = UnetEncoder(
            input_channels=input_feature_dim,
            output_channels=latent_feature_dim,
        )
        decoder_backbone = UnetDecoder(
            input_channels=latent_feature_dim,     # Latent channels
            output_channels=input_feature_dim,   # Reconstruct original channels
        )
        # VAE_Encoder_Wrapper will need to chunk on dim=1 for (B,C,S) format if Unet outputs channels first

    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Choose 'unet' or 'transformer'.")

    vae_model = VariationalAutoencoder(
        input_feature_dim=input_feature_dim,
        latent_feature_dim=latent_feature_dim,
        sequence_len=sequence_len,
        encoder_backbone=encoder_backbone,
        decoder_backbone=decoder_backbone,
        use_kluber_scaling=use_kluber_scaling,
    )
    return vae_model

def create_conditiaonl_vae_model(
    input_feature_dim: int,    # Number of features per time step in input
    sequence_len: int,         # Length of the input sequence.
    latent_feature_dim: int,   # Number of features per time step in the latent space
    conditional_dim: int,       # Number of features per time step in the condition
    scale_factor: int,         # Common for VAEs that reduce spatial/temporal dim
    backbone_type: str,        # 'unet' or 'transformer'
    use_crossattention: bool = False, # Use cross attention or not
    use_kluber_scaling: bool = True,
    unet_config: dict = None,      # Config for UNet backbone if type is 'unet'
    transformer_config: dict = None # Config for Transformer backbone if type is 'transformer'
) -> VariationalAutoencoder:
    """
    Factory function to create a VariationalAutoencoder model with a choice of backbone.

    Args:
        input_feature_dim (int): Dimension of features at each time step of input.
        sequence_len (int): Length of the input sequence.
        latent_feature_dim (int): Dimension of features at each time step of latent space.
        backbone_type (str): Type of backbone model ('unet' or 'transformer').
        use_kluber_scaling (bool): Whether to use the 0.18215 scaling.
        unet_config (dict, optional): Configuration for UNet backbone.
            Expected keys: e.g., 'seq_len_divisor' (int, e.g., 8),
                           'depth' (int), 'start_channels' (int), etc.
        transformer_config (dict, optional): Configuration for Transformer backbone.
            Expected keys: e.g., 'seq_len_divisor' (int, e.g., 8),
                           'embed_dim' (int), 'n_heads' (int), 'depth' (int), etc.

    Returns:
        VariationalAutoencoder: The instantiated VAE model.
    """
    encoder_backbone = None
    decoder_backbone = None

    if backbone_type.lower() == 'transformer':
        if transformer_config is None:
            transformer_config = {}

        # Encoder backbone output features = 2 * latent_feature_dim (for mean and log_var)
        # Transformer usually takes (Batch, Seq, Features)
        encoder_backbone = TransformerEncoder(
            input_dim=input_feature_dim,
            output_dim=latent_feature_dim * 2,
            scale_factor=scale_factor,
            **transformer_config
        )
        decoder_backbone = TransformerDecoder(
            input_dim=latent_feature_dim,
            output_dim=input_feature_dim,
            scale_factor=scale_factor,
            **transformer_config
        )
        # VAE_Encoder_Wrapper will expect chunking on dim=-1 for (B,S,F) format

    elif backbone_type.lower() == 'unet':
        if unet_config is None:
            unet_config = {}

        # UNets often take (Batch, Channels, SeqLen)
        # Encoder output channels = 2 * latent_feature_dim
        # Decoder input channels = latent_feature_dim
        # Note: The DummyUnet assumes input_channels is feature_dim for (B,C,S)
        # If your VAE processes (B,S,F), you might need to permute before/after Unet
        # or design Unet for (B,S,F) directly (less common for standard Unet convs)

        # For UNet, latent_feature_dim is often interpreted as number of channels in latent space
        encoder_backbone = UnetEncoder(
            input_channels=input_feature_dim,
            output_channels=latent_feature_dim,
        )
        decoder_backbone = UnetDecoder(
            input_channels=latent_feature_dim,     # Latent channels
            output_channels=input_feature_dim,   # Reconstruct original channels
        )
        # VAE_Encoder_Wrapper will need to chunk on dim=1 for (B,C,S) format if Unet outputs channels first

    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Choose 'unet' or 'transformer'.")

    if use_crossattention:
        vae_model = ConditionalVAE_CrossAttention(
            input_feature_dim=input_feature_dim,
            latent_feature_dim=latent_feature_dim,
            sequence_len=sequence_len,
            encoder_backbone=encoder_backbone,
            decoder_backbone=decoder_backbone,
            use_kluber_scaling=use_kluber_scaling,
            context_dim=conditional_dim
        )
    else:
        vae_model = ConditionalVAE(
            input_feature_dim=input_feature_dim,
            latent_feature_dim=latent_feature_dim,
            sequence_len=sequence_len,
            encoder_backbone=encoder_backbone,
            decoder_backbone=decoder_backbone,
            condition_dim=conditional_dim,
            use_kluber_scaling=use_kluber_scaling,
        )

    return vae_model
