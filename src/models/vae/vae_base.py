import torch
from torch import nn

class VAE_Encoder_Wrapper(nn.Module):
    def __init__(self, backbone_model: nn.Module):
        """
        Encoder part of the VAE.
        It uses a backbone_model to get features, then projects to mean and log_variance.

        Args:
            backbone_model (nn.Module): The core network (e.g., UNet, Transformer)
                                        that extracts features and outputs a tensor
                                        expected to be split into mean and log_variance.
                                        Expected output shape from backbone_model:
                                        (Batch_Size, Seq_Len_Latent, 2 * Latent_Feature_Dim)
                                        or (Batch_Size, 2 * Latent_Feature_Dim, Seq_Len_Latent)
                                        The chunking dimension will depend on this.
        """
        super().__init__()
        self.backbone_model = backbone_model

    def forward(self, x):
        # x: (Batch_Size, Seq_Len, Input_Features_Dim) or (Batch_Size, Input_Features_Dim, Seq_Len)
        # Output of backbone_model should be ready to be split into mean and log_var.
        # Example: if backbone_model outputs (Batch_Size, Seq_Len_Latent, 2 * Latent_Features_Dim)
        features = self.backbone_model(x)

        # Assuming the last dimension is 2 * Latent_Features_Dim and needs to be split.
        # If channels are dim 1, use dim=1.
        # Based on your TimeSeriesTransformer(2, 8*2) example, output_dim is the last one.
        mean, log_variance = torch.chunk(features, 2, dim=-1) # Splitting the feature dimension

        # It's common to clamp log_variance for stability.
        log_variance = torch.clamp(log_variance, -30, 20)

        return mean, log_variance


class VAE_Decoder_Wrapper(nn.Module):
    def __init__(self, backbone_model: nn.Module, apply_input_scaling: bool = True, scaling_factor: float = 0.18215):
        """
        Decoder part of the VAE.
        It uses a backbone_model to reconstruct data from the latent variable z.

        Args:
            backbone_model (nn.Module): The core network (e.g., UNet, Transformer)
                                        that reconstructs data from latent z.
                                        Expected input shape for backbone_model:
                                        (Batch_Size, Seq_Len_Latent, Latent_Feature_Dim)
                                        or (Batch_Size, Latent_Feature_Dim, Seq_Len_Latent)
            apply_input_scaling (bool): Whether to unscale the input z (if it was scaled).
            scaling_factor (float): The factor used for scaling/unscaling z.
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.apply_input_scaling = apply_input_scaling
        self.scaling_factor = scaling_factor

    def forward(self, z):
        # z: Latent variable, e.g., (Batch_Size, Seq_Len_Latent, Latent_Features_Dim)
        if self.apply_input_scaling:
            z = z / self.scaling_factor # Unscale if z was scaled before passing to decoder

        reconstructed_x = self.backbone_model(z)
        return reconstructed_x


class VariationalAutoencoder(nn.Module):
    model_name = "VAE" # Class attribute for model name

    def __init__(
            self,
            input_feature_dim: int, # Dimension of features at each time step of input sequence
            sequence_len: int,      # Length of the input sequence
            latent_feature_dim: int,# Dimension of features at each time step of latent sequence
            # latent_sequence_len will be determined by the encoder_backbone
            encoder_backbone: nn.Module, # The actual network for encoding
            decoder_backbone: nn.Module, # The actual network for decoding
            use_kluber_scaling: bool = True # Whether to use the 0.18215 scaling from SD VAE
    ):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.sequence_len = sequence_len
        self.latent_feature_dim = latent_feature_dim # This is per time-step in latent space

        # The VAE_Encoder_Wrapper will use the encoder_backbone
        self.encoder = VAE_Encoder_Wrapper(encoder_backbone)

        # The VAE_Decoder_Wrapper will use the decoder_backbone
        # The scaling factor is only applied if use_kluber_scaling is True
        self.decoder = VAE_Decoder_Wrapper(decoder_backbone,
                                           apply_input_scaling=use_kluber_scaling,
                                           scaling_factor=0.18215 if use_kluber_scaling else 1.0)

        self.use_kluber_scaling = use_kluber_scaling
        self.kluber_scaling_factor = 0.18215

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        # Sample epsilon from a standard normal distribution
        eps = torch.randn_like(std)
        # Element-wise multiplication
        return mean + eps * std

    def forward(self, x):
        # x shape depends on your backbone, e.g., (Batch, Seq_Len, Input_Features)
        # or (Batch, Input_Features, Seq_Len)
        mean, log_var = self.encoder(x)

        # z_unscaled is the direct sample from N(mean, var)
        z_unscaled = self.reparameterize(mean, log_var)

        if self.use_kluber_scaling:
            # This scaling is specific, e.g., used in Stable Diffusion's VAE.
            # It conditions the latent space before the decoder.
            z_to_decode = z_unscaled * self.kluber_scaling_factor
        else:
            z_to_decode = z_unscaled

        x_reconstructed = self.decoder(z_to_decode)

        # Return reconstructed x, and mean & log_var for KL divergence loss calculation
        return x_reconstructed, mean, log_var

    @torch.no_grad()
    def predict(self, x_in): # Renamed to avoid conflict with nn.Module.X
        """Generates reconstructions using the mean of the latent space."""
        self.eval()
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.FloatTensor(x_in)
        x_in = x_in.to(next(self.parameters()).device)

        # Pass input through the VAE_Encoder_Wrapper to get mean and log_var
        mean, _ = self.encoder(x_in) # We only need the mean for deterministic prediction

        if self.use_kluber_scaling:
            z_to_decode = mean * self.kluber_scaling_factor
        else:
            z_to_decode = mean

        x_reconstructed = self.decoder(z_to_decode)
        self.train() # Reset to training mode if changed
        return x_reconstructed.cpu().numpy()

    def get_num_trainable_variables(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_prior_samples(self, num_samples, latent_seq_len_for_sampling=None):
        """Samples from the prior (standard normal) and decodes."""
        self.eval()
        device = next(self.parameters()).device

        # Determine the latent sequence length.
        # This might be fixed by your encoder_backbone's architecture.
        # If your encoder_backbone always outputs a fixed latent_seq_len:
        if latent_seq_len_for_sampling is None:
            # You might need to infer this, e.g., by a dummy forward pass if not easily available
            # Or store it as an attribute if encoder guarantees fixed output length
            # For now, let's assume a placeholder or it needs to be provided.
            # Example: self.latent_sequence_len_output (set during __init__)
            # If TimeSeriesTransformer reduces length by /8, then:
            # latent_seq_len_for_sampling = self.sequence_len // 8 (approx)
            # This is a simplification; the actual latent_seq_len depends on encoder_backbone.
            # A robust way is to have encoder_backbone itself define its output latent_seq_len.
            print("Warning: latent_seq_len_for_sampling not provided, using self.sequence_len // 8 as a placeholder. Please verify.")
            latent_seq_len_for_sampling = self.sequence_len // 8


        # z_prior is sampled from N(0, I)
        z_prior = torch.randn(num_samples, latent_seq_len_for_sampling, self.latent_feature_dim).to(device)

        if self.use_kluber_scaling:
            # Note: The Kluber scaling is typically applied to z *from the encoder*.
            # When sampling from prior for generation, we usually sample z ~ N(0,I)
            # and directly decode it. The decoder then un-does its internal scaling.
            # So, if decoder un-scales by self.kluber_scaling_factor, the z_prior here
            # should be what the decoder expects *after* its internal un-scaling.
            # Effectively, if decoder does z_in / factor, then we can pass z_prior * factor
            # or more simply, pass z_prior and let decoder handle it.
            # The current VAE_Decoder_Wrapper un-scales its input if apply_input_scaling is True.
            # So, if we want the *effect* of a scaled latent to be decoded, we pass z_prior.
            # If the scaling factor is part of the "latent space definition" learned by encoder,
            # then prior samples should also be scaled.
            # Given the structure, the scaling is applied to the *output* of the reparameterization.
            # So for prior sampling, we sample z_unscaled, then scale it, then decode.
            z_to_decode = z_prior * self.kluber_scaling_factor
        else:
            z_to_decode = z_prior

        samples = self.decoder(z_to_decode)
        self.train() # Reset to training mode
        return samples.cpu().numpy()

    # get_prior_samples_given_Z can remain similar, ensure Z is scaled if needed before decoding.
    @torch.no_grad()
    def get_prior_samples_given_Z(self, Z_in): # Z_in is numpy array
        """Decodes a given latent variable Z."""
        self.eval()
        if not isinstance(Z_in, torch.Tensor):
            Z_in = torch.FloatTensor(Z_in)
        Z_in = Z_in.to(next(self.parameters()).device)

        # Z_in is assumed to be the "unscaled" latent sample from N(mean, var) or N(0,I)
        # The scaling is applied before decoding if use_kluber_scaling
        if self.use_kluber_scaling:
            z_to_decode = Z_in * self.kluber_scaling_factor
        else:
            z_to_decode = Z_in

        samples = self.decoder(z_to_decode)
        self.train()
        return samples.cpu().numpy()

