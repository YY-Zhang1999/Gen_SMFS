import torch
from torch import nn
from .vae_base import VariationalAutoencoder, VAE_Decoder_Wrapper
from .units import CrossAttention

class ConditionalVAE(VariationalAutoencoder):
    model_name = "ConditionalVAE"

    def __init__(
        self,
        input_feature_dim: int,
        sequence_len: int,
        latent_feature_dim: int,
        encoder_backbone: nn.Module,
        decoder_backbone: nn.Module,
        condition_dim: int,
        use_kluber_scaling: bool = True
    ):
        super().__init__(input_feature_dim + condition_dim, sequence_len, latent_feature_dim,
                         encoder_backbone, decoder_backbone, use_kluber_scaling)
        self.condition_dim = condition_dim

    def forward(self, x, condition):
        # Expand condition to match x's sequence shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(2).expand(-1, -1, x.shape[2])  # (B, C) -> (B, C, L)
        x_cond = torch.cat([x, condition], dim=1)  # (B, F+C, L)

        mean, log_var = self.encoder(x_cond)
        z_unscaled = self.reparameterize(mean, log_var)

        z_to_decode = z_unscaled * self.kluber_scaling_factor if self.use_kluber_scaling else z_unscaled

        if condition.dim() == 2:
            condition_z = condition.unsqueeze(2).expand(-1, -1, z_to_decode.shape[2])  # (B, C, Lz)
        else:
            condition_z = condition
        z_cond = torch.cat([z_to_decode, condition_z], dim=1)  # (B, latent_dim+C, Lz)

        x_reconstructed = self.decoder(z_cond)
        return x_reconstructed, mean, log_var

    def predict(self, x_in, condition):
        self.eval()
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.FloatTensor(x_in)
        if not isinstance(condition, torch.Tensor):
            condition = torch.FloatTensor(condition)

        x_in = x_in.to(next(self.parameters()).device)
        condition = condition.to(x_in.device)

        if condition.dim() == 2:
            condition = condition.unsqueeze(2).expand(-1, -1, x_in.shape[2])
        x_cond = torch.cat([x_in, condition], dim=1)

        mean, _ = self.encoder(x_cond)
        z_to_decode = mean * self.kluber_scaling_factor if self.use_kluber_scaling else mean

        if condition.dim() == 2:
            condition_z = condition.unsqueeze(2).expand(-1, -1, z_to_decode.shape[2])
        else:
            condition_z = condition
        z_cond = torch.cat([z_to_decode, condition_z], dim=1)

        x_reconstructed = self.decoder(z_cond)
        self.train()
        return x_reconstructed.cpu().numpy()

class CrossAttentionConditionalDecoder(nn.Module):
    def __init__(self, decoder_backbone: nn.Module, context_dim: int, latent_dim: int, heads: int = 4):
        super().__init__()
        self.cross_attn = CrossAttention(n_heads=heads, d_embed=latent_dim, d_cross=context_dim)
        self.decoder = decoder_backbone

    def forward(self, z, context):
        # context: (B, context_len, context_dim)
        # z: (B, latent_len, latent_dim)
        z_attended = self.cross_attn(z, context)  # Apply cross-attention
        return self.decoder(z_attended)

class CVAE_Decoder_Wrapper(VAE_Decoder_Wrapper):

    def forward(self, z, context):
        # z: Latent variable, e.g., (Batch_Size, Seq_Len_Latent, Latent_Features_Dim)
        if self.apply_input_scaling:
            z = z / self.scaling_factor # Unscale if z was scaled before passing to decoder

        reconstructed_x = self.backbone_model(z, context)
        return reconstructed_x


class ConditionalVAE_CrossAttention(VariationalAutoencoder):
    model_name = "ConditionalVAE_CrossAttention"

    def __init__(
        self,
        input_feature_dim: int,
        sequence_len: int,
        latent_feature_dim: int,
        encoder_backbone: nn.Module,
        decoder_backbone: nn.Module,
        context_dim: int,
        use_kluber_scaling: bool = True,
        attn_heads: int = 4
    ):
        # decoder is wrapped in cross-attention
        decoder_with_attn = CrossAttentionConditionalDecoder(
            decoder_backbone, context_dim, latent_feature_dim, heads=attn_heads
        )

        super().__init__(input_feature_dim, sequence_len, latent_feature_dim,
                         encoder_backbone, decoder_with_attn, use_kluber_scaling)
        self.context_dim = context_dim
        self.decoder = CVAE_Decoder_Wrapper(decoder_with_attn)

    def forward(self, x, context):
        mean, log_var = self.encoder(x)
        z_unscaled = self.reparameterize(mean, log_var)

        z_to_decode = z_unscaled * self.kluber_scaling_factor if self.use_kluber_scaling else z_unscaled
        x_reconstructed = self.decoder(z_to_decode, context)
        return x_reconstructed, mean, log_var

    def predict(self, x_in, context):
        self.eval()
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.FloatTensor(x_in)
        if not isinstance(context, torch.Tensor):
            context = torch.FloatTensor(context)

        x_in = x_in.to(next(self.parameters()).device)
        context = context.to(x_in.device)

        mean, _ = self.encoder(x_in)
        z_to_decode = mean * self.kluber_scaling_factor if self.use_kluber_scaling else mean

        x_reconstructed = self.decoder(z_to_decode, context)
        self.train()
        return x_reconstructed.cpu().numpy()


