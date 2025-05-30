# src/models/diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Union, List, Tuple

# Import components and encoders
try:
    from .protein_encoder import ProteinEncoder
    from .condition_encoder import ConditionEncoder
    from .components.attention import MultiHeadSelfAttention # If using attention
except ImportError:
    logging.warning("Could not import model components directly. Ensure src.models is in PYTHONPATH.")
    # Define placeholder classes if necessary for testing this file alone
    class ProteinEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, encoding_type='pretrained_embeddings', **kwargs):
            super().__init__()
            self.output_dim = output_dim
            self.fc = nn.Linear(input_dim if isinstance(input_dim, int) else np.prod(input_dim), output_dim) # Basic placeholder
            logging.warning("Using placeholder ProteinEncoder.")
        def forward(self, x):
            if isinstance(x, list): # Handle dummy raw strings list
                 x = torch.randn(len(x), self.fc.in_features, device=self.fc.weight.device) # Dummy tensor
            return self.fc(x.flatten(1)) # Flatten for simple linear
    class ConditionEncoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            logging.warning("Using placeholder ConditionEncoder.")
        def forward(self, x): return self.fc(x)
    class MultiHeadSelfAttention(nn.Module):
         def __init__(self, embed_dim, num_heads):
             super().__init__()
             self.identity = nn.Identity() # Placeholder
             logging.warning("Using placeholder MultiHeadSelfAttention.")
         def forward(self, x, mask=None): return self.identity(x)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message')

class PositionalEncoding(nn.Module):
    """ Basic sinusoidal positional encoding """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class TimeEmbedding(nn.Module):
    """ Embeds the diffusion timestep t """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # Example expansion
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # Sinusoidal positional embedding for the timestep itself
        # Borrowing from positional encoding concept for time
        self.embed_dim = embed_dim
        inv_freq = 1. / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape (batch_size,) or (batch_size, 1) representing the timestep
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1) # (batch_size, 1)

        # Apply sinusoidal embedding
        sinusoid_inp = torch.outer(t.flatten(), self.inv_freq) # (batch_size, embed_dim // 2)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1) # (batch_size, embed_dim)

        # Pass through MLP
        emb = self.mlp(emb)
        return emb


class ConditionalDenoisingModel(nn.Module):
    """
    The neural network that predicts the noise (or denoised data) in a Diffusion Model.
    It is conditioned on protein sequence embeddings, condition embeddings, and the timestep t.

    This is a placeholder structure. You will need to replace `nn.Sequential`
    with a more sophisticated architecture like a U-Net variant, Transformer blocks,
    or a sequence of residual blocks with attention and conditioning.
    """
    def __init__(self,
                 fe_curve_length: int,
                 fe_curve_channels: int, # Typically 1 for F-E curve (force)
                 protein_embed_dim: int,
                 condition_embed_dim: int,
                 time_embed_dim: int,
                 model_channels: int, # Base number of channels in the model
                 num_attention_heads: int = 8, # For attention layers if used
                 num_layers: int = 6 # Number of processing layers/blocks
                 ):
        """
        Initializes the ConditionalDenoisingModel.

        Args:
            fe_curve_length (int): The fixed length of the input F-E curve vector.
            fe_curve_channels (int): The number of channels in the F-E curve data (e.g., 1 for force).
            protein_embed_dim (int): Dimensionality of the protein embedding.
            condition_embed_dim (int): Dimensionality of the condition embedding.
            time_embed_dim (int): Dimensionality of the time embedding.
            model_channels (int): Base number of channels/features within the model layers.
            num_attention_heads (int): Number of heads for attention layers if used.
            num_layers (int): Number of processing layers/blocks in the denoising network.
        """
        super().__init__()
        self.fe_curve_length = fe_curve_length
        self.fe_curve_channels = fe_curve_channels
        self.model_channels = model_channels
        self.num_layers = num_layers

        # Input layer to project the noisy F-E curve to model_channels
        self.input_proj = nn.Linear(fe_curve_channels, model_channels)

        # Embedding for the time step
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_channels * 4), # Match time embedding size
            nn.ReLU(),
            nn.Linear(model_channels * 4, model_channels)
        )

        # Combine protein and condition embeddings
        # A simple concatenation and linear layer is one way,
        # or they can be used for conditional normalization/modulation later.
        self.conditional_embedding_dim = protein_embed_dim + condition_embed_dim
        self.conditional_mlp = nn.Sequential(
            nn.Linear(self.conditional_embedding_dim, model_channels * 4),
            nn.ReLU(),
            nn.Linear(model_channels * 4, model_channels) # Output matches model_channels
        )

        # --- Core Denoising Network (Placeholder) ---
        # This is where the main architecture (U-Net, Transformer, ResNet blocks) goes.
        # This example uses simple linear layers and residual connections with attention.

        # Project combined conditioning into modulation parameters (e.g., for FiLM, AdaIN, etc.)
        # Or simply concatenate the conditional embedding to the features at each layer.
        # Let's use concatenation for simplicity in this placeholder.
        total_input_features_per_pos = model_channels + model_channels # Model channels + Time embedding (broadcast) + Conditional embedding (broadcast)
        # Note: Time and Conditional embeddings need to be broadcast correctly across the sequence length.
        # A common way is to add them to the features after the initial projection.

        self.layers = nn.ModuleList()
        # Example: Sequence of residual blocks with attention and conditioning
        for _ in range(num_layers):
             # Basic structure: Add time and conditional embeddings, apply core block, add residual connection
             layer = nn.Sequential(
                 # Example: Simple block structure (you'd replace with ResNet block, Transformer layer, etc.)
                 nn.Linear(model_channels + model_channels, model_channels * 2), # Combine features + time/cond
                 nn.ReLU(),
                 nn.Linear(model_channels * 2, model_channels),
                 # Optional: Add Attention after the linear layers
                 # MultiHeadSelfAttention(model_channels, num_attention_heads),
                 nn.ReLU() # Example activation
             )
             self.layers.append(layer)

        # Add optional attention layers within the sequence processing if not using a Transformer variant already
        # For a simple sequence of blocks, placing attention here can work.
        # self.attention_layers = nn.ModuleList([
        #     MultiHeadSelfAttention(model_channels, num_attention_heads)
        #     for _ in range(num_attention_heads) # Example: one attention per head count
        # ])


        # --- End Core Denoising Network Placeholder ---

        # Output layer to project back to the shape of the F-E curve
        self.output_proj = nn.Linear(model_channels, fe_curve_channels)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                protein_embedding: torch.Tensor, condition_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConditionalDenoisingModel.

        Args:
            x_t (torch.Tensor): The noisy F-E curve at timestep t. Shape (batch_size, fe_curve_length, fe_curve_channels).
            t (torch.Tensor): The diffusion timestep. Shape (batch_size,) or (batch_size, 1).
            protein_embedding (torch.Tensor): The encoded protein embedding. Shape (batch_size, protein_embed_dim).
            condition_embedding (torch.Tensor): The encoded condition embedding. Shape (batch_size, condition_embed_dim).

        Returns:
            torch.Tensor: The predicted noise epsilon or the predicted denoised data x_0,
                          depending on the diffusion model formulation.
                          Shape (batch_size, fe_curve_length, fe_curve_channels).
        """
        batch_size, seq_len, num_channels = x_t.size()
        if seq_len != self.fe_curve_length or num_channels != self.fe_curve_channels:
             raise ValueError(f"Input x_t shape mismatch. Expected ({batch_size}, {self.fe_curve_length}, {self.fe_curve_channels}), got {x_t.shape}")


        # Project noisy input
        h = self.input_proj(x_t) # Shape: (batch_size, fe_curve_length, model_channels)

        # Embed time and conditions
        time_emb = self.time_mlp(t) # Shape: (batch_size, model_channels)
        conditional_emb = self.conditional_mlp(torch.cat([protein_embedding, condition_embedding], dim=-1)) # Shape: (batch_size, model_channels)

        # Combine time and conditional embeddings and broadcast over sequence length
        # Example: Add the combined embedding to the features at each position
        # (batch_size, 1, model_channels) + (batch_size, 1, model_channels) -> (batch_size, 1, model_channels)
        # Then broadcast to (batch_size, fe_curve_length, model_channels)
        combined_cond_emb = time_emb.unsqueeze(1) + conditional_emb.unsqueeze(1) # Shape: (batch_size, 1, model_channels)

        # --- Core Denoising Network Forward ---
        x = h # Start with the projected noisy input
        for i, layer in enumerate(self.layers):
             # Add the combined conditioning to the current features
             x_conditioned = x + combined_cond_emb # Shape: (batch_size, fe_curve_length, model_channels)

             # Apply the layer block (e.g., Linear, Conv, etc.)
             # Note: The current simple layer expects (batch_size, fe_curve_length, model_channels + model_channels)
             # Reshape or adapt layer structure
             # Let's adapt the layer to take (batch_size, fe_curve_length, model_channels) and conditional_emb
             # A common pattern is to use conditional normalization (e.g., AdaIN, FiLM)
             # Or simply concatenate features and broadcasted conditioning
             # Let's use concatenation + Linear for the placeholder
             batch_size, seq_len, current_channels = x.size()
             # Expand conditional embedding to match sequence length
             combined_cond_emb_expanded = combined_cond_emb.expand(-1, seq_len, -1) # (batch_size, fe_curve_length, model_channels)
             x_concat = torch.cat([x, combined_cond_emb_expanded], dim=-1) # (batch_size, fe_curve_length, model_channels * 2)

             # Apply the layer block
             # Assuming the layer input is (batch_size, fe_curve_length, model_channels * 2)
             # and output is (batch_size, fe_curve_length, model_channels) as in the placeholder
             residual = x # For residual connection
             x = layer(x_concat) # Apply the layer

             # Add residual connection (if applicable based on layer type)
             # This simple residual adds input of the block to output
             if x.shape == residual.shape: # Check if shapes match for adding
                 x = x + residual
             else:
                 logging.warning(f"Residual connection skipped in layer {i} due to shape mismatch: {x.shape} vs {residual.shape}")

             # Optional: Apply attention
             # if hasattr(self, 'attention_layers') and i < len(self.attention_layers):
             #      x = self.attention_layers[i](x)


        # --- End Core Denoising Network Forward ---


        # Output projection
        output = self.output_proj(x) # Shape: (batch_size, fe_curve_length, fe_curve_channels)

        return output


class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model for generating Force-Extension curves.
    Encapsulates the forward diffusion process and the conditional denoising model.
    """
    def __init__(self,
                 fe_curve_length: int,
                 fe_curve_channels: int = 1, # Force is typically 1 channel
                 protein_input_dim: Union[int, Tuple[int, int], None] = None, # Input dim for protein encoder
                 protein_embed_dim: int = 128, # Output dim for protein encoder
                 protein_encoding_type: str = 'pretrained_embeddings', # Type for protein encoder
                 protein_plm_name: str = None, # PLM name if protein_encoding_type is 'raw'
                 protein_plm_embed_dim: int = None, # PLM output dim if protein_encoding_type is 'raw'
                 protein_freeze_plm: bool = True, # Freeze PLM?
                 condition_input_dim: int = None, # Input dim for condition encoder
                 condition_embed_dim: int = 64, # Output dim for condition encoder
                 time_embed_dim: int = 128, # Dimension for time embedding
                 model_channels: int = 256, # Base channels for denoising model
                 num_diffusion_steps: int = 1000,
                 beta_schedule: str = 'linear' # 'linear', 'cosine', etc.
                 ):
        """
        Initializes the ConditionalDiffusionModel.

        Args:
            fe_curve_length (int): The fixed length of the F-E curve vector.
            fe_curve_channels (int): Number of channels in the F-E curve (default 1 for force).
            protein_input_dim, protein_embed_dim, protein_encoding_type, etc.: Parameters for ProteinEncoder.
            condition_input_dim, condition_embed_dim: Parameters for ConditionEncoder.
            time_embed_dim (int): Dimension for time embedding.
            model_channels (int): Base number of channels in the denoising network.
            num_diffusion_steps (int): Number of diffusion steps (T).
            beta_schedule (str): Type of variance schedule ('linear' or 'cosine').
        """
        super().__init__()
        self.fe_curve_length = fe_curve_length
        self.fe_curve_channels = fe_curve_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = beta_schedule

        # --- Define Variance Schedule (betas) ---
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models" (Nichol et al. 2021)
            s = 0.008
            ts = torch.linspace(0, num_diffusion_steps, num_diffusion_steps + 1)
            alphas_bar = torch.cos(((ts / num_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5)**2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999) # Clip to prevent problems

        # Pre-compute alpha, alpha_bar, and related terms
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        # For reverse process (denoising)
        self.posterior_variance = self.betas * (1.0 - self.alphas_bar[:-1]) / (1.0 - self.alphas_bar[1:])
        self.posterior_variance = torch.cat([self.posterior_variance[0].unsqueeze(0), self.posterior_variance]) # Pad the first element


        # --- Encoders ---
        if protein_input_dim is None and protein_encoding_type != 'raw':
             logging.error("protein_input_dim must be provided for encoding types other than 'raw'.")
             raise ValueError("protein_input_dim missing.")
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_input_dim,
            output_dim=protein_embed_dim,
            encoding_type=protein_encoding_type,
            plm_model_name=protein_plm_name,
            plm_embedding_dim=protein_plm_embed_dim,
            freeze_plm=protein_freeze_plm
        )

        if condition_input_dim is None:
             logging.error("condition_input_dim must be provided for ConditionEncoder.")
             raise ValueError("condition_input_dim missing.")
        self.condition_encoder = ConditionEncoder(
            input_dim=condition_input_dim,
            output_dim=condition_embed_dim
        )

        # --- Time Embedding ---
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim)

        # --- Denoising Model (the neural network) ---
        self.denoising_model = ConditionalDenoisingModel(
            fe_curve_length=fe_curve_length,
            fe_curve_channels=fe_curve_channels,
            protein_embed_dim=protein_embed_dim,
            condition_embed_dim=condition_embed_dim,
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            # num_attention_heads=... # Pass attention heads if using attention in DenoisingModel
            # num_layers=... # Pass number of layers
        )

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        The forward diffusion process: adds noise to the data x_0 at timestep t.

        Args:
            x_0 (torch.Tensor): The original (clean) data. Shape (batch_size, fe_curve_length, fe_curve_channels).
            t (torch.Tensor): The diffusion timestep. Shape (batch_size,).
            noise (torch.Tensor, optional): Optional pre-sampled noise. Shape same as x_0.
                                           Defaults to None, in which case noise is sampled.

        Returns:
            torch.Tensor: The noisy data x_t at timestep t. Shape (batch_size, fe_curve_length, fe_curve_channels).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get alpha_bar_t for each sample in the batch
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1) # Shape (batch_size, 1, 1)
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1) # Shape (batch_size, 1, 1)

        # Apply the diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise

        return x_t, noise # Return both x_t and the noise that was added (useful for training loss)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Samples random diffusion timesteps for a batch.

        Args:
            batch_size (int): The number of samples in the batch.
            device (torch.device): The device to place the sampled timesteps on.

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) with random timesteps (integers from 0 to num_diffusion_steps-1).
        """
        return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

    def forward(self, x_0: torch.Tensor, sequence_data: Union[torch.Tensor, List[str]],
                conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training the Diffusion Model.
        Performs one step of the forward diffusion process and runs the denoising model.

        Args:
            x_0 (torch.Tensor): The original (clean) F-E curve. Shape (batch_size, fe_curve_length, fe_curve_channels).
            sequence_data (Union[torch.Tensor, List[str]]): Protein sequence data from the dataset batch.
                                                          Shape depends on protein_encoding_type.
            conditions (torch.Tensor): Experimental condition data from the dataset batch.
                                       Shape (batch_size, condition_input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_t: The noisy F-E curve at a random timestep.
                - t: The random timestep.
                - predicted_noise: The noise predicted by the denoising model.
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # 1. Sample random timestep
        t = self.sample_timesteps(batch_size, device)

        # 2. Add noise to x_0 at timestep t
        x_t, noise = self.forward_diffusion(x_0, t)

        # 3. Encode protein sequence and conditions
        protein_emb = self.protein_encoder(sequence_data)
        condition_emb = self.condition_encoder(conditions)
        time_emb = self.time_embedding(t)

        # 4. Run the denoising model to predict noise
        # Note: The denoising_model forward expects time_emb and conditional embeddings
        # separately or combined depending on its internal implementation.
        # The placeholder ConditionalDenoisingModel expects protein_embedding and condition_embedding
        # Let's update its signature or adapt the call here.
        # Based on the placeholder's __init__, it takes protein_embed, condition_embed, time_embed
        predicted_noise = self.denoising_model(x_t, time_emb, protein_emb, condition_emb)

        return x_t, t, predicted_noise, noise # Return added noise for training loss calculation


    @torch.no_grad() # Inference should not calculate gradients
    def generate(self, sequence_data: Union[torch.Tensor, List[str]],
                 conditions: torch.Tensor, num_samples: int = 1, device: torch.device = 'cpu') -> torch.Tensor:
        """
        Generates new Force-Extension curves using the trained Diffusion Model.

        Args:
            sequence_data (Union[torch.Tensor, List[str]]): Protein sequence data for generation.
                                                          Shape depends on protein_encoding_type.
                                                          Should represent the batch of samples to generate for.
                                                          Batch size determined by sequence_data/conditions size.
            conditions (torch.Tensor): Experimental condition data for generation.
                                       Shape (num_samples, condition_input_dim) or (batch_size, condition_input_dim).
                                       Number of rows determines the batch size for generation.
            num_samples (int): Number of F-E curves to generate for each set of sequence/condition.
                               Note: Current implementation generates one curve per sequence/condition in the batch.
                               To generate multiple for the same inputs, you'd duplicate inputs.
            device (torch.device): The device to perform generation on.

        Returns:
            torch.Tensor: Generated F-E curves. Shape (num_samples, fe_curve_length, fe_curve_channels).
        """
        self.eval() # Set model to evaluation mode

        # Ensure sequence_data and conditions batch sizes match
        batch_size = len(sequence_data) if isinstance(sequence_data, list) else sequence_data.size(0)
        if batch_size != conditions.size(0):
             raise ValueError("Batch sizes of sequence_data and conditions must match for generation.")
        if num_samples != batch_size:
             logging.warning(f"generate received num_samples={num_samples} but batch size from inputs is {batch_size}. "
                             f"Generating {batch_size} samples, one for each input in the batch.")


        # Encode protein sequence and conditions
        protein_emb = self.protein_encoder(sequence_data).to(device)
        condition_emb = self.condition_encoder(conditions).to(device)


        # Start with random noise (x_T)
        x_t = torch.randn(batch_size, self.fe_curve_length, self.fe_curve_channels, device=device)

        logging.info(f"Starting reverse diffusion process for {batch_size} samples...")

        # Iterate through timesteps in reverse (from T to 1)
        # Use torch.flip to get timesteps in descending order
        timesteps = torch.arange(self.num_diffusion_steps - 1, -1, -1, device=device)

        for i in timesteps:
            t = torch.tensor([i] * batch_size, device=device) # Current timestep for the batch
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_bar[i]
            sqrt_alpha_bar_t = self.sqrt_alphas_bar[i]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]

            # Predict noise at timestep t
            # The denoising_model expects time_emb and conditional embeddings
            time_emb_t = self.time_embedding(t).to(device)

            # Pass protein and condition embeddings (already calculated and on device)
            predicted_noise = self.denoising_model(x_t, time_emb_t, protein_emb, condition_emb)

            # Use the predicted noise to estimate x_0 or the mean of the posterior distribution
            # Using the "predict epsilon" formulation: x_0_hat = (x_t - sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
            x_0_hat = (x_t - sqrt_one_minus_alpha_bar_t.view(-1, 1, 1) * predicted_noise) / sqrt_alpha_bar_t.view(-1, 1, 1)

            # Clip the predicted x_0 to the expected data range (optional but can help stability)
            # You might need to determine the typical range of your normalized F-E curves.
            # Example clipping to [-1, 1] if normalized accordingly:
            # x_0_hat = torch.clamp(x_0_hat, -1., 1.)

            # Compute the mean of the posterior distribution q(x_{t-1} | x_t, x_0)
            # mean = (alpha_t * sqrt(alpha_bar_{t-1}) * x_0_hat + sqrt(1 - alpha_t) * sqrt(1 - alpha_bar_t) * x_t) / (1 - alpha_bar_t) # This formula is complex
            # A simpler reparameterization based on predicting x_0_hat or mean directly is common
            # Let's use the reparameterization from predict epsilon to predict x_{t-1}
            # mean = (x_t - self.betas[i].view(-1, 1, 1) / sqrt_one_minus_alpha_bar_t.view(-1, 1, 1) * predicted_noise) / torch.sqrt(alpha_t).view(-1, 1, 1) # Check this reparam formula

            # A common prediction for the mean of q(x_{t-1} | x_t, x_0) given predicted epsilon is:
            mean = (x_t - self.betas[i].view(-1, 1, 1) * predicted_noise / sqrt_one_minus_alpha_bar_t.view(-1, 1, 1)) / torch.sqrt(alpha_t).view(-1, 1, 1)


            # Sample from the posterior distribution p(x_{t-1} | x_t) which is approximated by q(x_{t-1} | x_t, x_0_hat)
            # x_{t-1} = mean + sqrt(posterior_variance_t) * z, where z ~ N(0, 1)
            if i > 0: # Don't add noise at the last step (t=0 -> x_0)
                variance_t = self.posterior_variance[i]
                # Or use the schedule's beta_t for variance (simpler, less theoretically optimal)
                # variance_t = self.betas[i] # Simpler variance

                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance_t).view(-1, 1, 1) * z
            else:
                x_t = mean # At t=0, the mean is the final denoised sample x_0

        self.train() # Set model back to training mode
        logging.info("Reverse diffusion process complete.")

        return x_t # This is the generated x_0


# Example Usage
if __name__ == "__main__":
    print("--- Testing ConditionalDiffusionModel ---")

    # Define model parameters
    fe_curve_length = 200
    fe_curve_channels = 1 # Force
    protein_embedding_dim = 128
    condition_embedding_dim = 64
    time_embedding_dim = 128
    model_base_channels = 256
    num_diffusion_steps = 100
    batch_size = 8
    num_condition_features = 2 # e.g., pulling speed, temperature

    # Assume protein sequence encoding type is 'pretrained_embeddings' for this test
    protein_input_dim = 1024 # Dimension of the pre-computed embeddings


    # Instantiate the Diffusion Model
    try:
        diffusion_model = ConditionalDiffusionModel(
            fe_curve_length=fe_curve_length,
            fe_curve_channels=fe_curve_channels,
            protein_input_dim=protein_input_dim,
            protein_embed_dim=protein_embedding_dim,
            protein_encoding_type='pretrained_embeddings', # Or 'raw' if you implemented PLM
            # If using 'raw', need to provide plm_model_name and plm_embed_dim
            condition_input_dim=num_condition_features,
            condition_embed_dim=condition_embedding_dim,
            time_embed_dim=time_embedding_dim,
            model_channels=model_base_channels,
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule='linear'
        )
        print("Diffusion model instantiated successfully.")

        # Create dummy input data (simulating a batch from the DataLoader)
        dummy_x0 = torch.randn(batch_size, fe_curve_length, fe_curve_channels) # Clean F-E curves
        # For 'pretrained_embeddings', input is a tensor
        dummy_seq_data = torch.randn(batch_size, protein_input_dim)
        dummy_conditions = torch.randn(batch_size, num_condition_features)

        # --- Test Forward Pass (for training) ---
        print("\n--- Testing forward pass (training) ---")
        x_t, t, predicted_noise, true_noise = diffusion_model(dummy_x0, dummy_seq_data, dummy_conditions)

        print(f"Input x_0 shape: {dummy_x0.shape}")
        print(f"Noisy x_t shape: {x_t.shape}")
        print(f"Sampled timestep t: {t.tolist()}") # Show sampled timesteps
        print(f"Predicted noise shape: {predicted_noise.shape}")
        print(f"True added noise shape: {true_noise.shape}")

        # Check if shapes match as expected for loss calculation (e.g., MSE between predicted_noise and true_noise)
        assert predicted_noise.shape == true_noise.shape
        print("Predicted noise and true noise shapes match.")

        # Test backward pass (should work if gradients are enabled)
        loss = F.mse_loss(predicted_noise, true_noise)
        print(f"Dummy loss: {loss.item():.4f}")
        loss.backward()
        print("Backward pass successful.")


        # --- Test Generation (Inference) ---
        print("\n--- Testing generation (inference) ---")
        num_generate = 4 # Generate 4 samples

        # Create dummy inputs for generation (same format as training inputs)
        dummy_seq_gen = torch.randn(num_generate, protein_input_dim)
        dummy_cond_gen = torch.randn(num_generate, num_condition_features)

        generated_curves = diffusion_model.generate(dummy_seq_gen, dummy_cond_gen, num_samples=num_generate)
        print(f"Generated curves shape: {generated_curves.shape}")

    except ValueError as e:
        print(f"Model initialization error: {e}. Make sure input dimensions match encoding types.")
    except NotImplementedError as e:
        print(f"NotImplementedError during test: {e}. Placeholder logic needs implementation.")
    except Exception as e:
        print(f"An error occurred during Diffusion Model test: {e}")