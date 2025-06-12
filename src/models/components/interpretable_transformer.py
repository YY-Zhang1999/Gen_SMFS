import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
from diffusion_transformer import DiT1D

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(DiT1D):  # Inherits from DiT1D
    def __init__(
            self,
            # Parameters for Interpretable Transformer
            n_feat,  # Input feature dim (DiT1D in_channels)
            n_channel,  # Sequence length (DiT1D seq_len)
            n_layer_enc=5,
            n_layer_dec=14,
            n_embd=512,  # Embedding dim (DiT1D hidden_size)
            n_heads=8,  # (DiT1D num_heads)
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            max_len=2048,  # Max sequence length for pos encoding
            conv_params=None,  # For Conv_MLP, (kernel_size, padding)

            # Parameters for DiT1D base class
            # Map some interpretable params to DiT1D params
            # n_channel -> seq_len for DiT1D
            # n_feat    -> in_channels for DiT1D
            # n_embd    -> hidden_size for DiT1D
            # n_heads   -> num_heads for DiT1D
            # depth for DiT1D (e.g., average of enc/dec, or a new param)
            dit_depth=12,  # Default or can be passed
            dit_patch_size=4,  # Default or can be passed
            dit_mlp_ratio=4.0,
            dit_class_dropout_prob=0.1,
            dit_num_classes=0,  # Default to no class conditioning for DiT part unless specified
            dit_learn_sigma=True,
            **kwargs  # Absorb any other DiT1D or interpretable_transformer specific args
    ):
        # Initialize the DiT1D part
        super().__init__(
            seq_len=n_channel,  # Map n_channel to DiT1D's seq_len
            patch_size=dit_patch_size,
            in_channels=n_feat,  # Map n_feat to DiT1D's in_channels
            hidden_size=n_embd,  # Map n_embd to DiT1D's hidden_size
            depth=dit_depth,
            num_heads=n_heads,
            mlp_ratio=dit_mlp_ratio,
            class_dropout_prob=dit_class_dropout_prob,
            num_classes=dit_num_classes,
            learn_sigma=dit_learn_sigma
        )
        # Note: Now self will have DiT1D attributes like self.x_embedder, self.t_embedder, etc.

        # Initialize Interpretable Transformer specific components
        # These might override or coexist with DiT1D components
        self.n_feat = n_feat
        self.n_channel = n_channel  # Sequence length for interpretable part
        self.n_embd = n_embd  # Internal embedding dimension

        # Embedding layer for interpretable transformer
        # Input (B, n_feat, n_channel) -> Output (B, n_channel, n_embd)
        self.it_emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop, )
        # Inverse projection for interpretable transformer
        # Input (B, n_channel, n_embd) -> Output (B, n_feat, n_channel)
        self.it_inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop,
                                   )  # Transpose happens inside Conv_MLP now

        # These conv_params are for combine_s, not for the main emb/inverse
        kernel_size, padding = (5, 2)  # Default if not Fedformer-style detail
        if conv_params is not None and conv_params[0] is not None:
            kernel_size, padding = conv_params

        # Combine layers for seasonality and mean components
        # combine_s input (B, D_embd, L_seq=n_channel), output (B, n_feat, n_channel)
        self.it_combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                      padding_mode='circular', bias=False)
        # combine_m input (B, n_layer_dec, n_feat_target_dim), output (B, 1, n_feat_target_dim)
        self.it_combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                      padding_mode='circular', bias=False)

        # Interpretable Transformer's Encoder and Decoder
        self.it_encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times,
                                  block_activate)
        self.it_pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.it_decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop,
                                  mlp_hidden_times,
                                  block_activate, condition_dim=n_embd)
        self.it_pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        # The DiT1D's t_embedder and y_embedder are inherited.
        # The Interpretable Transformer needs a way to get timestep and label embeddings.
        # It passes `t` (timestep_emb) and `label_emb` to its EncoderBlocks and DecoderBlocks.
        # We can reuse self.t_embedder and self.y_embedder from DiT1D.

    def forward(self, input_data, time_steps, padding_masks=None, class_labels=None, return_res=False):
        # `input_data`: (B, N_FEAT, N_CHANNEL) - this is (B, C_in, L_seq) for DiT1D
        # `time_steps`: (B,) tensor of diffusion timesteps (scalar for each batch item)
        # `class_labels`: (B,) tensor of class labels (optional)

        # 1. Get timestep and label embeddings using DiT1D's embedders
        # These embeddings are (B, hidden_size) or (B, n_embd)
        t_emb = self.t_embedder(time_steps)  # Inherited from DiT1D

        y_emb = None
        if self.y_embedder is not None and class_labels is not None:
            y_emb = self.y_embedder(class_labels, self.training)  # Inherited

        # The interpretable Transformer's Encoder/Decoder uses AdaLayerNorm which
        # expects conditioning_signal_t and label_emb.
        # DiTBlock combines t_emb and y_emb into a single 'c'.
        # We need to decide how to pass these to it_encoder/it_decoder.
        # Let's assume AdaLayerNorm in model_utils is adapted to take t_emb and optional y_emb.
        # The placeholder AdaLayerNorm takes conditioning_signal and label_emb separately.
        # Let's make the combined DiT-style conditioning signal 'c' if y_emb exists.
        # Or pass t_emb and y_emb separately to Encoder/Decoder which then pass to AdaLayerNorm.

        # Let's pass t_emb and y_emb separately as expected by the interpretable Encoder/Decoder blocks

        # 2. Embed input for the interpretable transformer
        # it_emb expects (B, n_feat, n_channel) and outputs (B, n_channel, n_embd)
        emb = self.it_emb(input_data)  # (B, n_channel, n_embd)

        # 3. Interpretable Encoder pass
        inp_enc = self.it_pos_enc(emb)  # (B, n_channel, n_embd)
        # it_encoder.forward expects (input_seq, t_emb, padding_masks, label_emb)
        enc_cond = self.it_encoder(inp_enc, t_emb, padding_masks=padding_masks,
                                   label_emb=y_emb)  # (B, n_channel, n_embd)

        # 4. Interpretable Decoder pass
        # inp_dec is based on the same initial embedding
        inp_dec = self.it_pos_dec(emb)  # (B, n_channel, n_embd)
        # it_decoder.forward expects (x, t_emb, enc_output, padding_masks, label_emb)
        output, mean_components, trend_components, season_components = \
            self.it_decoder(inp_dec, t_emb, enc_cond, padding_masks=padding_masks, label_emb=y_emb)
        # output: (B, n_channel_len, n_embd_feat)
        # mean_components: (B, n_layer_dec, n_feat_out)
        # trend_components: (B, n_channel_len, n_feat_out)
        # season_components: (B, n_channel_len, n_embd_feat)

        # 5. Final projections and combinations for interpretable output
        # Inverse projection of decoder output (residual part)
        # it_inverse expects (B, n_channel, n_embd) input if Conv_MLP, transposing if needed
        # output is (B, n_channel, n_embd). Need to ensure Conv_MLP handles this.
        # If Conv_MLP's self.is_conv_mlp=True, it expects (B, D_in, L_seq) = (B, n_embd, n_channel)
        res = self.it_inverse(output.transpose(1,
                                               2) if self.it_inverse.is_conv_mlp else output)  # res is (B, n_channel, n_feat) if Conv_MLP is fixed
        if self.it_inverse.is_conv_mlp:  # it_inverse output is (B,L,D_out)=(B,n_channel,n_feat)
            pass  # Shape is already (B, n_channel, n_feat)
        else:  # if simple MLP, output might be (B, n_channel, n_feat) if input was (B,n_channel,n_embd)
            pass

        # Reshape/transpose `res` if necessary to be (B, n_feat, n_channel) for combining
        if res.shape[1] == self.n_channel and res.shape[2] == self.n_feat:
            res = res.transpose(1, 2)  # to (B, n_feat, n_channel)
        elif res.shape[1] == self.n_feat and res.shape[2] == self.n_channel:
            pass  # already (B, n_feat, n_channel)
        else:
            raise ValueError(f"Shape of res {res.shape} is unexpected after it_inverse.")

        res_m = torch.mean(res, dim=2, keepdim=True)  # Mean over n_channel (sequence length) -> (B, n_feat, 1)

        # Seasonality error
        # it_combine_s expects (B, D_embd, L_seq=n_channel)
        # season_components is (B, n_channel_len, n_embd_feat)
        season_error = self.it_combine_s(season_components.transpose(1, 2))  # Output (B, n_feat, n_channel)
        season_error = season_error + res - res_m  # Adding residual part

        # Trend combination
        # it_combine_m expects (B, n_layer_dec, n_feat_target_dim)
        # mean_components is (B, n_layer_dec, n_feat_out)
        # trend_components is (B, n_channel_len, n_feat_out)

        # Ensure mean_components is (B, n_layer_dec, n_feat)
        combined_mean = self.it_combine_m(mean_components)  # (B, 1, n_feat)

        # Ensure trend_components is (B, n_feat, n_channel) for addition with res_m
        if trend_components.shape[1] == self.n_channel and trend_components.shape[2] == self.n_feat:
            trend_final = trend_components.transpose(1, 2)  # (B, n_feat, n_channel)
        elif trend_components.shape[1] == self.n_feat and trend_components.shape[2] == self.n_channel:
            trend_final = trend_components  # (B, n_feat, n_channel)
        else:
            raise ValueError(f"Shape of trend_components {trend_components.shape} is unexpected.")

        trend_output = combined_mean.transpose(1, 2) + res_m + trend_final  # (B, n_feat, n_channel)

        if return_res:  # Return decomposed components
            # Ensure shapes are (B, n_feat, n_channel)
            return trend_output, season_error, (res - res_m).transpose(1, 2) if (res - res_m).shape[
                                                                                    1] == self.n_channel else (
                        res - res_m)

        # The final output of the interpretable transformer is typically trend + season (or error)
        final_prediction = trend_output + season_error  # (B, n_feat, n_channel)
        return final_prediction


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters for the Interpretable Transformer (now inheriting DiT1D)
    bs = 4
    n_feat_test = 3  # Number of input features (e.g., 3 channels for a multivariate series)
    n_channel_test = 64  # Sequence length
    n_embd_test = 128  # Embedding dimension

    # DiT specific params (can be tuned)
    dit_patch_size_test = 8  # n_channel_test must be divisible by this
    dit_depth_test = 6
    dit_num_classes_test = 0  # No class conditioning for DiT part in this example

    interpretable_dit_model = Transformer(
        n_feat=n_feat_test,
        n_channel=n_channel_test,
        n_embd=n_embd_test,
        n_heads=4,
        n_layer_enc=3,
        n_layer_dec=3,
        dit_patch_size=dit_patch_size_test,
        dit_depth=dit_depth_test,
        dit_num_classes=dit_num_classes_test,
        max_len=n_channel_test  # For positional encodings
    ).to(device)

    print(f"Interpretable Transformer (inheriting DiT1D) instantiated: {type(interpretable_dit_model)}")
    # print(interpretable_dit_model) # Print model structure

    # Dummy input for the interpretable Transformer's forward method
    dummy_it_input = torch.randn(bs, n_feat_test, n_channel_test).to(device)
    dummy_it_timesteps = torch.randint(0, 1000, (bs,), device=device)  # Timesteps for DiT embedders
    # dummy_it_labels = None (if dit_num_classes=0)
    # if dit_num_classes_test > 0:
    #     dummy_it_labels = torch.randint(0, dit_num_classes_test, (bs,), device=device)

    print(f"\nTesting Interpretable Transformer's forward method:")
    print(f"Input data shape: {dummy_it_input.shape}")  # (B, n_feat, n_channel)
    print(f"Timesteps shape: {dummy_it_timesteps.shape}")  # (B,)

    # Call the forward method
    # If class conditioning is enabled in DiT part, pass labels. Otherwise, it handles None.
    output_prediction = interpretable_dit_model(dummy_it_input, dummy_it_timesteps, class_labels=None)
    print(f"Output prediction shape: {output_prediction.shape}")  # Expected (B, n_feat, n_channel)
    assert output_prediction.shape == (bs, n_feat_test, n_channel_test)

    output_decomposed = interpretable_dit_model(dummy_it_input, dummy_it_timesteps, class_labels=None, return_res=True)
    print(
        f"Output decomposed shapes: trend={output_decomposed[0].shape}, season={output_decomposed[1].shape}, res={output_decomposed[2].shape}")
    assert output_decomposed[0].shape == (bs, n_feat_test, n_channel_test)
    assert output_decomposed[1].shape == (bs, n_feat_test, n_channel_test)
    assert output_decomposed[2].shape == (bs, n_feat_test, n_channel_test)

    print("\nTest with DiT1D's inherited forward method (for diffusion-like behavior):")
    # This would require the input to be structured for DiT's patching,
    # and the output would be the noise prediction or denoised sample.
    # The current `interpretable_dit_model.forward` IS the interpretable one.
    # To call DiT1D's original forward, one might need to call it explicitly if not overridden,
    # or if the interpretable `Transformer.forward` was named differently.
    # Since `Transformer.forward` overrides `DiT1D.forward`, to test DiT's logic,
    # you'd call `super(Transformer, self).forward(...)` or use a DiT1D instance directly.

    # Example: if you wanted to see DiT's patching and block processing (conceptual)
    # This assumes `input_data` (B, n_feat, n_channel) is compatible with DiT1D's x_embedder input.
    try:
        # Manually call DiT1D's components to simulate its path (if `Transformer.forward` didn't exist)
        # Note: This is illustrative. `interpretable_dit_model.forward` *is* the one defined in Transformer class.

        # 1. DiT-style Patch Embedding (inherited)
        # DiT1D.x_embedder expects (B, C_in=n_feat, L_seq=n_channel)
        patched_x = interpretable_dit_model.x_embedder(dummy_it_input) + interpretable_dit_model.pos_embed
        print(f"DiT Patched x shape: {patched_x.shape}")  # (B, num_patches, n_embd)

        # 2. DiT Timestep and Label Embedding (inherited)
        t_emb_dit = interpretable_dit_model.t_embedder(dummy_it_timesteps)
        # y_emb_dit = None
        # if interpretable_dit_model.y_embedder is not None and dummy_it_labels is not None:
        #     y_emb_dit = interpretable_dit_model.y_embedder(dummy_it_labels, interpretable_dit_model.training)

        c_dit = t_emb_dit
        # if y_emb_dit is not None:
        #     c_dit = t_emb_dit + y_emb_dit

        print(f"DiT Conditioning vector 'c_dit' shape: {c_dit.shape}")  # (B, n_embd)

        # 3. DiT Blocks (inherited)
        x_processed_dit = patched_x
        for block in interpretable_dit_model.blocks:  # These are DiTBlocks
            x_processed_dit = block(x_processed_dit, c_dit)
        print(f"DiT Blocks output shape: {x_processed_dit.shape}")  # (B, num_patches, n_embd)

        # 4. DiT Final Layer and Unpatchify (inherited)
        final_out_dit_patched = interpretable_dit_model.final_layer(x_processed_dit, c_dit)
        final_out_dit_unpatched = interpretable_dit_model.unpatchify1D(final_out_dit_patched)
        print(f"DiT Final Unpatched Output shape: {final_out_dit_unpatched.shape}")
        # Expected (B, DiT1D.out_channels, n_channel_test)
        assert final_out_dit_unpatched.shape == (bs, interpretable_dit_model.out_channels, n_channel_test)
        print("Conceptual test of inherited DiT components successful.")

    except Exception as e:
        print(f"Error during conceptual DiT component test: {e}")
        import traceback

        traceback.print_exc()

    print("\nAll tests finished.")