# Implements VQVAE encoder and decoder model to convert spectrogram to latent space and back to spectrogram.
# Code in part referenced from https://github.com/explainingai-code/StableDiffusion-PyTorch

import torch.nn as nn
import random
import torch
import torchaudio
import torch.functional as F
import torch.nn.functional as NF
import typing
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from .config import VAEConfig
from .pqmf import PQMF


class DownBlock1D(nn.Module):
    r"""
    1D Down-convolutional block with optional attention, adapted for audio.
    Sequence of the following operations for each layer:
    1. Resnet block (two 1D convolutions with a residual connection)
    2. Optional Attention block
    And finally, an optional Downsample operation.

    Input tensor format: (batch, channel, length)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_sample: int,
        num_heads: int,
        num_layers: int,
        attn: bool,
        norm_channels: int,
        activation: typing.Callable[[], nn.Module],
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn

        # First convolution in each ResNet layer
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                activation(),
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        # Second convolution in each ResNet layer
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                activation(),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        if self.attn:
            # Normalization before attention
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            # MultiheadAttention layers
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True, bias=False)
                 for _ in range(num_layers)]
            )

        # Convolution for residual connection (if channels change)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        # Downsampling convolution
        self.down_sample_conv = nn.Conv1d(out_channels, out_channels, kernel_size=self.down_sample*2, stride=self.down_sample, padding=self.down_sample-1) \
            if self.down_sample > 1 else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            resnet_input = out  # Store for residual connection

            # First part of ResNet block
            out = self.resnet_conv_first[i](out)
            # Second part of ResNet block
            out = self.resnet_conv_second[i](out)
            # Add residual connection
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                # Attention block
                batch_size, channels, length = out.shape

                # Normalize before attention
                in_attn = self.attention_norms[i](out)

                # Prepare for MultiheadAttention: (N, C, L) -> (N, L, C)
                in_attn = in_attn.transpose(1, 2)

                # Gradient checkpointing
                if self.use_gradient_checkpointing and self.training:
                    out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn, use_reentrant=False)   # type: ignore
                else:
                    out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)

                # Reshape back: (N, L, C) -> (N, C, L)
                out_attn = out_attn.transpose(1, 2)

                # Add attention output to ResNet output
                out = out + out_attn

        # Optional downsampling
        out = self.down_sample_conv(out)
        return out


class MidBlock1D(nn.Module):
    r"""
    1D Mid-convolutional block with attention, adapted for audio.
    Sequence of the following blocks:
    1. Resnet block
    2. For num_layers:
        a. Attention block
        b. Resnet block
    Input: (B, in_channels, L)
    Output: (B, out_channels, L) (Note: in_channels will be out_channels after the first ResNet block)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        norm_channels: int,
        activation: typing.Callable[[], nn.Module],
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ResNet convolutional layers
        # There are (num_layers + 1) ResNet blocks in total.
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                activation(),
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers + 1)  # One initial ResNet, then one per attention layer
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                activation(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers + 1)
        ])

        # Attention layers (num_layers attention blocks)
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(norm_channels, out_channels)
            for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True, bias=False)
            for _ in range(num_layers)
        ])

        # Residual connection convolutions (for each ResNet block)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x):
        out = x

        # First ResNet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Loop for subsequent Attention and ResNet blocks
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, length = out.shape  # L is the sequence length for 1D

            # Normalize before attention
            in_attn = self.attention_norms[i](out)

            # Prepare for MultiheadAttention: (N, C, L) -> (N, L, C)
            in_attn = in_attn.transpose(1, 2)

            if self.use_gradient_checkpointing and self.training:  # Gradient checkpointing only in training
                out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn, use_reentrant=False)   # type: ignore
            else:
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)

            # Reshape back: (N, L, C) -> (N, C, L)
            out_attn = out_attn.transpose(1, 2)

            # Add attention output
            out = out + out_attn

            # Corresponding ResNet Block
            resnet_input = out
            # We use i + 1 because the first ResNet block (index 0) is already processed
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock1D(nn.Module):
    r"""
    1D Up-convolutional block with optional attention, adapted for audio.
    Sequence of following blocks:
    1. Upsample
    2. For num_layers:
        a. Resnet block
        b. Optional Attention Block

    Input: (B, in_channels, L_in)
    Output: (B, out_channels, L_out) where L_out depends on upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_sample: int,
        num_heads: int,
        num_layers: int,
        attn: bool,
        norm_channels: int,
        activation: typing.Callable[[], nn.Module],
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample_factor = up_sample  # Renamed from self.up_sample to avoid conflict with up_sample arg
        self.attn = attn
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Upsampling Layer
        if not self.up_sample_factor > 1:
            self.up_sample_conv = nn.Identity()
        else:
            self.up_sample_conv = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=self.up_sample_factor, stride=self.up_sample_factor)

        # ResNet convolutional layers
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                activation(),
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                activation(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ])

            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True, bias=False)
                for _ in range(num_layers)
            ])

        # Residual connection convolutions
        self.residual_input_conv = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

    def forward(self, x):
        # Upsample
        # x: (B, C_in, L_in)
        x = self.up_sample_conv(x)
        # x: (B, C_in, L_intermediate)

        out = x
        for i in range(self.num_layers):
            resnet_input = out

            # ResNet Block
            # The first resnet_conv_first[0] expects `in_channels` (channels of `x` after upsampling).
            # Subsequent ones expect `out_channels`.
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Self Attention
            if self.attn:
                batch_size, channels, length = out.shape

                # Normalize before attention
                in_attn = self.attention_norms[i](out)

                # Prepare for MultiheadAttention: (N, C, L) -> (N, L, C)
                in_attn = in_attn.transpose(1, 2)

                if self.use_gradient_checkpointing and self.training:
                    out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn, use_reentrant=False)  # type: ignore
                else:
                    out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)

                # Reshape back: (N, L, C) -> (N, C, L)
                out_attn = out_attn.transpose(1, 2)

                out = out + out_attn
        return out


class Quantizer(nn.Module):
    """Takes in x of shape (B, C=z_channels, L) and returns quantized output and indices

    Returns:
        quant_out: Tensor of shape (B, C, L)
        quantize_losses: dict with 'codebook_loss' and 'commitment_loss'
        min_encoding_indices: Tensor of shape (B, L)
    """

    def __init__(self, codebook_size: int, z_channels: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, z_channels)

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape

        # B, C, L -> B, L, C
        x = x.permute(0, 2, 1)

        # Find nearest embedding/codebook vector
        # dist between (B, L, C) and (1, K, C) -> (B, L, K)
        dist = torch.cdist(x, self.codebook.weight[None, :].repeat((x.size(0), 1, 1)))
        # B, L
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # B*L, C
        quant_out = torch.index_select(self.codebook.weight, 0, min_encoding_indices.view(-1))

        # x -> B*L, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }

        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # B*L, C -> B, C, L
        quant_out = quant_out.reshape((B, L, C)).permute(0, 2, 1)
        min_encoding_indices = min_encoding_indices.reshape((B, L))
        return quant_out, quantize_losses, min_encoding_indices


@dataclass
class VAEOutput:
    audio: Tensor
    z: Tensor
    losses: dict[str, Tensor]


class BiModalRVQVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Assertion to validate the channel information
        assert self.config.mid_channels[0] == self.config.down_channels[-1]
        assert self.config.mid_channels[-1] == self.config.down_channels[-1]
        assert len(self.config.down_sample) == len(self.config.down_channels) - 1
        assert len(self.config.attn_down) == len(self.config.down_channels) - 1

        self.up_sample = list(reversed(self.config.down_sample))
        self.init_encoder()
        self.init_decoder()

    def init_encoder(self):
        self.pqmf = PQMF(attenuation=self.config.attenuation, n_band=self.config.bands, sr=self.config.sample_rate, polyphase=True)
        self.encoder_conv_in = nn.Conv1d(self.config.bands, self.config.down_channels[0], kernel_size=3, padding=1)

        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.config.down_channels) - 1):
            self.encoder_layers.append(DownBlock1D(
                self.config.down_channels[i],
                self.config.down_channels[i + 1],
                down_sample=self.config.down_sample[i],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_down_layers,
                attn=self.config.attn_down[i],
                norm_channels=self.config.norm_channels,
                activation=self.config.activation,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.config.mid_channels) - 1):
            self.encoder_mids.append(MidBlock1D(
                self.config.mid_channels[i],
                self.config.mid_channels[i + 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_mid_layers,
                norm_channels=self.config.norm_channels,
                activation=self.config.activation,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.encoder_norm_out = nn.GroupNorm(self.config.norm_channels, self.config.down_channels[-1])
        self.encoder_conv_out = nn.Conv1d(self.config.down_channels[-1], self.config.z_channels, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv1d(self.config.z_channels, self.config.z_channels, kernel_size=1)

        # Codebook
        self.quants = nn.ModuleList([
            Quantizer(codebook_size=self.config.codebook_size, z_channels=self.config.z_channels)
            for _ in range(self.config.nquantizers)
        ])

    def init_decoder(self):
        # Post Quantization Convolution
        zchannels = self.config.nstems * self.config.z_channels
        self.post_quant_conv = nn.Conv1d(zchannels, zchannels, kernel_size=1)
        self.decoder_conv_in = nn.Conv1d(zchannels, self.config.mid_channels[-1], kernel_size=3, padding=1)

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.config.mid_channels))):
            self.decoder_mids.append(MidBlock1D(
                self.config.mid_channels[i],
                self.config.mid_channels[i - 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_mid_layers,
                norm_channels=self.config.norm_channels,
                activation=self.config.activation,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.config.down_channels))):
            self.decoder_layers.append(UpBlock1D(
                self.config.down_channels[i],
                self.config.down_channels[i - 1],
                up_sample=self.config.down_sample[i - 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_up_layers,
                attn=self.config.attn_down[i-1],
                norm_channels=self.config.norm_channels,
                activation=self.config.activation,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            ))

        self.decoder_norm_out = nn.GroupNorm(self.config.norm_channels, self.config.down_channels[0])
        self.decoder_conv_out = nn.Conv1d(self.config.down_channels[0], self.config.bands, kernel_size=1, padding=0)

    def encode(self, x: Tensor):
        # x: (B, S, L)
        B, S, L = x.shape
        out = x.flatten(0, 1)

        # x: (B * S, bands, L // bands)
        out = self.pqmf(out)
        out = self.encoder_conv_in(out)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = self.config.activation()(out)
        out = self.encoder_conv_out(out)
        # out: (B * S, z_channels, L') where L' is the length after downsampling
        out = self.pre_quant_conv(out)
        quant_losses = {
            "codebook_loss": torch.tensor(0.0, device=x.device),
            "commitment_loss": torch.tensor(0.0, device=x.device),
        }
        skip_quant = random.random() < self.config.p_skip_quantization
        if not self.training or not skip_quant:
            residual = out
            for idx, quant in enumerate(self.quants):
                quant_out, q_losses, _ = quant(residual)
                quant_losses["codebook_loss"] += q_losses["codebook_loss"]
                quant_losses["commitment_loss"] += q_losses["commitment_loss"]
                if idx == 0:
                    out = quant_out
                else:
                    out += quant_out
                residual -= quant_out
        out = out.unflatten(0, (B, S))
        # out shape: (B, S, z_channels, L')
        return out, quant_losses

    def decode(self, z: Tensor):
        B, S, Z, L_ = z.shape
        assert Z == self.config.z_channels, f"Expected z_channels {self.config.z_channels}, got {Z}"
        assert S == self.config.nstems, f"Expected nstems {self.config.nstems}, got {S}"
        # B, S, Z, L' -> B, S * Z, L'
        out = z.flatten(1, 2)
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = self.config.activation()(out)
        out = self.decoder_conv_out(out)
        # out: (B, band, L)
        return out

    def forward(self, x):
        # x: (B, S, L)
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        audio_out = self.pqmf.inverse(out)
        return VAEOutput(
            audio=audio_out,
            z=z,
            losses=quant_losses
        )

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, weight: dict[str, float]):
        B, S, L = input.shape
        B_, L_ = target.shape
        assert B == B_, f"Batch size mismatch: {B} != {B_}"
        assert L == L_, f"Length mismatch: {L} != {L_}"
        assert S == self.config.nstems, f"Expected nstems {self.config.nstems}, got {S}"
        z, quant_losses = self.encode(input)
        y_pred = self.decode(z)
        y_true = self.pqmf(target)
        recon_loss = NF.mse_loss(y_pred, y_true, reduction='mean')
        audio_out = self.pqmf.inverse(y_pred)
        losses = {
            'Reconstruction Loss': recon_loss,
            'Codebook Loss': quant_losses['codebook_loss'],
            'Commitment Loss': quant_losses['commitment_loss']
        }
        total = sum((losses[key] * weight.get(key, 1.0) for key in losses), start=torch.tensor(0.0, device=input.device))
        return total, VAEOutput(
            audio=audio_out,
            z=z,
            losses=losses
        )


def test_up_block_1d():
    def test(factor: int):
        block = UpBlock1D(
            in_channels=64,
            out_channels=64,
            up_sample=factor,
            num_heads=8,
            num_layers=2,
            attn=True,
            norm_channels=32,
            activation=nn.SiLU,
            use_gradient_checkpointing=False
        )
        input_tensor = torch.randn(1, 64, 720)
        output_tensor = block(input_tensor)
        assert output_tensor.shape == (1, 64, 720 * factor), f"Output shape mismatch: {output_tensor.shape}, expected {(1, 64, 720 * factor)} (factor={factor})"
        print("UpBlock1D test passed successfully.")

    for i in range(2, 100):
        test(i)
    print("All tests passed.")


def test_down_block_1d():
    def test(factor: int):
        downblock = nn.Conv1d(64, 64, kernel_size=2*factor, stride=factor, padding=factor-1)
        input_tensor = torch.randn(1, 64, factor*720)
        output_tensor = downblock(input_tensor)
        assert output_tensor.shape == (1, 64, 720), f"Output shape mismatch: {output_tensor.shape}, expected (1, 64, 720) (factor={factor})"

    for i in range(2, 100):
        test(i)
    print("All tests passed.")
