# Implements VQVAE encoder and decoder model to convert spectrogram to latent space and back to spectrogram.
# Code in part referenced from https://github.com/explainingai-code/StableDiffusion-PyTorch

import random
import torch
import torch.functional as F
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from AutoMasher.fyp.audio.base.audio_collection import DemucsCollection
from .config import VAEConfig


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(self, in_channels, out_channels, down_sample, num_heads, num_layers, attn, norm_channels, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True, bias=False)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                # Attention block of Unet
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                if self.use_gradient_checkpointing:
                    out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn)  # type: ignore
                else:
                    out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """

    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_channels, use_gradient_checkpointing=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True, bias=False)
             for _ in range(num_layers)]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            if self.use_gradient_checkpointing:
                out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn)  # type: ignore
            else:
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, in_channels, out_channels, up_sample, num_heads, num_layers, attn, norm_channels, use_gradient_checkpointing=False, final=True):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True, bias=False)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # Add one to the final conv transpose to make it 2n + 1
        if not self.up_sample:
            self.up_sample_conv = nn.Identity()
        elif final:
            self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(3, 2), stride=2, padding=0, output_padding=(1, 0))
        else:
            self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.final = final

    def forward(self, x):
        # Upsample
        # x: (B, C, N, T)
        x = self.up_sample_conv(x)
        if self.final:
            x = x[:, :, :-1, :]

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                if self.use_gradient_checkpointing:
                    out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn)  # type: ignore
                else:
                    out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class QuantizeModule(nn.Module):
    """Takes in x of shape (B, C, H, W) and returns quantized output and indices

    Returns:
        quant_out: Tensor of shape (B, C, H, W)
        quantize_losses: dict with 'codebook_loss' and 'commitment_loss'
        min_encoding_indices: Tensor of shape (B, H*W)
    """

    def __init__(self, codebook_size: int, z_channels: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, z_channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.codebook.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.codebook.weight, 0, min_encoding_indices.view(-1))

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices


class VAEOutput:
    def __init__(self, output: Tensor, z: Tensor, codebook_loss: Tensor, commitment_loss: Tensor, entropy_loss: Tensor):
        self.output = output
        self.z = z
        self.codebook_loss = codebook_loss
        self.commitment_loss = commitment_loss
        self.entropy_loss = entropy_loss


class RVQVAE(nn.Module):
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
        self.encoder_conv_in = nn.Conv2d(self.config.nsources, self.config.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.config.down_channels) - 1):
            self.encoder_layers.append(DownBlock(
                self.config.down_channels[i],
                self.config.down_channels[i + 1],
                down_sample=self.config.down_sample[i],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_down_layers,
                attn=self.config.attn_down[i],
                norm_channels=self.config.norm_channels,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.config.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(
                self.config.mid_channels[i],
                self.config.mid_channels[i + 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_mid_layers,
                norm_channels=self.config.norm_channels,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.encoder_norm_out = nn.GroupNorm(self.config.norm_channels, self.config.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.config.down_channels[-1], self.config.z_channels, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.config.z_channels, self.config.z_channels, kernel_size=1)

        # Codebook
        self.quants = nn.ModuleList([
            QuantizeModule(codebook_size=self.config.codebook_size, z_channels=self.config.z_channels)
            for _ in range(self.config.nquantizers)
        ])

    def init_decoder(self):
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.config.z_channels, self.config.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.config.z_channels, self.config.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.config.mid_channels))):
            self.decoder_mids.append(MidBlock(
                self.config.mid_channels[i],
                self.config.mid_channels[i - 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_mid_layers,
                norm_channels=self.config.norm_channels,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            ))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.config.down_channels))):
            self.decoder_layers.append(UpBlock(
                self.config.down_channels[i],
                self.config.down_channels[i - 1],
                up_sample=self.config.down_sample[i - 1],
                num_heads=self.config.num_heads,
                num_layers=self.config.num_up_layers,
                attn=self.config.attn_down[i-1],
                norm_channels=self.config.norm_channels,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
                final=i == 1
            ))

        self.decoder_norm_out = nn.GroupNorm(self.config.norm_channels, self.config.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.config.down_channels[0], self.config.nsources, kernel_size=3, padding=1)

    def encode(self, x: Tensor):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        quant_losses = {
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "entropy_loss": 0.0
        }
        if random.random() > self.config.p_skip_quantization:
            residual = out
            for idx, quant in enumerate(self.quants):
                quant_out, q_losses, indices = quant(residual)
                quant_losses["codebook_loss"] += q_losses["codebook_loss"]
                quant_losses["commitment_loss"] += q_losses["commitment_loss"]
                p = torch.bincount(indices.flatten(), minlength=self.config.codebook_size) / indices.size(0)
                entropy_loss = -torch.mean(p * torch.log(p + 1e-10))
                quant_losses["entropy_loss"] += entropy_loss  # type: ignore
                if idx == 0:
                    out = quant_out
                else:
                    out += quant_out
                residual -= quant_out
        return out, quant_losses

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return VAEOutput(
            output=out,
            z=z,
            codebook_loss=quant_losses["codebook_loss"],  # type: ignore
            commitment_loss=quant_losses["commitment_loss"],  # type: ignore
            entropy_loss=quant_losses["entropy_loss"]  # type: ignore
        )
