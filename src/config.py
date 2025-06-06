import os
from dataclasses import dataclass, asdict, field
import yaml
import torch.nn as nn


def get_random_string(length: int = 14) -> str:
    """Generates a random string of fixed length."""
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@dataclass(frozen=True)
class VAEConfig:
    separator: str
    nstems: int
    dataset_dir: str
    output_dir: str
    val_count: int
    sample_rate: int
    length: int

    codebook_size: int
    nquantizers: int
    z_channels: int
    bands: int
    attenuation: int
    down_channels: tuple[int, ...]
    mid_channels: tuple[int, ...]
    down_sample: tuple[int, ...]
    attn_down: tuple[bool, ...]
    norm_channels: int
    num_heads: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    activation_fn: str

    seed: int
    gradient_checkpointing: bool
    num_workers_dl: int
    batch_size: int
    codebook_weight: float
    commitment_beta: float
    p_skip_quantization: float
    steps: int
    autoencoder_lr: float
    autoencoder_acc_steps: int
    save_steps: int
    ckpt_name: str
    run_name: str
    val_steps: int
    warmup_steps: int
    validate_at_step_1: bool

    _run_id: str = field(init=False, repr=False, default_factory=lambda: get_random_string())

    @property
    def audio_length(self) -> int:
        """Sometimes we want to define the audio length in terms of spectrogram dimensions
        This provides a consistent getter for the audio length"""
        return self.length

    def activation(self) -> nn.Module:
        """Returns the activation function used in the VAE"""
        return {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(negative_slope=0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }[self.activation_fn]

    @staticmethod
    def load(file_path: str) -> 'VAEConfig':
        """Loads the VAEConfig from a YAML file. By default loads the one inside resources/config"""
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        return VAEConfig(**yaml_data)

    def get_vae_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, self._run_id, f"step-{step:06d}", self.ckpt_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def asdict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d
