import os
from dataclasses import dataclass, asdict
from typing import List
import yaml
from .stft import STFT


@dataclass(frozen=True)
class VAEConfig:
    separator: str
    nstems: int
    dataset_dir: str
    output_dir: str
    val_count: int
    sample_rate: int
    ntimeframes: int
    nfft: int

    codebook_size: int
    nquantizers: int
    down_channels: tuple[int, ...]
    mid_channels: tuple[int, ...]
    down_sample: tuple[int, ...]
    attn_down: tuple[bool, ...]
    norm_channels: int
    num_heads: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int

    seed: int
    gradient_checkpointing: bool
    num_workers_dl: int
    batch_size: int
    codebook_weight: float
    commitment_beta: float
    entropy_weight: float
    steps: int
    autoencoder_lr: float
    autoencoder_acc_steps: int
    save_steps: int
    ckpt_name: str
    run_name: str
    val_steps: int
    warmup_steps: int
    validate_at_step_1: bool

    @property
    def nsources(self):
        """Returns the number of spectrograms that we will work with
        Different from nstems in the sense that it is the number of channels for the VAE input
        which could be different from the number of stems in the dataset."""
        return self.nstems * 2

    @property
    def z_channels(self) -> int:
        """Number of conv channels in the latent space."""
        return self.nstems

    @property
    def audio_length(self) -> int:
        return STFT(self.nfft, self.ntimeframes).l

    @staticmethod
    def load(file_path: str) -> 'VAEConfig':
        """Loads the VAEConfig from a YAML file. By default loads the one inside resources/config"""
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        return VAEConfig(**yaml_data)

    def get_vae_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, f"step-{step:06d}", self.ckpt_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def asdict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d
