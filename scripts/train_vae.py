# This script is used to train the VQ-VAE model with a discriminator for adversarial loss
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
# Adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
import yaml
import argparse
import torch
import random
import os
import hashlib
import wandb
import pickle
import numpy as np
import json
import torch.nn.functional as F
import time
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
from accelerate import Accelerator
from math import isclose

from src.audio import YouTubeURL
from src.vae import RVQVAE as VAE, VAEOutput
from src.vggish import Vggish
from src.config import VAEConfig
from src.stft import STFT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, config: VAEConfig, urls: list[str]):
        self.urls = urls
        self.config = config
        ...

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx: int):
        ...


def load_lpips(config: VAEConfig):
    class _PerceptualLossWrapper(nn.Module):
        def __init__(self, in_sr: int):
            super().__init__()
            self.lpips = Vggish()
            self.in_sr = in_sr

        def forward(self, pred_audio, targ_audio):
            pred1 = self.lpips((pred_audio, self.in_sr))
            targ1 = self.lpips((targ_audio, self.in_sr))
            return F.mse_loss(pred1, targ1)
            # pred2 = self.lpips((pred_audio, self.lpips.input_sr))
            # targ2 = self.lpips((targ_audio, self.lpips.input_sr))
            # return (F.mse_loss(pred1, targ1) + F.mse_loss(pred2, targ2)) * 0.5

    model = _PerceptualLossWrapper(config.sample_rate)
    model = model.eval().to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def partition_files(urls: list[str], percents: dict[str, float]) -> dict[str, list[str]]:
    # Use a hacky deterministic partitioning based on the SHA-256 hash of the URL
    # to accomodate the expansion of our dataset in the near future
    assert sum(percents.values()) == 1.0, "Percentages must sum to 1.0"
    splits = {k: set() for k in percents.keys()}
    for url in urls:
        hash_object = hashlib.sha256(url.encode())
        hash_digest = hash_object.hexdigest()
        probability = int(hash_digest, 16) / (2**256 - 1)
        for split, x_percent in percents.items():
            if probability < x_percent:
                splits[split].add(url)
                break
            else:
                probability -= x_percent
    return {split: sorted(urls) for split, urls in splits.items()}


def inference(
    target_audio: Tensor,
    config: VAEConfig,
    model: VAE,
    stft: STFT,
    reconstruction_loss: nn.Module,
):
    target_audio = target_audio.flatten(0, 1)  # im shape: B, 4, L

    assert isinstance(target_audio, torch.Tensor)
    assert target_audio.dim() == 3  # im shape: B, nsources/2=4, L
    assert target_audio.shape[1] == config.nstems

    batch_size = target_audio.shape[0]
    target_audio = target_audio.float().to(device)
    target_spec = stft.forward(
        target_audio.float().flatten(0, 1)  # B*4, L
    )
    target_spec = torch.view_as_real(target_spec).permute(0, 3, 1, 2)  # B*4, 2, N, T
    target_spec = target_spec.unflatten(0, (batch_size, 4)).flatten(1, 2).float().to(device)  # B, 4*2, N, T

    # Fetch autoencoders output(reconstructions)
    with autocast('cuda'):
        model_output: VAEOutput = model(target_spec)

    pred_spec = model_output.output
    pred_audio = stft.griffin_lim(
        pred_spec.float().flatten(0, 1)
    ).unflatten(0, (batch_size, config.nstems))

    with autocast('cuda'):
        recon_loss = reconstruction_loss(pred_spec, target_spec)

    g_loss: torch.Tensor = recon_loss + \
        config.codebook_weight * model_output.codebook_loss + \
        config.commitment_beta * model_output.commitment_loss
    g_loss /= config.autoencoder_acc_steps
    return model_output, pred_spec, pred_audio, recon_loss, g_loss


def validate(
    config: VAEConfig,
    model: VAE,
    val_data_loader: DataLoader,
    reconstruction_loss: nn.Module,
    step_count: int,
    stft: STFT
):
    val_count_ = 0
    if step_count % config.val_steps != (1 if config.validate_at_step_1 else 0):
        return

    model.eval()
    with torch.no_grad():
        val_recon_losses = []
        val_codebook_losses = []
        val_commitment_losses = []
        for target_audio in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(config.val_count, len(val_data_loader))):
            val_count_ += 1
            if val_count_ > config.val_count:
                break

            ###  Copy pasted from training loop ###
            model_output, pred_spec, pred_audio, recon_loss, g_loss = inference(
                target_audio,
                config,
                model,
                stft,
                reconstruction_loss
            )

            #######################################
            val_recon_loss = recon_loss.item()
            val_recon_losses.append(val_recon_loss)

            val_codebook_loss = model_output.codebook_loss.item()
            val_codebook_losses.append(val_codebook_loss)

            val_commitment_loss = model_output.commitment_loss.item()
            val_commitment_losses.append(val_commitment_loss)

    wandb.log({
        "Val Reconstruction Loss": np.mean(val_recon_losses),
        "Val Codebook Loss": np.mean(val_codebook_losses),
        "Val Commitment Loss": np.mean(val_commitment_losses),
    }, step=step_count)

    tqdm.write(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Codebook loss: {np.mean(val_codebook_losses)}")
    model.train()


def train(config_path: str, start_from_iter: int = 0):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""
    config = VAEConfig.load(config_path)
    set_seed(config.seed)

    # Create the model and dataset #
    model = VAE(config).to(device)
    print(f"Starting from iteration {start_from_iter}")

    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))

    # Create the dataset
    files = os.listdir(config.dataset_dir)
    split = partition_files(files, {
        "train": TRAIN_SPLIT_PERCENTAGE,
        "val": VALIDATION_SPLIT_PERCENTAGE,
        "test": TEST_SPLIT_PERCENTAGE
    })
    im_dataset = VAEDataset(config, split['train'])
    val_dataset = VAEDataset(config, split['val'])

    print('Dataset size: {}'.format(len(im_dataset)))
    print(f"Effective audio length: {config.audio_length / config.sample_rate} seconds")

    data_loader = DataLoader(
        im_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=False
    )

    os.makedirs(config.output_dir, exist_ok=True)

    reconstruction_loss = torch.nn.MSELoss()

    def warmup_lr_scheduler(optimizer, warmup_steps, base_lr):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    optimizer_g = Adam(model.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))

    warmup_steps = config.warmup_steps
    scheduler_g = warmup_lr_scheduler(optimizer_g, warmup_steps, config.autoencoder_lr)

    accelerator = Accelerator(mixed_precision="bf16")

    step_count = 0
    progress_bar = tqdm(total=config.steps + start_from_iter, desc="Training Progress")

    # Reload checkpoint
    if start_from_iter > 0:
        model_save_path = config.get_vae_save_path(start_from_iter)
        model_sd = torch.load(model_save_path)
        model.load_state_dict(model_sd)
        step_count = start_from_iter
        progress_bar.update(start_from_iter)

    model, optimizer_g, data_loader = accelerator.prepare(
        model, optimizer_g, data_loader
    )

    stft = STFT(config.nfft, config.ntimeframes)

    wandb.init(
        # set the wandb project where this run will be logged
        project=config.run_name,
        config=config.asdict()
    )

    model.train()

    while True:
        optimizer_g.zero_grad()
        stop_training: bool = False

        for target_audio in data_loader:
            step_count += 1
            progress_bar.update(1)
            if step_count >= config.steps + start_from_iter:
                stop_training = True
                break

            model_output, pred_spec, pred_audio, recon_loss, g_loss = inference(
                target_audio,
                config,
                model,
                stft,
                reconstruction_loss
            )

            accelerator.backward(g_loss)

            if step_count % config.autoencoder_acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Log losses
            wandb.log({
                "Reconstruction Loss": recon_loss.item(),
                "Codebook Loss": model_output.codebook_loss.item(),
                "Commitment Loss": model_output.commitment_loss.item(),
                "Total Generator Loss": g_loss.item(),
                "Generator Learning Rate": optimizer_g.param_groups[0]['lr'],
            }, step=step_count)

            if step_count % config.save_steps == 0:
                model_save_path = VAEConfig.get_vae_save_path(config, step_count)
                torch.save(model.state_dict(), model_save_path)

            scheduler_g.step()

            ########### Perform Validation #############
            with torch.no_grad():
                validate(
                    config,
                    model,
                    val_data_loader,
                    reconstruction_loss,
                    step_count,
                    stft
                )

        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_g.step()
        optimizer_g.zero_grad()
        model_save_path = config.get_vae_save_path(step_count)
        torch.save(model.state_dict(), model_save_path)

        if stop_training:
            break

    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='resources/config/vae.yaml', type=str)
    parser.add_argument('--start_iter', dest='start_iter', type=int, default=0)
    args = parser.parse_args()
    train(args.config_path, start_from_iter=args.start_iter)
