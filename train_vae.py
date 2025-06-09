import line_profiler
import audiofile as af
import argparse
import torch
import random
import os
import hashlib
import wandb
import numpy as np
import torch.nn.functional as F
import random
import json  # Added import
from collections import defaultdict
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
from accelerate import Accelerator
from math import isclose

from src.audio import YouTubeURL, Audio
from src.modules import VAE, VAEOutput
from src.vggish import Vggish
from src.config import VAEConfig
from src.separate import Separator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)


class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, config: VAEConfig, paths: list[str]):
        self.paths = paths
        self.config = config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = os.path.join(self.config.dataset_dir, self.paths[idx])
        total_samples = af.samples(path)
        orig_sr = af.sampling_rate(path)
        chunk_length = int(self.config.audio_length * orig_sr / self.config.sample_rate)
        if chunk_length > total_samples:
            print(f"Chunk length {chunk_length} is greater than total samples {total_samples} in {path}")
            return self.__getitem__(random.randint(0, len(self.paths) - 1))

        start_sample = random.randint(0, total_samples - chunk_length)
        stop_sample = start_sample + chunk_length
        try:
            signal, sampling_rate = af.read(path, offset=f"{start_sample}", duration=f"{chunk_length}", always_2d=True)
            audio = Audio(torch.from_numpy(signal), sampling_rate)
        except Exception as e:
            # There is a small but nonzero chance that the audio cannot be read
            print(f"Error reading audio from {path}: {e}")
            return self.__getitem__(random.randint(0, len(self.paths) - 1))
        try:
            separator = Separator(model_name=self.config.separator)
            separated_audio = separator.separate(audio)
        except Exception as e:
            # There is a small but nonzero chance that the audio cannot be read after it is separated
            return self.__getitem__(random.randint(0, len(self.paths) - 1))
        if len(separated_audio) != self.config.nstems:
            raise ValueError(
                f"Expected {self.config.nstems} stems, but got {len(separated_audio)} stems from {path}"
            )
        take_left_channel = int(random.choice([True, False]))
        separated_audio = [a.resample(self.config.sample_rate)
                            .pad(self.config.audio_length, warn=1024)
                            .to_nchannels(2) for a in separated_audio]
        audio = audio \
            .resample(self.config.sample_rate) \
            .pad(self.config.audio_length, warn=1024) \
            .to_nchannels(2)
        separated_audio.append(audio)
        thing = torch.stack([a.data[take_left_channel] for a in separated_audio], dim=0)  # shape: S+1, L
        return thing


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def partition_files(paths: list[str], percents: dict[str, float]) -> dict[str, list[str]]:
    # Use a hacky deterministic partitioning based on the SHA-256 hash of the path
    # to accomodate the expansion of our dataset in the near future
    assert sum(percents.values()) == 1.0, "Percentages must sum to 1.0"
    splits = {k: set() for k in percents.keys()}
    for path in paths:
        hash_object = hashlib.sha256(path.encode())
        hash_digest = hash_object.hexdigest()
        probability = int(hash_digest, 16) / (2**256 - 1)
        for split, x_percent in percents.items():
            if probability < x_percent:
                splits[split].add(path)
                break
            else:
                probability -= x_percent
    return {s: sorted(p) for s, p in splits.items()}


def signal_noise_ratio(pred_audio: Tensor, target_audio: Tensor, zero_mean: bool = False) -> float:
    eps = torch.finfo(pred_audio.dtype).eps
    if zero_mean:
        target_audio = target_audio - torch.mean(target_audio, dim=-1, keepdim=True)
        pred_audio = pred_audio - torch.mean(pred_audio, dim=-1, keepdim=True)
    noise = target_audio - pred_audio
    snr_value = (torch.sum(target_audio**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return torch.mean(snr_value).item()


def inference(
    target_audio: Tensor,
    config: VAEConfig,
    model: VAE,
):
    assert isinstance(target_audio, torch.Tensor)
    assert target_audio.dim() == 3

    input_audio = target_audio[:, :-1]  # Remove the last channel which is the real audio
    target_audio = target_audio[:, -1]
    assert input_audio.shape[1:] == (config.nstems, config.audio_length), \
        f"Expected target audio shape to be (_, {config.nstems}, {config.audio_length}), " \
        f"but got {input_audio.shape}"

    input_audio = input_audio.float().to(device)
    target_audio = target_audio.float().to(device)

    # Fetch autoencoders output(reconstructions)
    with autocast('cuda'):
        g_loss, model_output = model.compute_loss(input_audio, target_audio, {
            "Codebook Loss": config.codebook_weight,
            "Commitment Loss": config.commitment_beta,
        })

    pred_audio = model_output.audio  # B, L
    assert target_audio.device == pred_audio.device, f"Target audio and predicted audio must be on the same device, got {target_audio.device} and {pred_audio.device}"
    assert target_audio.shape == pred_audio.shape, f"Target audio and predicted audio must have the same shape, got {target_audio.shape} and {pred_audio.shape}"

    snr = signal_noise_ratio(pred_audio, target_audio, zero_mean=True)

    components: dict[str, float] = {k: v.item() for k, v in model_output.losses.items()} | {
        "Generator Loss": g_loss.item(),
        "SNR": snr,
    }

    return model_output, components, g_loss


def validate(
    config: VAEConfig,
    model: VAE,
    val_data_loader: DataLoader,
    step_count: int,
):
    val_count_ = 0
    if step_count % config.val_steps != (1 if config.validate_at_step_1 else 0):
        return

    model.eval()
    log_audio = None
    val_loss_components = defaultdict(list)
    with torch.no_grad():
        for target_audio in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(config.val_count, len(val_data_loader))):
            val_count_ += 1
            if val_count_ > config.val_count:
                break

            model_output, components, g_loss = inference(
                target_audio,
                config,
                model
            )
            pred_audio = model_output.audio  # B, L

            for c, x in components.items():
                val_loss_components[c].append(x)

            if log_audio is None:
                log_audio = (target_audio[0], pred_audio[0])

    wandb.log({f"Val {c}": np.mean(x) for c, x in val_loss_components.items()}, step=step_count)
    tqdm.write(f"Validation complete")

    # Log an audio sample
    if log_audio is not None:
        target_audio, pred_audio = log_audio
        assert target_audio.dim() == 2 and pred_audio.dim() == 1, f"Target and predicted audio must be 1D tensors, got {target_audio.dim()} and {pred_audio.dim()}"
        log_audios = {}
        for i in range(len(target_audio)):
            log_audios[f"target_{i + 1}"] = wandb.Audio(target_audio[i].float().cpu().numpy(), sample_rate=config.sample_rate, caption=f"Target Channel {i + 1}")
        log_audios["predicted"] = wandb.Audio(pred_audio.float().cpu().numpy(), sample_rate=config.sample_rate, caption="Predicted Audio")
        wandb.log({
            "Validation Audio": log_audios
        }, step=step_count)
    model.train()


def train(config: VAEConfig, start_from_iter: int = 0):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""
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
        shuffle=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=False
    )

    os.makedirs(config.output_dir, exist_ok=True)

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

            _, components, g_loss = inference(
                target_audio,
                config,
                model,
            )

            accelerator.backward(g_loss / config.autoencoder_acc_steps)

            if step_count % config.autoencoder_acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            losses = {c: x for c, x in components.items()}
            losses["Total Generator Loss"] = g_loss.item()
            losses["Generator Learning Rate"] = optimizer_g.param_groups[0]['lr']
            wandb.log(losses, step=step_count)

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
                    step_count,
                )

        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_g.step()
        optimizer_g.zero_grad()
        model_save_path = config.get_vae_save_path(step_count)
        torch.save(model.state_dict(), model_save_path)

        if stop_training:
            break

    print('Done Training...')


def main():
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='resources/config/vae.yaml', type=str)
    parser.add_argument('--start_iter', dest='start_iter', type=int, default=0)
    parser.add_argument('--sweep', dest='sweep', action='store_true', default=False,)
    args = parser.parse_args()
    config = VAEConfig.load(args.config_path)
    wandb.login()

    if args.sweep:
        sweep_config_path = './resources/config/vaesweep.json'
        with open(sweep_config_path, 'r') as f:
            sweep_config = json.load(f)

        def train_sweep():
            with wandb.init(project=config.run_name):
                config_ = config.asdict()
                config_.update(wandb.config)
                del config_['_run_id']
                sweep_run_config = VAEConfig(**config_)
                print("Sweep run config:", sweep_run_config.asdict())
                train(sweep_run_config, start_from_iter=args.start_iter)

        sweep_id = wandb.sweep(sweep_config, project=config.run_name)
        wandb.agent(sweep_id, function=train_sweep, count=sweep_config.get('run_cap', None))
    else:
        with wandb.init(
            project=config.run_name,
            config=config.asdict(),
        ):
            train(config, start_from_iter=args.start_iter)


if __name__ == '__main__':
    main()
