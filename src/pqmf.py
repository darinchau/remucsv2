import math
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from einops import rearrange
from scipy.optimize import fmin
from scipy.signal import firwin, kaiser_beta, kaiserord
from scipy.signal.windows import kaiser
from src.audio import Audio


def reverse_half(x: torch.Tensor):
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1
    return x * mask


def center_pad_next_pow_2(x: torch.Tensor):
    next_2 = 2**math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))


def freq_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700)


def get_qmf_bank(h: torch.Tensor, n_band: int):
    """
    Modulates an input protoype filter into a bank of cosine modulated filters
    Args:
        h (torch.Tensor): Prototype filter (T,)
        n_band (int): Number of sub-bands
    """
    k = torch.arange(n_band).reshape(-1, 1)
    N = h.shape[-1]
    t = torch.arange(-(N // 2), N // 2 + 1)
    p = (-1) ** k * math.pi / 4
    mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)
    hk = 2 * h * mod
    return hk


def kaiser_filter(wc: float, atten: float, N: int | None = None) -> NDArray[np.floating]:
    """
    Computes a kaiser lowpass filter
    Args:
        wc (float): Cutoff frequency in radians
        atten (float): Attenuation in dB
        N (int | None): Filter length, if None it will be computed based on the attenuation
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, fs=np.pi*2)  # type: ignore
    return h


def loss_wc(wc: float, atten: float, M: int, N: int | None = None):
    # https://ieeexplore.ieee.org/document/681427
    h = kaiser_filter(wc, atten, N)
    g = np.convolve(h, h[::-1], "full")
    g = abs(g[g.shape[-1] // 2::2 * M][1:])
    return np.max(g)


def get_prototype(atten: float, M: int, N: int | None = None):
    """
    Given an attenuation objective and the number of bands
    returns the corresponding lowpass filter
    """
    wc = fmin(lambda w: loss_wc(w, atten, M, N), 1 / M, disp=0)[0]
    return kaiser_filter(wc, atten, N)


def polyphase_forward(x: torch.Tensor, hk: torch.Tensor, rearrange_filter: bool = True):
    """
    Polyphase implementation of the analysis process (fast)
    Args:
        x (torch.Tensor): Signal to analyze (B, 1, T)
        hk (torch.Tensor): Filter bank (M, T)
        rearrange_filter (bool): Whether to rearrange the filter bank for polyphase processing
    """
    x = rearrange(x, "b c (t m) -> b (c m) t", m=hk.shape[0])
    if rearrange_filter:
        hk = rearrange(hk, "c (t m) -> c m t", m=hk.shape[0])
    x = nn.functional.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]
    return x


def polyphase_inverse(x: torch.Tensor, hk: torch.Tensor, rearrange_filter=True):
    """
    Polyphase implementation of the synthesis process (fast)
    Args:
        x (torch.Tensor): Signal to synthesize from (B, 1, T)
        hk (torch.Tensor): Filter bank (M, T)
        rearrange_filter (bool): Whether to rearrange the filter bank for polyphase processing
    """
    m = hk.shape[0]

    if rearrange_filter:
        hk = hk.flip(-1)
        hk = rearrange(hk, "c (t m) -> m c t", m=m)  # polyphase

    pad = hk.shape[-1] // 2 + 1
    x = nn.functional.conv1d(x, hk, padding=int(pad))[..., :-1] * m

    x = x.flip(1)
    x = rearrange(x, "b (c m) t -> b c (t m)", m=m)
    x = x[..., 2 * hk.shape[1]:]
    return x


def classic_forward(x: torch.Tensor, hk: torch.Tensor):
    """
    Naive implementation of the analysis process (slow)
    Args:
        x (torch.Tensor): Signal to analyze (B, 1, T)
        hk (torch.Tensor): Filter bank (M, T)
    """
    x = nn.functional.conv1d(
        x,
        hk.unsqueeze(1),
        stride=hk.shape[0],
        padding=hk.shape[-1] // 2,
    )[..., :-1]
    return x


def classic_inverse(x: torch.Tensor, hk: torch.Tensor):
    """
    Naive implementation of the synthesis process (slow)
    Args:
        x (torch.Tensor): Signal to synthesize from (B, 1, T)
        hk (torch.Tensor): Filter bank (M, T)
    """
    hk = hk.flip(-1)
    y = torch.zeros(*x.shape[:2], hk.shape[0] * x.shape[-1]).to(x)
    y[..., ::hk.shape[0]] = x * hk.shape[0]
    y = nn.functional.conv1d(
        y,
        hk.unsqueeze(0),
        padding=hk.shape[-1] // 2,
    )[..., 1:]
    return y


class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter multiband decomposition / reconstruction
    Args:
        attenuation (float): Attenuation in dB for the prototype filter.
        n_band (int): Number of sub-bands. Must be a power of 2 if polyphase is True.
        polyphase (bool): Whether to use the polyphase algorithm for analysis and synthesis.
        n_channels (int): Number of channels in the input signal. Default is 1.

    forward(x: torch.Tensor) -> torch.Tensor:
        Applies the PQMF analysis to the input signal x with shape (B, T)

    inverse(x: torch.Tensor) -> torch.Tensor:
        Applies the PQMF synthesis to the input signal x with shape (B, M, T)
    """

    def __init__(self, attenuation: float, n_band: int, polyphase: bool = True):
        super().__init__()
        h = get_prototype(attenuation, n_band)

        if polyphase:
            power = math.log2(n_band)
            assert power == math.floor(power), "when using the polyphase algorithm, n_band must be a power of 2"

        h = torch.from_numpy(h).float()
        hk = get_qmf_bank(h, n_band)
        hk = center_pad_next_pow_2(hk)

        self.register_buffer("hk", hk)
        self.register_buffer("h", h)
        self.n_band = n_band
        self.polyphase = polyphase

    def forward(self, x: torch.Tensor):
        B, L = x.shape
        x = x.unsqueeze(1)
        if self.n_band == 1:
            return x
        if self.polyphase:
            x = polyphase_forward(x, self.hk)  # type: ignore
        else:
            x = classic_forward(x, self.hk)  # type: ignore
        x = reverse_half(x)
        return x

    def inverse(self, x: torch.Tensor):
        B, M, T = x.shape
        if self.n_band == 1:
            x = x.squeeze(1)
            return x
        if M != self.n_band:
            raise ValueError(f"Expected {self.n_band} bands, got {M} bands")
        x = reverse_half(x)
        if self.polyphase:
            y = polyphase_inverse(x, self.hk)  # type: ignore
        else:
            y = classic_inverse(x, self.hk)  # type: ignore
        y = y.squeeze(1)
        return y
