import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt


class STFT:
    """Try to enforce the constraint about the inputs and outputs of the stft such that input is always (B, L) and output is (B, N, T)"""

    def __init__(self, n: int, t: int):
        # Check that N must be one more than a power of 2 and n//4 > 1
        assert (n > 3) and (((n - 1) & (n - 2)) == 0), f"n must be one more than a power of 2, got {n}"
        self.n = n
        self.t = t
        self.n_fft = 2 * n - 2
        self.hop_length = n // 4
        self._skip_checks = False

    @property
    def l(self):
        # Check that L is solvable: T = 1 + L // hop_length
        return (self.t - 1) * self.hop_length

    def _assert_is_audio(self, x: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(x, torch.Tensor), f"Expected input to be a tensor, got {type(x)}"
        assert x.ndim == 2, f"Expected input to have 2 dimensions, got {x.ndim}"
        if bs is None:
            bs = x.shape[0]
        assert x.shape == (bs, self.l), f"Expected input to have shape (B, {self.l}), got {x.shape}"

    def _assert_is_spectrogram(self, z: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(z, torch.Tensor), f"Expected output to be a tensor, got {type(z)}"
        if bs is None:
            bs = z.shape[0]
        assert z.shape == (bs, self.n, self.t), f"Expected output to have shape (B, {self.n}, {self.t}), got {z.shape}"
        assert torch.is_complex(z), f"Expected output to be complex, got {z.dtype}"

    def _assert_is_mel(self, mel: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(mel, torch.Tensor), f"Expected output to be a tensor, got {type(mel)}"
        if bs is None:
            bs = mel.shape[0]
        assert mel.shape[2] == self.t, f"Expected output to have shape (B, nmels, {self.t}), got {mel.shape}"
        assert torch.is_floating_point(mel), f"Expected output to be floating point, got {mel.dtype}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._assert_is_audio(x)

        z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            onesided=True,
        )

        self._assert_is_spectrogram(z, bs=x.shape[0])
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        self._assert_is_spectrogram(z)

        # We expect no trimming or padding to be done, so length=None
        x = torch.istft(
            z,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(z.real),
            win_length=self.n_fft,
            normalized=True,
            center=True
        )

        self._assert_is_audio(x, bs=z.shape[0])
        return x

    def griffin_lim(self, z: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
        # TODO: test this function
        self._assert_is_mel(z)  # mel spectrograms checks for real-valued specs
        if not self._skip_checks:
            assert z.shape[1] == self.n

        # We expect no trimming or padding to be done, so length=None
        x = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            n_iter=n_iter,
            power=1,
            length=self.l
        ).to(z.device)(z)

        self._assert_is_audio(x, bs=z.shape[0])
        return x

    def mel(self, x: torch.Tensor, sample_rate: int, nmels: int = 80) -> torch.Tensor:
        self._assert_is_audio(x)

        # Compute the mel filterbank
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            n_mels=nmels,
            center=True,
            pad_mode='reflect',
            power=2.0,
        )(x)

        self._assert_is_mel(mel, bs=x.shape[0])
        return mel

    def mel_librosa(self, x: torch.Tensor, sample_rate: int, nmels: int = 80):
        self._assert_is_audio(x)
        assert x.ndim == 2, f"Expected input to have 2 dimensions, got {x.ndim}"
        assert x.shape[1] == self.l, f"Expected input to have shape (B, {self.l}), got {x.shape}"

        stft_ = np.abs(librosa.stft(x.numpy(), n_fft=self.n_fft, hop_length=self.hop_length))
        mel = librosa.feature.melspectrogram(sr=sample_rate, S=stft_**2, n_mels=nmels)
        mel = torch.from_numpy(mel)

        self._assert_is_mel(mel, bs=x.shape[0])
        return mel

    def show_mel(self, mel: torch.Tensor, sample_rate: int):
        self._assert_is_mel(mel)
        fig, ax = plt.subplots()
        melspec = librosa.power_to_db(mel.detach().cpu().numpy()[0], ref=np.max)
        img = librosa.display.specshow(melspec, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis')
        cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig, ax

    def show_spec(self, z: torch.Tensor, sample_rate: int):
        self._assert_is_spectrogram(z)
        fig, ax = plt.subplots()
        spec = librosa.amplitude_to_db(z.abs().detach().cpu().numpy()[0], ref=np.max)
        img = librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='linear', cmap='viridis')
        cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig, ax


def make_stft(
    audio_length: int | None = None,
    n: int | None = None,
    t: int | None = None,
    hop_length: int | None = None,
    ensure_relations: bool = True,
) -> STFT:
    """Make a STFT object with the given parameters. If ensure_relations is True, we will try to enforce the relations between the parameters."""
    if n is None and t is None:
        # Give a random default to n and move on
        n = 513

    # If audio_length is not None and the other ones are not None
    if n is not None and t is not None:
        stft = STFT(n, t)
        if hop_length is not None:
            stft.hop_length = hop_length
        if ensure_relations and audio_length is not None:
            assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
        stft._skip_checks = not ensure_relations
        return stft

    assert audio_length is not None, "Audio length must be provided if exactly one of n or t are not provided"
    assert n is not None or t is not None, "Either n or t must be provided"

    if n is not None:
        if hop_length is None:
            hop_length = n // 4
        t = 1 + audio_length // hop_length
        stft = STFT(n, t)
        stft.hop_length = hop_length
        assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
        stft._skip_checks = not ensure_relations
        return stft

    assert t is not None
    assert audio_length % (t - 1) == 0, "The audio length must be divisible by t - 1"
    assert t > 1, "t must be greater than 1"

    if hop_length is None:
        hop_length = audio_length // (t - 1)

    # Try to make more overlap by default - set n to the previous power of 2 of 4 * hop_length and then + 1
    n = 1 << np.floor(np.log2(4 * hop_length)) + 1
    assert n is not None

    stft = STFT(n, t)
    stft.hop_length = hop_length
    assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
    stft._skip_checks = not ensure_relations
    return stft
