# Implements VGGish feature extractor for Audio
import torch
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
from AutoMasher.fyp import Audio
import torchaudio.functional as F
from torch import nn


class Vggish(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.model = VGGISH.get_model()

    def forward(self, audio: Audio | tuple[torch.Tensor, int]):
        if isinstance(audio, Audio):
            x = audio.resample(self.input_sr).data
        else:
            y, sr = audio
            if sr != self.input_sr:
                x = F.resample(y, sr, self.input_sr)
            else:
                x = y
        assert len(x.shape) in (2, 3), f"Expected 2D or 3D tensor, got {x.shape}"
        has_batch = len(x.shape) == 3
        if has_batch:
            b, c, t = x.shape
            x = x.view(-1, t)
        else:
            # To apease the typechecker
            b = 1
            c, t = x.shape
        feats = torch.stack([self.model(self.input_proc(w)) for w in x]).flatten(-2, -1)
        if has_batch:
            feats = feats.view(b, c, -1)
        return feats

    def __call__(self, audio: Audio | tuple[torch.Tensor, int]) -> torch.Tensor:
        return super().__call__(audio)
