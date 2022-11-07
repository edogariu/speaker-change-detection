import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as TA
import torchvision.transforms as TV

from utils import QUERY_DURATION, SAMPLE_RATE

class AudioPipeline(nn.Module):
    def __init__(
        self,
        input_len_s: float=QUERY_DURATION,
        n_mel: int=256,
        use_augmentation: bool=False,
        use_adaptive_rescaling: bool=False,  # whether to detect blank columns and rescale them away
        input_freq=SAMPLE_RATE,
        resample_freq=SAMPLE_RATE,
        n_fft=512,
    ):
        super().__init__()
        
        self.input_len_s = input_len_s
        self.resample_freq = resample_freq
        self.n_mel = n_mel
        self.use_aug = use_augmentation
        self.adaptive_rescale = use_adaptive_rescaling
        
        self.resample = TA.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = TA.Spectrogram(n_fft=n_fft)
        self.spec_aug = nn.Sequential(  # tiny tiny augmentation
            TA.TimeStretch(np.random.rand() * 0.1 + 0.95, fixed_rate=True),  # random time stretch in (0.95, 1.05)
            TA.FrequencyMasking(freq_mask_param=10),  # random masking up to `param` idxs
            TA.TimeMasking(time_mask_param=10),
        )
        self.mel_scale = TA.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        self.to_log = TA.AmplitudeToDB()
        self.resize = TV.Resize((n_mel, n_mel))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            resampled = self.resample(waveform.float())  # resample the input
            spec = self.spec(resampled)  # convert to power spectrogram
            # if self.use_aug and self.training: spec = self.spec_aug(spec)  # apply SpecAugment
            mel = self.mel_scale(spec)  # convert to mel scale
            mel = self.to_log(mel)  # convert to log-mel scale

            # resize to square (should be close to square anyway), but may contain empty columns on the right
            if self.adaptive_rescale:
                mel = torch.cat([self.resize(m[:, :mel.shape[2] - (m.std(dim=0) == 0).sum()].unsqueeze(0)) for m in mel], dim=0)  
            else: 
                mel = self.resize(mel) 
                
            # normalize
            z = mel.reshape(mel.shape[0], -1)
            mel -= z.mean(dim=1).reshape(-1, 1, 1)
            mel /= z.std(dim=1).reshape(-1, 1, 1)
            return mel  
        