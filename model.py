import torch
import torch.nn as nn
import torchaudio.transforms as T

from utils import exponential_linspace_int, count_parameters
import architectures


class AudioPipeline(nn.Module):
    def __init__(
        self,
        input_len_s=1.,
        input_freq=8000,
        resample_freq=8000,
        n_fft=2048,
        n_mel=128,
        stretch_factor=1.0,
    ):
        super().__init__()
        
        self.input_len_s = input_len_s
        self.resample_freq = resample_freq
        self.n_mel = n_mel
        
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = T.Spectrogram(n_fft=n_fft, power=2, hop_length=int(n_mel * input_len_s / (2 * input_freq / resample_freq)))
        self.spec_aug = torch.nn.Sequential(
            T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=80),
        )
        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        self.to_log = T.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)  # resample the input
        spec = self.spec(resampled)  # convert to power spectrogram
        # spec = self.spec_aug(spec)  # apply SpecAugment
        mel = self.mel_scale(spec)  # convert to mel scale
        mel = self.to_log(mel)  # convert to log-mel scale
        return resampled, mel

class SpeakerEmbedding(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 body_type: str,
                 pooling_type: str,
                 
                 spec_depth: int,
                 spec_nchan: int,
                 spec_pool_every: int,
                 spec_pool_size: int,
                 
                 body_depth: int
                 ):
        super().__init__()
        
        assert body_type in ['linear', 'transformer']
        assert pooling_type in ['max', 'average', 'attention'] 
        
        spec_pools = {'max': architectures.MaxPool2D,
                      'average': architectures.AvgPool2D,
                      'attention': architectures.AttentionPool2D}
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        self.args = {'in_dim': in_dim,
                    'out_dim': out_dim,
                    'hidden_dim': hidden_dim,
                    'body_type': body_type,
                    'pooling_type': pooling_type,
                    'spec_depth': spec_depth,
                    'spec_nchan': spec_nchan,
                    'spec_pool_every': spec_pool_every,
                    'spec_pool_size': spec_pool_size,
                    'body_depth': body_depth}
        
        self.pipe = AudioPipeline()
        
        # head to inference over spectrogram via 2D convolutions
        spec_channels = exponential_linspace_int(start=1, end=spec_nchan, num=spec_depth + 1)
        self.spec_head = [nn.Unflatten(1, (1, self.pipe.n_mel))]
        for i in range(spec_depth):
            in_chan, out_chan = spec_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3))]
            if (i + 1) % spec_pool_every == 0:
                layer.append(spec_pools[pooling_type](out_chan, spec_pool_size))
            layer.append(nn.ReLU())
            self.spec_head.extend(layer)
        self.spec_head.append(nn.Flatten(1, 3))
        self.spec_head = nn.Sequential(*self.spec_head)
        
        # figure out conv tower output dims and project
        _, test_s = self.pipe(torch.zeros(1, self.in_dim))
        spec_out_dim = self.spec_head(test_s).shape[1]; del test_s  # find output dim of conv tower
        spec_mid_dim = (spec_out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.spec_head.append(nn.Sequential(nn.Linear(spec_out_dim, spec_mid_dim), nn.ReLU(), nn.Linear(spec_mid_dim, self.hidden_dim), nn.ReLU()))
        
        # use body to bring down to final output dimension
        body_dims = exponential_linspace_int(self.hidden_dim, self.out_dim, body_depth + 1)
        self.body = [nn.BatchNorm1d(self.hidden_dim), nn.Dropout(0.2)]
        for i in range(body_depth):
            in_dim, out_dim = body_dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.body.extend(layer)
        self.body.pop()  # remove last ReLU
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        _, s = self.pipe(x)  # get resampled waveform and spectrograms
        # from librosa import display; import matplotlib.pyplot as plt; display.specshow(s.detach().numpy()[0], x_axis='time', y_axis='log'); plt.show()  # view spectrograms
        s = self.spec_head(s)  # get output from spectrogram head
        out = self.body(s)  # project to embedding dimension
        return out
    
    @staticmethod
    def load(model_path: str):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path)
        model = SpeakerEmbedding(**params['args'])
        model.pipe = params['pipe']
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ 
        Save the model to a file.
        """

        params = {
            'args': self.args,   # args to remake the model object
            'pipe': self.pipe,   # the vocab object
            'state_dict': self.state_dict()   # the model params
        }
        torch.save(params, path)
