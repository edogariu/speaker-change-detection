import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TV
import torchaudio.transforms as TA

from utils import exponential_linspace_int, DEVICE, QUERY_DURATION, CONTEXT_DURATION, SAMPLE_RATE
import architectures


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
        self.spec = TA.Spectrogram(n_fft=n_fft, win_length=int(n_fft / 2.5), power=2, hop_length=int(input_len_s * resample_freq / n_mel))
        self.spec_aug = torch.nn.Sequential(  # tiny tiny augmentation
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
            return ((mel + 100) / 130)  # rescale the values

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
        
        self.pipe = AudioPipeline(input_len_s=QUERY_DURATION, n_mel=256, use_augmentation=True, use_adaptive_rescaling=False)
        
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
        test_s = self.pipe(torch.zeros(1, self.in_dim))
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
        s = self.pipe(x)  # get resampled waveform and spectrograms
        # from librosa import display; import matplotlib.pyplot as plt; display.specshow(s.cpu().detach().numpy()[0], x_axis='time', y_axis='log'); plt.show(); exit(0)  # view spectrograms
        # import matplotlib.pyplot as plt; plt.imshow(s.cpu().detach().numpy()[0]); plt.show(); exit(0)
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

class SpeakerEnergy(nn.Module):
    def __init__(self,
                 in_dim: int,
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
        self.hidden_dim = hidden_dim
        
        self.args = {'in_dim': in_dim,
                    'hidden_dim': hidden_dim,
                    'body_type': body_type,
                    'pooling_type': pooling_type,
                    'spec_depth': spec_depth,
                    'spec_nchan': spec_nchan,
                    'spec_pool_every': spec_pool_every,
                    'spec_pool_size': spec_pool_size,
                    'body_depth': body_depth}
        
        self.pipe = AudioPipeline()
        
        # first head        
        # head to inference over spectrogram via 2D convolutions
        spec_channels = exponential_linspace_int(start=1, end=spec_nchan, num=spec_depth + 1)
        self.spec_head_1 = [nn.Unflatten(1, (1, self.pipe.n_mel))]
        for i in range(spec_depth):
            in_chan, out_chan = spec_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3))]
            if (i + 1) % spec_pool_every == 0:
                layer.append(spec_pools[pooling_type](out_chan, spec_pool_size))
            layer.append(nn.ReLU())
            self.spec_head_1.extend(layer)
        self.spec_head_1.append(nn.Flatten(1, 3))
        self.spec_head_1 = nn.Sequential(*self.spec_head_1)
        
        # second head        
        # head to inference over spectrogram via 2D convolutions
        spec_channels = exponential_linspace_int(start=1, end=spec_nchan, num=spec_depth + 1)
        self.spec_head_2 = [nn.Unflatten(1, (1, self.pipe.n_mel))]
        for i in range(spec_depth):
            in_chan, out_chan = spec_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3))]
            if (i + 1) % spec_pool_every == 0:
                layer.append(spec_pools[pooling_type](out_chan, spec_pool_size))
            layer.append(nn.ReLU())
            self.spec_head_2.extend(layer)
        self.spec_head_2.append(nn.Flatten(1, 3))
        self.spec_head_2 = nn.Sequential(*self.spec_head_2)
        
        # figure out conv tower output dims and project
        test_s = self.pipe(torch.zeros(1, self.in_dim))
        spec_out_dim = self.spec_head_1(test_s).shape[1]  # find output dim of conv tower
        spec_mid_dim = (spec_out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.spec_head_1.append(nn.Sequential(nn.Linear(spec_out_dim, spec_mid_dim), nn.ReLU(), nn.Linear(spec_mid_dim, self.hidden_dim), nn.ReLU()))
        spec_out_dim = self.spec_head_2(test_s).shape[1]  # find output dim of conv tower
        spec_mid_dim = (spec_out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.spec_head_2.append(nn.Sequential(nn.Linear(spec_out_dim, spec_mid_dim), nn.ReLU(), nn.Linear(spec_mid_dim, self.hidden_dim), nn.ReLU()))
        del test_s
        
        # use body to bring down to final output dimension
        body_dims = exponential_linspace_int(2 * self.hidden_dim, 1, body_depth + 1)
        self.body = [nn.BatchNorm1d(2 * self.hidden_dim), nn.Dropout(0.2)]
        for i in range(body_depth):
            in_dim, out_dim = body_dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.body.extend(layer)
        self.body[-1] = nn.Flatten(0)  # remove last ReLU
        self.body = nn.Sequential(*self.body)

    def forward(self, x1, x2):
        # get resampled waveform and spectrograms
        s1 = self.pipe(x1)  
        s2 = self.pipe(x2)
        
        # from librosa import display; import matplotlib.pyplot as plt; display.specshow(s.detach().numpy()[0], x_axis='time', y_axis='log'); plt.show()  # view spectrograms
        
        # get output from spectrogram head
        s1 = self.spec_head_1(s1)  
        s2 = self.spec_head_2(s2)  
        cat = torch.cat((s1, s2), dim=-1)
        out = self.body(cat)  # project to embedding dimension
        return out
    
    @staticmethod
    def load(model_path: str):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path)
        model = SpeakerEnergy(**params['args'])
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
    
class SpeakerContextModel(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 body_type: str,
                 pooling_type: str,
                 
                 context_mel_size: int,
                 context_depth: int,
                 context_nchan: int,
                 context_pool_every: int,
                 context_pool_size: int,
                 
                 query_mel_size: int,
                 query_depth: int,
                 query_nchan: int,
                 query_pool_every: int,
                 query_pool_size: int,
                 
                 body_depth: int
                 ):
        super().__init__()
        
        assert body_type in ['linear', 'transformer']
        assert pooling_type in ['max', 'average', 'attention'] 
        
        pools = {'max': architectures.MaxPool2D,
                'average': architectures.AvgPool2D,
                'attention': architectures.AttentionPool2D}
        
        self.hidden_dim = hidden_dim
        
        self.args = {'hidden_dim': hidden_dim,
                    'body_type': body_type,
                    'pooling_type': pooling_type,
                    'context_mel_size':  context_mel_size,
                    'context_depth': context_depth,
                    'context_nchan': context_nchan,
                    'context_pool_every': context_pool_every,
                    'context_pool_size': context_pool_size,
                    'query_mel_size': query_mel_size,
                    'query_depth': query_depth,
                    'query_nchan': query_nchan,
                    'query_pool_every': query_pool_every,
                    'query_pool_size': query_pool_size,
                    'body_depth': body_depth}
        
        from pipeline import AudioPipeline
        self.context_pipe = AudioPipeline(input_len_s=CONTEXT_DURATION, n_mel=context_mel_size, use_augmentation=False, use_adaptive_rescaling=True)
        self.query_pipe = AudioPipeline(input_len_s=QUERY_DURATION, n_mel=query_mel_size, use_augmentation=True)

        # head to inference over context spec via 2D convolutions
        context_channels = exponential_linspace_int(start=1, end=context_nchan, num=context_depth + 1)
        self.context_head = [nn.Unflatten(1, (1, context_mel_size))]
        for i in range(context_depth):
            in_chan, out_chan = context_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), bias=False)]
            if (i + 1) % context_pool_every == 0:
                layer.append(pools[pooling_type](out_chan, context_pool_size))
            layer.append(nn.ReLU())
            self.context_head.extend(layer)
        self.context_head.append(nn.Flatten(1, 3))
        self.context_head = nn.Sequential(*self.context_head)
        
        # head to inference over query spec via 2D convolutions
        query_channels = exponential_linspace_int(start=1, end=query_nchan, num=query_depth + 1)
        self.query_head = [nn.Unflatten(1, (1, query_mel_size))]
        for i in range(query_depth):
            in_chan, out_chan = query_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), bias=False)]
            if (i + 1) % query_pool_every == 0:
                layer.append(pools[pooling_type](out_chan, query_pool_size))
            layer.append(nn.ReLU())
            self.query_head.extend(layer)
        self.query_head.append(nn.Flatten(1, 3))
        self.query_head = nn.Sequential(*self.query_head)
        
        # figure out conv tower output dims and project
        test_c, test_q = torch.zeros(3, context_mel_size, context_mel_size), torch.zeros(3, query_mel_size, query_mel_size)
        context_out_dim = self.context_head(test_c).shape[1]  # find output dim of context conv tower
        context_mid_dim = (context_out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.context_head.append(nn.Sequential(nn.Linear(context_out_dim, context_mid_dim), nn.ReLU(), 
                                            #    nn.Linear(context_mid_dim, context_mid_dim), nn.ReLU(), 
                                            #    nn.Linear(context_mid_dim, context_mid_dim), nn.ReLU(), 
                                               nn.Linear(context_mid_dim, context_mid_dim), nn.ReLU(), 
                                               nn.Linear(context_mid_dim, self.hidden_dim), nn.ReLU()))
        query_out_dim = self.query_head(test_q).shape[1]  # find output dim of query conv tower
        query_mid_dim = (query_out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.query_head.append(nn.Sequential(nn.Linear(query_out_dim, query_mid_dim), nn.ReLU(), 
                                            #  nn.Linear(query_mid_dim, query_mid_dim), nn.ReLU(), 
                                             nn.Linear(query_mid_dim, query_mid_dim), nn.ReLU(), 
                                             nn.Linear(query_mid_dim, self.hidden_dim), nn.ReLU()))
        del test_c, test_q
        
        # use body to bring down to final output dimension
        body_dims = exponential_linspace_int(2 * self.hidden_dim, 1, body_depth + 1)
        self.body = [nn.BatchNorm1d(2 * self.hidden_dim), nn.Dropout(0.2)]
        for i in range(body_depth):
            in_dim, out_dim = body_dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.body.extend(layer)
        self.body[-1] = nn.Flatten(0)  # remove last ReLU
        self.body = nn.Sequential(*self.body)

    def forward(self, context, query):
        # get resampled waveform and spectrograms
        with torch.no_grad():
            sc = self.context_pipe.eval()(context)  
            sq = self.query_pipe.eval()(query)

        # uncomment this to view spectrograms
        # from librosa import display; import matplotlib.pyplot as plt; display.specshow(sc.cpu().detach().numpy()[0], x_axis='time', y_axis='log', sr=SAMPLE_RATE); plt.show(); exit(0) 
        # import matplotlib.pyplot as plt; plt.imshow(sq.cpu().detach().numpy()[0]); plt.show(); exit(0)

        # get output from spectrogram heads
        sc = self.context_head(sc)  
        sq = self.query_head(sq)  

        # project to final dimension
        cat = torch.cat((sc, sq), dim=-1)
        out = self.body(cat)  
        return out
    
    @staticmethod
    def load(model_path: str):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path, map_location='cpu')
        model = SpeakerContextModel(**params['args'])
        model.load_state_dict(params['state_dict'], strict=False)
        return model

    def save(self, path: str):
        """ 
        Save the model to a file.
        """

        params = {
            'args': self.args,   # args to remake the model object
            'state_dict': self.state_dict()   # the model params
        }
        torch.save(params, path)
        