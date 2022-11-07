import pandas as pd
import numpy as np
import torch.utils.data as D
from scipy.io import wavfile

from utils import CONTEXT_DURATION, QUERY_DURATION, SAMPLE_RATE

SPLIT_INTERVALS = {'train': 0.8,  # intervals for each split (note that split occurs by speakers here, since that is what we hope to generalize over)
                   'val': 0.2,
                   'test': 0.00}

import os
class VCTKDataset(D.Dataset):
    def __init__(self, split: str):
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        # make dataframe describing dataset
        self.df = pd.DataFrame()
        paths = [p for p in np.sort(os.listdir('data/VCTK')) if '.wav' in p]
        self.df['paths'] = paths
        self.df['id_strs'] = [p.split('_')[0] for p in paths]
        id_to_int = {}
        for id in self.df['id_strs']:
            id_to_int[id] = len(id_to_int)
        self.df['ids'] = [id_to_int[id] for id in self.df['id_strs']]
            
        # split the dataset the same way every time
        np.random.seed(0)
        rand_idxs = np.random.permutation(len(self.df))
        val_idx = int(len(rand_idxs) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(rand_idxs) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        self.idxs = {'train': train_idxs,
                     'val': val_idxs,
                     'test': test_idxs,
                     'all': rand_idxs}[split]
        np.random.seed()

        self.length = len(self.idxs)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        idx = self.idxs[index]
        
        datapoint = self.df.iloc[idx]
        path = datapoint['paths']
        label = datapoint['ids']
        _, inp = wavfile.read(f'data/VCTK/{path}')
        
        idx_range = len(inp) - int(QUERY_DURATION * SAMPLE_RATE)
        rand_index = np.random.randint(idx_range)
        inp = inp[rand_index: int(QUERY_DURATION * SAMPLE_RATE) + rand_index].astype(float)
        
        return (inp,), label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)


class VoxDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of `.wav` clips from VoxCeleb1 dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        # get dataframe describing dataset
        self.df = pd.read_csv('data/vox1/dataset.csv')
        
        self.speaker_to_idx = {}
        for s in np.unique(self.df['speaker']):
            self.speaker_to_idx[s] = len(self.speaker_to_idx)
            
        # split the dataset the same way every time
        np.random.seed(0)
        rand_idxs = np.random.permutation(len(self.df))
        val_idx = int(len(rand_idxs) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(rand_idxs) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        self.idxs = {'train': train_idxs,
                     'val': val_idxs,
                     'test': test_idxs,
                     'all': rand_idxs}[split]
        np.random.seed()

        self.length = len(self.idxs)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        idx = self.idxs[index]
        
        datapoint = self.df.iloc[idx]
        path = datapoint['path']
        label = self.speaker_to_idx[datapoint['speaker']]
        
        _, inp = wavfile.read(f'data/vox1/{path}')
        
        idx_range = len(inp) - int(QUERY_DURATION * SAMPLE_RATE)
        rand_index = np.random.randint(idx_range)
        inp = inp[rand_index: int(QUERY_DURATION * SAMPLE_RATE) + rand_index]
        
        inp = (inp / 2 ** 15).astype(float)
        return (inp,), label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torch
import torch.nn as nn
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
            return mel  # rescale the values

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import exponential_linspace_int

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, chan, stride=1):
        super(ResBlock, self).__init__()
        self.in_conv = nn.Conv2d(chan, chan, kernel_size=(3, 3), stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(chan, chan, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(chan)

    def forward(self, x):
        h = self.in_conv(x)
        h = self.relu(h)
        h = self.out_conv(h)
        h = self.bn(h)
        h = self.relu(h + x)  # residual connection
        return h

class VoxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        n_layers = 2
        n_mel = 64
        n_classes = 111
        n_chan = 128
        self.args = {}
        
        self.pipe = AudioPipeline(n_mel=n_mel, n_fft=1024)
        
        channels = np.rint(np.linspace(1, n_chan, n_layers + 1)).astype(int)
        self.conv_tower = [nn.Unflatten(1, (1, n_mel)),]
        for i in range(n_layers):
            in_chan, out_chan = channels[i: i + 2]
            self.conv_tower.extend([nn.Conv2d(in_chan, out_chan, kernel_size=(5, 5), stride=2), nn.ReLU(), nn.BatchNorm2d(out_chan), ResBlock(out_chan)])
        self.conv_tower = nn.Sequential(*self.conv_tower, nn.AdaptiveAvgPool2d((1, None)), nn.Flatten(1))
        
        test_t = torch.zeros(1, n_mel, n_mel)
        conv_tower_out_dim = self.conv_tower(test_t).shape[1]
        del test_t
        
        self.fc = nn.Linear(conv_tower_out_dim, n_classes)
    
    def forward(self, x):
        x = self.pipe(x)
        x = self.conv_tower(x)
        x = self.fc(x)
        return x
    
    @staticmethod
    def load(model_path: str):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path)
        model = VoxModel(**params['args'])
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

     
from trainer import Trainer 
        
model_name = 'shit'
batch_size = 512
trainer_args = {'initial_lr': 0.04,
                'lr_decay_period': 1,
                'lr_decay_gamma': 0.5,
                'weight_decay': 0.0002}
train_args = {'num_epochs': 60,
                'eval_every': 1,
                'patience': 3,
                'num_tries': 4}
train_dataloader = VCTKDataset('train').get_dataloader(batch_size)
val_dataloader = VCTKDataset('val').get_dataloader(batch_size)

model = VoxModel()
t = Trainer(model_name, model, train_dataloader, val_dataloader, **trainer_args)
t.train(**train_args)

