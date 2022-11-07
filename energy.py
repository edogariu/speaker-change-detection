import pandas as pd
import numpy as np
import torch.utils.data as D
import torch
import torch.nn as nn
import os
import torchaudio.transforms as T
from scipy.io import wavfile
from collections import defaultdict

from trainer import Trainer
import architectures
from utils import exponential_linspace_int

QUERY_DURATION = 0.5

SPLIT_INTERVALS = {'train': 0.75,  # intervals for each split (note that split occurs by speakers here, since that is what we hope to generalize over)
                   'val': 0.2,
                   'test': 0.05}

class EnergyDataset(D.Dataset):
    def __init__(self, split: str, prob: float):
        """
        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        prob_same : float
            probability that paired datapoint comes from the same speaker
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        self.prob = prob
        
        # make dataframe describing dataset
        vctk_df = pd.DataFrame()
        paths = [p for p in np.sort(os.listdir('data/VCTK')) if '.wav' in p]
        vctk_df['paths'] = paths
        vctk_df['id_strs'] = [p.split('_')[0] for p in paths]
                      
        # get dataframe describing dataset
        vox_df = pd.read_csv('data/vox1/dataset.csv')
        
        # combine into one dataframe
        id_to_int = {}
        for id in vctk_df['id_strs']:
            id_to_int[id] = len(id_to_int)
        for id in vox_df['speaker']:
            id_to_int[str(id)] = len(id_to_int)

        combined = pd.DataFrame()
        ids = []
        paths = []
        srs = []
        for id, path in zip(vctk_df['id_strs'], vctk_df['paths']):
            ids.append(id_to_int[id])
            paths.append('data/VCTK/' + path)
            srs.append(8000)
        for d in vox_df.values:
            _, id, path, _ = d
            paths.append('data/vox1/' + path)
            ids.append(id_to_int[str(id)])
            srs.append(16000)
        combined['id'] = ids
        combined['path'] = paths
        combined['sr'] = srs
        self.df = combined
            
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
        
        self.speakers_to_rows = defaultdict(list)
        for i in self.idxs:
            datapoint = self.df.iloc[i]
            self.speakers_to_rows[datapoint['id']].append(i)
        self.length = len(self.idxs)
        
        self.resample = T.Resample(16000, 8000)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        d = self.df.iloc[self.idxs[index]]
        label = d['id']
        path = d['path']
        sr = d['sr']
        _sr, query = wavfile.read(path)
        assert sr == _sr
        
        query = torch.tensor(query)
        if sr == 16000:   # if this is a vox datapoint
            query = (query / 2 ** 15).float()
            query = self.resample(query)
        rand_index = np.random.randint(query.shape[0] - int(QUERY_DURATION * 8000))  # pick random starting point
        query = query[rand_index: int(QUERY_DURATION * 8000) + rand_index].float()

        if np.random.rand() < self.prob:  # if we want to yield a pair from the same speaker
            speaker = self.speakers_to_rows[label]
            datapoint = self.df.iloc[speaker[np.random.randint(len(speaker))]]
            assert datapoint['id'] == label
        else:
            datapoint = self.df.iloc[self.idxs[np.random.randint(self.length)]]
        
        other_label = datapoint['id']
        other_path = datapoint['path']
        other_sr = datapoint['sr']
        _sr, other = wavfile.read(other_path)
        assert other_sr == _sr
        other = torch.tensor(other)
        if other_sr == 16000:  # if this is a vox datapoint
            other = (other / 2 ** 15).float()
            other = self.resample(other)
        rand_index = np.random.randint(other.shape[0] - int(QUERY_DURATION * 8000))  # pick random starting point
        other = other[rand_index: int(QUERY_DURATION * 8000) + rand_index].float()

        return (other, query), float(label == other_label)
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class EnergyModel(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 body_type: str,
                 pooling_type: str,
                 
                 mel_size: int,
                 depth: int,
                 nchan: int,
                 pool_every: int,
                 pool_size: int,
                 
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
                    'mel_size': mel_size,
                    'depth': depth,
                    'nchan': nchan,
                    'pool_every': pool_every,
                    'pool_size': pool_size,
                    'body_depth': body_depth}
        
        from pipeline import AudioPipeline
        self.pipe = AudioPipeline(input_len_s=QUERY_DURATION, n_mel=mel_size, use_augmentation=True)

        # head to inference over query spec via 2D convolutions
        channels = exponential_linspace_int(start=1, end=nchan, num=depth + 1)
        self.head = [nn.Unflatten(1, (1, mel_size))]
        for i in range(depth):
            in_chan, out_chan = channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), bias=False)]
            if (i + 1) % pool_every == 0:
                layer.append(pools[pooling_type](out_chan, pool_size))
            layer.append(nn.ReLU())
            self.head.extend(layer)
        self.head.append(nn.Flatten(1, 3))
        self.head = nn.Sequential(*self.head)
        
        # figure out conv tower output dims and project
        test = torch.zeros(3, mel_size, mel_size)
        out_dim = self.head(test).shape[1]  # find output dim of conv tower
        mid_dim = (out_dim + self.hidden_dim) // 2  # find middle dim for projection
        self.head.append(nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(), 
                                        #  nn.Linear(query_mid_dim, query_mid_dim), nn.ReLU(), 
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), 
                                        nn.Linear(mid_dim, self.hidden_dim), nn.ReLU()))
        del test
        
        # use body to bring down to final output dimension
        body_dims = exponential_linspace_int(2 * self.hidden_dim, 1, body_depth + 1)
        self.body = [nn.BatchNorm1d(2 * self.hidden_dim), nn.Dropout(0.2)]
        for i in range(body_depth):
            in_dim, out_dim = body_dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.body.extend(layer)
        self.body[-1] = nn.Flatten(0)  # remove last ReLU
        self.body = nn.Sequential(*self.body)

    def forward(self, q1, q2):
        # get resampled waveform and spectrograms
        with torch.no_grad():
            self.pipe.eval()
            q1 = self.pipe(q1)  
            q2 = self.pipe(q2)

        # uncomment this to view spectrograms
        # from librosa import display; import matplotlib.pyplot as plt; display.specshow(sc.cpu().detach().numpy()[0], x_axis='time', y_axis='log', sr=SAMPLE_RATE); plt.show(); exit(0) 
        # import matplotlib.pyplot as plt; plt.imshow(q1.cpu().detach().numpy()[0]); plt.show(); exit(0)

        # get output from spectrogram heads
        q1 = self.head(q1)  
        q2 = self.head(q2)

        # project to final dimension
        cat = torch.cat((q1, q2), dim=-1)
        out = self.body(cat)  
        return out
    
    @staticmethod
    def load(model_path: str):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path, map_location='cpu')
        model = EnergyModel(**params['args'])
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
        
if __name__ == '__main__':
    model_name = 'energy'
    batch_size = 512
    trainer_args = {'initial_lr': 0.04,
                    'lr_decay_period': 1,
                    'lr_decay_gamma': 0.6,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 60,
                    'eval_every': 1,
                    'patience': 3,
                    'num_tries': 4}

    model_args = {'hidden_dim': 64,
                  'body_type': 'linear',
                  'pooling_type': 'max',
                  'depth': 6,
                  'mel_size': 128,
                  'nchan': 256,
                  'pool_every': 1,
                  'pool_size': 2,
                  'body_depth': 4}

    train_dataloader = EnergyDataset('train', 0.5).get_dataloader(batch_size)
    val_dataloader = EnergyDataset('val', 0.5).get_dataloader(batch_size)

    model = EnergyModel(**model_args)
    
    t = Trainer(model_name, model, train_dataloader, val_dataloader, criterion=nn.BCEWithLogitsLoss(), **trainer_args)
    t.train(**train_args)
    