import pandas as pd
import numpy as np
import torch.utils.data as D
import os
import torch
import torch.nn as nn
from scipy.io import wavfile
from collections import defaultdict

from trainer import Trainer
import pytorch_metric_learning.losses as pml
from losses import TripletMarginLoss
import architectures
from pipeline import AudioPipeline
from utils import SPLIT_INTERVALS

QUERY_DURATION = 0.5

# -----------------------------------------------------------------------------------------------
# ------------------------- DATASETS FOR INITIAL SPEAKER CLASSIFIER TRAINING --------------------
# -----------------------------------------------------------------------------------------------

# This step of the process is to train a model to classify each speaker with an embedding model with
# a linear classifier head. We will rip the head off for the next step to get good embedding models.

class VCTKClassifierDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of `.wav` clips from VCTK dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        # make dataframe describing dataset
        self.df = pd.DataFrame()
        paths = [p for p in np.sort(os.listdir('data/VCTK')) if '.wav' in p]
        self.df['path'] = paths
        self.df['id_str'] = [p.split('_')[0] for p in paths]
        id_to_int = {}
        for id in self.df['id_str']:
            id_to_int[id] = len(id_to_int)
        self.df['id'] = [id_to_int[id] for id in self.df['id_str']]
            
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
        label = datapoint['id']
        _, inp = wavfile.read(f'data/VCTK/{path}')
        
        idx_range = len(inp) - int(QUERY_DURATION * 8000)
        rand_index = np.random.randint(idx_range)
        inp = inp[rand_index: int(QUERY_DURATION * 8000) + rand_index].astype(float)
        
        return (inp,), label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class VoxClassifierDataset(D.Dataset):
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
        
        idx_range = len(inp) - int(QUERY_DURATION * 16000)
        rand_index = np.random.randint(idx_range)
        inp = inp[rand_index: int(QUERY_DURATION * 16000) + rand_index]
        
        inp = (inp / 2 ** 15).astype(float)
        return (inp,), label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    

# -----------------------------------------------------------------------------------------------
# ------------------------- DATASETS FOR CONTRASTIVE EMBEDDING TRAINING -------------------------
# -----------------------------------------------------------------------------------------------
    
class VCTKTripletDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of pairs of 1 second `.wav` clips from filtered VCTK dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        # make dataframe describing dataset
        self.df = pd.DataFrame()
        paths = [p for p in np.sort(os.listdir('data/VCTK')) if '.wav' in p]
        self.df['path'] = paths
        self.df['id_str'] = [p.split('_')[0] for p in paths]
        id_to_int = {}
        for id in self.df['id_str']:
            id_to_int[id] = len(id_to_int)
        self.df['id'] = [id_to_int[id] for id in self.df['id_str']]
        
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
        
        self.speakers_to_paths = defaultdict(list)
        for i in self.idxs:
            datapoint = self.df.iloc[i]
            self.speakers_to_paths[datapoint['id']].append(datapoint['path'])
        self.length = len(self.idxs)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        idx = self.idxs[index]
        
        # get anchor
        datapoint = self.df.iloc[idx]
        path = datapoint['path']
        label = datapoint['id']
        _, anchor = wavfile.read(f'data/VCTK/{path}')
        idx_range = len(anchor) - int(QUERY_DURATION * 8000)
        rand_index = np.random.randint(idx_range)
        anchor = anchor[rand_index: int(QUERY_DURATION * 8000) + rand_index].astype(float)

        # get pos pair
        speaker = self.speakers_to_paths[label]
        rand_path = speaker[np.random.randint(len(speaker))]
        _, pos = wavfile.read(f'data/VCTK/{rand_path}')
        idx_range = len(pos) - int(QUERY_DURATION * 8000)
        rand_index = np.random.randint(idx_range)
        pos = pos[rand_index: int(QUERY_DURATION * 8000) + rand_index].astype(float)
        
        # get neg pair
        speaker_list = list(self.speakers_to_paths.keys())
        speaker = self.speakers_to_paths[speaker_list[np.random.randint(len(speaker_list))]]
        rand_path = speaker[np.random.randint(len(speaker))]
        _, neg = wavfile.read(f'data/VCTK/{rand_path}')
        idx_range = len(neg) - int(QUERY_DURATION * 8000)
        rand_index = np.random.randint(idx_range)
        neg = neg[rand_index: int(QUERY_DURATION * 8000) + rand_index].astype(float)
        return (anchor, pos, neg), -1
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class VoxTripletDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of pairs of 1 second `.wav` clips from VoxCeleb dataset.

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
        
        self.speakers_to_paths = defaultdict(list)
        for i in self.idxs:
            datapoint = self.df.iloc[i]
            self.speakers_to_paths[self.speaker_to_idx[datapoint['speaker']]].append(datapoint['path'])

        self.length = len(self.idxs)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        idx = self.idxs[index]
        
        # get anchor
        datapoint = self.df.iloc[idx]
        path = datapoint['path']
        label = self.speaker_to_idx[datapoint['speaker']]
        _, anchor = wavfile.read(f'data/vox1/{path}')
        idx_range = len(anchor) - int(QUERY_DURATION * 16000)
        rand_index = np.random.randint(idx_range)
        anchor = anchor[rand_index: int(QUERY_DURATION * 16000) + rand_index]
        anchor = (anchor / 2 ** 15).astype(float)

        # get pos pair
        speaker = self.speakers_to_paths[label]
        rand_path = speaker[np.random.randint(len(speaker))]
        _, pos = wavfile.read(f'data/vox1/{rand_path}')
        idx_range = len(pos) - int(QUERY_DURATION * 16000)
        rand_index = np.random.randint(idx_range)
        pos = pos[rand_index: int(QUERY_DURATION * 16000) + rand_index].astype(float)
        pos = (pos / 2 ** 15).astype(float)
        
        # get neg pair
        speaker_list = list(self.speakers_to_paths.keys())
        speaker = self.speakers_to_paths[speaker_list[np.random.randint(len(speaker_list))]]
        rand_path = speaker[np.random.randint(len(speaker))]
        _, neg = wavfile.read(f'data/vox1/{rand_path}')
        idx_range = len(neg) - int(QUERY_DURATION * 16000)
        rand_index = np.random.randint(idx_range)
        neg = neg[rand_index: int(QUERY_DURATION * 16000) + rand_index].astype(float)
        neg = (neg / 2 ** 15).astype(float)

        return (anchor, pos, neg), -1
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class VoxContrastiveDataloader():
    def __init__(self, split: str, batch_size: int):
        """
        Creates dataset of pairs of 1 second `.wav` clips from VoxCeleb dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
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
        
        self.i = 0
        self.num_batches_yielded = 0
        self.batch_size = batch_size
        
        self.length = len(self.idxs)
        
    def __len__(self):
        return self.length // self.batch_size
    
    def __iter__(self):
        self.i = 0
        self.num_batches_yielded = 0
        np.random.shuffle(self.idxs)
        return self
    
    def __next__(self):
        idxs = self.idxs[self.i: self.i + self.batch_size]

        if self.num_batches_yielded >= len(self) or len(idxs) < self.batch_size: 
            raise StopIteration

        count = 0
        
        datapoints = self.df.iloc[idxs]
        batch_x = []
        batch_y = []
        for datapoint in datapoints.values:
            _, label, path, length = datapoint
            label = self.speaker_to_idx[label]
            sr, anchor = wavfile.read(f'data/vox1/{path}')
            assert sr == 16000
            anchor = np.split(anchor, np.arange(0, len(anchor), int(QUERY_DURATION * sr)))[1:-1]
            assert len(anchor) == np.floor(length / QUERY_DURATION)
            for a in anchor[:self.batch_size // 3]:  # make sure one clip never takes up more than a third of the batch
                if np.std(a) < 200: continue  # filter out silence
                batch_x.append((a / 2 ** 15).astype(float))
                batch_y.append(label)
            count += 1
            if len(batch_y) > self.batch_size: break
        
        self.i += count
        self.num_batches_yielded += 1
        batch_x = np.stack(batch_x, axis=0)
        batch_y = np.stack(batch_y, axis=0)

        return (torch.tensor(batch_x),), torch.tensor(batch_y)

# -----------------------------------------------------------------------------------------------
# --------------------------------------- THE MODEL ---------------------------------------------
# -----------------------------------------------------------------------------------------------

class Embedding(nn.Module):
    def __init__(self,
                 in_freq: int,
                 n_layers: int,
                 n_mel: int,
                 emb_dim: int,
                 n_chan: int,
                 n_classes: int):
        super().__init__()
        
        self.emb_mode = False
        
        self.args = {'in_freq': in_freq,
                     'n_layers': n_layers,
                     'n_mel': n_mel,
                     'emb_dim': emb_dim,
                     'n_classes': n_classes,
                     'n_chan': n_chan}
        
        self.pipe = AudioPipeline(n_mel=n_mel, n_fft=1024, input_freq=in_freq, resample_freq=8000)
        
        channels = np.rint(np.linspace(1, n_chan, n_layers + 1)).astype(int)
        self.conv_tower = [nn.Unflatten(1, (1, n_mel)),]
        for i in range(n_layers):
            in_chan, out_chan = channels[i: i + 2]
            self.conv_tower.extend([nn.Conv2d(in_chan, out_chan, kernel_size=(5, 5), stride=2, bias=False), nn.ReLU(), nn.BatchNorm2d(out_chan), architectures.ResBlock(out_chan)])
        self.conv_tower = nn.Sequential(*self.conv_tower, nn.AdaptiveAvgPool2d((1, None)), nn.Flatten(1))
        
        test_t = torch.zeros(1, n_mel, n_mel)
        conv_tower_out_dim = self.conv_tower(test_t).shape[1]
        del test_t
        
        self.conv_tower.append(nn.Linear(conv_tower_out_dim, emb_dim))
        self.fc = nn.Linear(emb_dim, n_classes)
    
    def toggle_emb_mode(self):
        """
        Toggle embedding mode
        """
        self.emb_mode = not self.emb_mode
        s = 'on' if self.emb_mode else 'off'
        print(f'Turned embedding mode {s}!')
        return self
    
    def _classify_forward(self, x):
        x = self.pipe(x)
        x = self.conv_tower(x)
        x = self.fc(x)
        return x
    
    def _triplet_forward(self, anchor, pos, neg):
        anchor = self.emb(anchor)
        pos = self.emb(pos)
        neg = self.emb(neg)
        return (anchor, pos, neg)
    
    def emb(self, x):
        x = self.pipe(x)
        x = self.conv_tower(x)
        return x
    
    def forward(self, *x):
        if self.emb_mode: return self.emb(*x)# self._triplet_forward(*x)
        else: return self._classify_forward(*x)
    
    @staticmethod
    def load(model_path: str, **kwargs):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path, map_location='cpu')
        model = Embedding(**params['args'], **kwargs)
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
        
if __name__ == '__main__':
    
    dataset_name = 'Vox'; assert dataset_name in ['VCTK', 'Vox']
    mode = 'embedding'; assert mode in ['classifier', 'embedding']
        
    model_name = f'{dataset_name.lower()}_emb'
    batch_size = 512
    trainer_args = {'initial_lr': 0.016,
                    'lr_decay_period': 3,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 60,
                    'eval_every': 1,
                    'patience': 3,
                    'num_tries': 4}

    model_args = {'n_layers': 3,
                  'n_mel': 86,
                  'emb_dim': 256 if dataset_name == 'VCTK' else 256,
                  'n_classes': 111 if dataset_name == 'VCTK' else 1211,
                  'n_chan': 256,
                  'in_freq': 8000 if dataset_name == 'VCTK' else 16000}

    # if mode == 'classifier': dataset = VCTKClassifierDataset if dataset_name == 'VCTK' else VoxClassifierDataset
    # else: dataset = VCTKTripletDataset if dataset_name == 'VCTK' else VoxTripletDataset
    # train_dataloader = dataset('train').get_dataloader(batch_size)
    # val_dataloader = dataset('val').get_dataloader(batch_size)
    train_dataloader = VoxContrastiveDataloader('train', batch_size)
    val_dataloader = VoxContrastiveDataloader('val', batch_size)

    model = Embedding(**model_args)
    
    if mode == 'embedding': model.toggle_emb_mode()
    
    criterion = pml.SupConLoss() if mode == 'embedding' else nn.CrossEntropyLoss()
    t = Trainer(model_name, model, train_dataloader, val_dataloader, criterion=criterion, **trainer_args)
    t.train(**train_args)
    