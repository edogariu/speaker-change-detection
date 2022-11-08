import pandas as pd
import numpy as np
import torch.utils.data as D
import tqdm
import os
import torch
from scipy.io import wavfile
from collections import defaultdict
import pickle; import dill

from utils import CONTEXT_DURATION, QUERY_DURATION, SAMPLE_RATE

SPLIT_INTERVALS = {'train': 0.8,  # intervals for each split (note that split occurs by speakers here, since that is what we hope to generalize over)
                   'val': 0.2,
                   'test': 0.00}

class MozillaDataset(D.Dataset):
    def __init__(self, split: str, std_thresh: float=600):
        """
        Creates dataset of 1 second `.wav` clips from Mozilla Common Voice dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        std_thresh : float, optional
            minimum standard deviation of waveform to include in dataset, by default 600
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        self.df = pd.read_csv('data/mozilla/dataset.csv')
        self.df = self.df[self.df['stds'].gt(std_thresh)]  # filter out silent or non-useful clips
        
        # split the dataset the same way every time
        np.random.seed(0)
        speaker_ids = np.unique(self.df['ids'])
        rand_idxs = np.random.permutation(len(speaker_ids))
        val_idx = int(len(speaker_ids) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(speaker_ids) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        speakers_by_split = {'train': speaker_ids[train_idxs],
                             'val': speaker_ids[val_idxs],
                             'test': speaker_ids[test_idxs],
                             'all': speaker_ids}
        np.random.seed()
        
        speakers = speakers_by_split[split]
        self.paths = []
        self.labels = []
        self.speakers_to_paths = defaultdict(list)
        print(f'Constructing {split} dataset!')
        for id, path in tqdm.tqdm(zip(self.df['ids'], self.df['paths']), total=len(self.df)):
            self.speakers_to_paths[id].append(path)
            if id in speakers: 
                self.paths.append(path)
                self.labels.append(id)
        self.length = len(self.paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        inp = wavfile.read(f'data/mozilla/wavs/{self.paths[index]}')[1]
        inp = np.float32(inp / 2 ** 15)  # normalize
        return (inp,), self.labels[index]  # tuple of waveform data and speaker label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class VCTKDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of 1 second `.wav` clips from filtered VCTK dataset.

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
        self.df['paths'] = paths
        self.df['id_strs'] = [p.split('_')[0] for p in paths]
        id_to_int = {}
        for id in self.df['id_strs']:
            id_to_int[id] = len(id_to_int)
        self.df['ids'] = [id_to_int[id] for id in self.df['id_strs']]
        
        # split the dataset the same way every time
        np.random.seed(0)
        speaker_ids = np.unique(self.df['ids'])
        rand_idxs = np.random.permutation(len(speaker_ids))
        val_idx = int(len(speaker_ids) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(speaker_ids) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        speakers_by_split = {'train': speaker_ids[train_idxs],
                             'val': speaker_ids[val_idxs],
                             'test': speaker_ids[test_idxs],
                             'all': speaker_ids}
        np.random.seed()
        
        speakers = speakers_by_split[split]
        self.paths = []
        self.labels = []
        self.speakers_to_paths = defaultdict(list)
        print(f'Constructing {split} dataset!')
        for id, path in tqdm.tqdm(zip(self.df['ids'], self.df['paths']), total=len(self.df)):
            self.speakers_to_paths[id].append(path)
            if id in speakers: 
                self.paths.append(path)
                self.labels.append(id)
        self.length = len(self.paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        inp = wavfile.read(f'data/VCTK/{self.paths[index]}')[1]
        return (inp,), self.labels[index]  # tuple of waveform data and speaker label
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class PairedDataset(D.Dataset):
    def __init__(self, split: str, prob: float):
        """
        Creates dataset of pairs of 1 second `.wav` clips from filtered VCTK dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        prob : float
            probability that paired datapoint comes from the same speaker
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        self.prob = prob
        
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
        speaker_ids = np.unique(self.df['ids'])
        rand_idxs = np.random.permutation(len(speaker_ids))
        val_idx = int(len(speaker_ids) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(speaker_ids) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        speakers_by_split = {'train': speaker_ids[train_idxs],
                             'val': speaker_ids[val_idxs],
                             'test': speaker_ids[test_idxs],
                             'all': speaker_ids}
        np.random.seed()
        
        speakers = speakers_by_split[split]
        self.paths = []
        self.labels = []
        self.speakers_to_paths = defaultdict(list)
        print(f'Constructing {split} dataset!')
        for id, path in tqdm.tqdm(zip(self.df['ids'], self.df['paths']), total=len(self.df)):
            self.speakers_to_paths[id].append(path)
            if id in speakers: 
                self.paths.append(path)
                self.labels.append(id)
        self.length = len(self.paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        inp1 = wavfile.read(f'data/VCTK/{self.paths[index]}')[1]
        
        if np.random.rand() < self.prob:
            speaker = self.speakers_to_paths[self.labels[index]]
            rand_path = speaker[np.random.randint(len(speaker))]
            inp2 = wavfile.read(f'data/VCTK/{rand_path}')[1]
            comp = 1.
        else:
            rand_index = np.random.randint(self.length)
            inp2 = wavfile.read(f'data/VCTK/{self.paths[rand_index]}')[1]
            comp = float(self.labels[index] == self.labels[rand_index])
        return (inp1, inp2), comp
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class ContextDataset(D.Dataset):
    def __init__(self, split: str, prob: float):
        """
        Creates dataset of pairs of 1 second `.wav` clips from filtered VCTK dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        prob : float
            probability that paired datapoint comes from the same speaker
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        
        self.prob = prob
        
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
        speaker_ids = np.unique(self.df['ids'])
        rand_idxs = np.random.permutation(len(speaker_ids))
        val_idx = int(len(speaker_ids) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(speaker_ids) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        speakers_by_split = {'train': speaker_ids[train_idxs],
                             'val': speaker_ids[val_idxs],
                             'test': speaker_ids[test_idxs],
                             'all': speaker_ids}
        np.random.seed()
        
        self.paths = []
        self.labels = []
        self.id_strs = []
        print(f'Constructing {split} dataset!')
        for id, path, id_str in zip(self.df['ids'], self.df['paths'], self.df['id_strs']):
            if id in speakers_by_split[split]: 
                self.paths.append(path)
                self.labels.append(id)
                self.id_strs.append(id_str)
        self.speaker_to_context = dill.load(open('data/cache/metadata.pkl', 'rb'))
        self.length = len(self.paths)
        
        self.context = pickle.load(open('data/cache/contexts.pkl', 'rb'))

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        path = self.paths[index]
        id_str = self.id_strs[index]
        
        query = wavfile.read(f'data/VCTK/{path}')[1]

        if np.random.rand() < self.prob:
            speaker = self.speaker_to_context[id_str]
            rand_path = id_str + '_' + speaker[list(speaker.keys())[np.random.randint(len(speaker))]]['sentence_id'] + '_cache.pkl'
            context = self.context[rand_path]
            comp = 1.
        else:
            rand_index = np.random.randint(len(self.speaker_to_context))
            rand_id = list(self.speaker_to_context.keys())[rand_index]
            speaker = self.speaker_to_context[rand_id]
            rand_path = rand_id + '_' + speaker[list(speaker.keys())[np.random.randint(len(speaker))]]['sentence_id'] + '_cache.pkl'
            context = self.context[rand_path]
            comp = float(id_str == rand_id)

        idx_range = len(context) - int(CONTEXT_DURATION * SAMPLE_RATE)
        if idx_range < 0:
            context = np.pad(context, (0, -idx_range))
        else:
            rand_index = np.random.randint(idx_range)
            context = context[rand_index: int(CONTEXT_DURATION * SAMPLE_RATE) + rand_index]

        return (context, query), comp
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
    
class VoxDataset(D.Dataset):
    def __init__(self, split: str, prob: float, method: str):
        """
        Creates dataset of `.wav` clips from VoxCeleb1 dataset.

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        prob : float
            probability that paired datapoint comes from the same speaker
        method : str
            how to pass data. must be one of `['pairs', 'context']`
        """
        super().__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        assert method in ['pairs', 'context']
        
        self.prob = prob
        self.method = method
                
        # get dataframe describing dataset
        df = pd.read_csv('data/vox1/dataset.csv')
                
        # split the dataset the same way every time
        np.random.seed(0)
        speaker_ids = np.unique(df['speaker'])
        rand_idxs = np.random.permutation(len(speaker_ids))
        val_idx = int(len(speaker_ids) * SPLIT_INTERVALS['train']); test_idx = val_idx + int(len(speaker_ids) * SPLIT_INTERVALS['val'])
        train_idxs, val_idxs, test_idxs = rand_idxs[:val_idx], rand_idxs[val_idx: test_idx], rand_idxs[test_idx:]
        speakers_by_split = {'train': speaker_ids[train_idxs],
                             'val': speaker_ids[val_idxs],
                             'test': speaker_ids[test_idxs],
                             'all': speaker_ids}
        np.random.seed()

        self.paths = []
        self.labels = []
        self.speaker_to_paths = defaultdict(list)  # to find clips from a speaker quickly
        print(f'Constructing {split} dataset!')
        for d in df.values:
            _, id, path, _ = d
            if id in speakers_by_split[split]: 
                self.paths.append(path)
                self.labels.append(id)
                self.speaker_to_paths[id].append(path)

        self.length = len(self.paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        label = self.labels[index]
        path = self.paths[index]        
        _, query = wavfile.read(f'data/vox1/{path}')
        # rand_index = np.random.randint(len(query) - int(QUERY_DURATION * SAMPLE_RATE))  # pick random starting point
        # query = query[rand_index: int(QUERY_DURATION * SAMPLE_RATE) + rand_index]

        if np.random.rand() < self.prob:  # if we want to yield a pair from the same speaker
            duration = CONTEXT_DURATION if self.method == 'context' else QUERY_DURATION
        
            idx_range = len(query) - int(duration * SAMPLE_RATE) - int(QUERY_DURATION * SAMPLE_RATE)
            rand_index = np.random.randint(idx_range)
            other = query[rand_index: int(duration * SAMPLE_RATE) + rand_index]
            query = query[int(duration * SAMPLE_RATE) + rand_index: int(QUERY_DURATION * SAMPLE_RATE) + int(duration * SAMPLE_RATE) + rand_index]
            
            # speaker = self.speaker_to_paths[label]
            # rand_path = speaker[np.random.randint(len(speaker))]  # choose a random clip of our desired speaker
            comp = 1.
        else:
            rand_speaker = list(self.speaker_to_paths.keys())[np.random.randint(len(self.speaker_to_paths))]  # pick a random speaker
            speaker = self.speaker_to_paths[rand_speaker]
            rand_path = speaker[np.random.randint(len(speaker))]  # choose a random clip of that speaker
            _, other = wavfile.read(f'data/vox1/{rand_path}')
            duration = CONTEXT_DURATION if self.method == 'context' else QUERY_DURATION
            
            idx_range = len(other) - int(duration * SAMPLE_RATE)
            if idx_range < 0:
                other = np.pad(other, (0, -idx_range))
            else:
                rand_index = np.random.randint(idx_range)
            other = other[rand_index: int(duration * SAMPLE_RATE) + rand_index]
            rand_index = np.random.randint(len(query) - int(QUERY_DURATION * SAMPLE_RATE))  # pick random starting point
            query = query[rand_index: int(QUERY_DURATION * SAMPLE_RATE) + rand_index]
            comp = float(label == rand_speaker)

        # _, other = wavfile.read(f'data/vox1/{rand_path}')
        
        # duration = CONTEXT_DURATION if self.method == 'context' else QUERY_DURATION
        
        # idx_range = len(other) - int(duration * SAMPLE_RATE)
        # if idx_range < 0:
        #     other = np.pad(other, (0, -idx_range))
        # else:
        #     rand_index = np.random.randint(idx_range)
        #     other = other[rand_index: int(duration * SAMPLE_RATE) + rand_index]

        # rescale
        other = (other / 2 ** 15).astype(float)
        query = (query / 2 ** 15).astype(float)
        return (other, query), comp
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
# d = VoxDataset('train', 0.3, 'context').get_dataloader(1)
# import matplotlib.pyplot as plt
# for x, y in d:
#     print(x[0].shape, x[1].shape, y)
#     print(x[0].mean())
#     plt.plot(x[1][0])
#     plt.show()
    # break
