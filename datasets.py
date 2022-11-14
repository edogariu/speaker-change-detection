import numpy as np
import torch
import pandas as pd
import torch.utils.data as D
import os
from scipy.io import wavfile
from collections import defaultdict
import tqdm

from utils import SPLIT_INTERVALS, QUERY_DURATION


class VCTKDataset(D.Dataset):
    def __init__(self, split: str, num_speakers: int=10000):
        """
        Creates dataset of `.wav` clips from VCTK dataset. Yields `(clip, label)` pairs

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        num_speakers : int
            how many different speakers to use in the dataset
        """
        super().__init__()
        
        print(f'preparing VCTK:{split} dataset')
        
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
        
        # grab only num_speakers speakers
        speakers = np.unique(self.df['id_str'].values)[:num_speakers]  
        idxs = [i for i, s in enumerate(self.df['id_str'].values) if s in speakers]
        self.df = self.df.iloc[idxs]
            
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

class VoxDataset(D.Dataset):
    def __init__(self, split: str, num_speakers: int=10000):
        """
        Creates dataset of `.wav` clips from VoxCeleb1 dataset. Yields `(clip, label)` pairs

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        num_speakers : int
            how many different speakers to use in the dataset
        """
        super().__init__()
        
        print(f'preparing VoxCeleb1:{split} dataset')
        
        assert split in ['train', 'val', 'test', 'all']
        
        # get dataframe describing dataset
        self.df = pd.read_csv('data/vox1/dataset.csv')
        
        # grab only the num_speakers most common speakers
        speaker_counts = self.df['speaker'].value_counts()
        top_speakers = list(speaker_counts.keys())[:num_speakers]
        idxs = []
        for i in tqdm.tqdm(range(len(self.df))):
            speaker = self.df.iloc[i]['speaker']
            if speaker in top_speakers: idxs.append(i)
        self.df = self.df.iloc[idxs]
        
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
    
class VCTKTripletDataset(D.Dataset):
    def __init__(self, split: str):
        """
        Creates dataset of triplets of `.wav` clips from VCTK dataset. Yields `(anchor, positive, negative)` triplets

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
        super().__init__()
        
        print(f'preparing VCTK triplet:{split} dataset')
        
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
        Creates dataset of triplets of `.wav` clips from VoxCeleb dataset. Yields `(anchor, positive, negative)` triplets

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        """
        super().__init__()
        
        print(f'preparing VoxCeleb1 triplet:{split} dataset')
        
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
    def __init__(self, split: str, batch_size: int, num_speakers: int=10000):
        """
        Creates dataset of pairs of 1 second `.wav` clips from VoxCeleb dataset. Yields `(clip, label)` pairs

        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        batch_size : int
            batch size for dataloader
        num_speakers : int
            number of distinct speakers to use
        """
        print(f'preparing VoxCeleb1 contrastive:{split} dataloader')
        
        assert split in ['train', 'val', 'test', 'all']
        
        # get dataframe describing dataset
        self.df = pd.read_csv('data/vox1/dataset.csv')

        # grab only the num_speakers most common speakers
        speaker_counts = self.df['speaker'].value_counts()
        top_speakers = list(speaker_counts.keys())[:num_speakers]
        idxs = []
        for i in tqdm.tqdm(range(len(self.df))):
            speaker = self.df.iloc[i]['speaker']
            if speaker in top_speakers: idxs.append(i)
        self.df = self.df.iloc[idxs]
        
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
        
        self.speakers_to_paths = defaultdict(list)
        for i in self.idxs:
            datapoint = self.df.iloc[i]
            self.speakers_to_paths[self.speaker_to_idx[datapoint['speaker']]].append(datapoint['path'])
        
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
            # grab anchor
            _, label, path, length = datapoint
            label = self.speaker_to_idx[label]
            sr, anchor = wavfile.read(f'data/vox1/{path}')
            assert sr == 16000
            anchor = np.split(anchor, np.arange(0, len(anchor), int(QUERY_DURATION * sr)))[1:-1]
            assert len(anchor) == np.floor(length / QUERY_DURATION)
            if len(anchor) > self.batch_size // 16: # make sure one clip never takes up more than a 8th of the batch
                r = np.random.randint(len(anchor) - self.batch_size // 16)
                anchor = anchor[r: r + self.batch_size // 16]
            for a in anchor: 
                if np.std(a) < 75: continue  # filter out silence
                batch_x.append((a / 2 ** 15).astype(float))
                batch_y.append(label)
                
            # # grab one positive pair
            # speaker = self.speakers_to_paths[label]
            # rand_path = speaker[np.random.randint(len(speaker))]
            # sr, pos = wavfile.read(f'data/vox1/{rand_path}')
            # assert sr == 16000
            # pos = np.split(pos, np.arange(0, len(pos), int(QUERY_DURATION * sr)))[1:-1]
            # if len(pos) > self.batch_size // 16: # make sure one clip never takes up more than a 8th of the batch
            #     r = np.random.randint(len(pos) - self.batch_size // 16)
            #     pos = pos[r: r + self.batch_size // 16]
            # for a in pos:  # make sure one clip never takes up more than a 8th of the batch
            #     if np.std(a) < 200: continue  # filter out silence
            #     batch_x.append((a / 2 ** 15).astype(float))
            #     batch_y.append(label)

            count += 1
            if len(batch_y) > self.batch_size: break
        
        self.i += count
        self.num_batches_yielded += 1
        batch_x = np.stack(batch_x, axis=0)
        batch_y = np.stack(batch_y, axis=0)

        return (torch.tensor(batch_x),), torch.tensor(batch_y)

class EnergyDataset(D.Dataset):
    def __init__(self, split: str, prob: float, mode: str):
        """
        Parameters
        ----------
        split : str
            which split to make dataset from. must be one of `['train', 'val', 'test', 'all']`
        mode : str
            which dataset to work with. must be one of `['vctk', 'vox']`
        prob_same : float
            probability that paired datapoint comes from the same speaker
        """
        super().__init__()
        
        print('preparing Energy:{split} dataset')
        
        assert mode in ['vctk', 'vox']
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
        
        if mode == 'vctk':
            for id, path in zip(vctk_df['id_strs'], vctk_df['paths']):
                ids.append(id_to_int[id])
                paths.append('data/VCTK/' + path)
                srs.append(8000)
        else:
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
            query = query / 2 ** 15
            if np.random.rand() < self.prob:  # if we want to yield a pair from the same speaker
                rand_index = np.random.randint(query.shape[0] - 2 * int(QUERY_DURATION * sr))  # pick random starting point
                other = query[int(QUERY_DURATION * sr) + rand_index: 2 * int(QUERY_DURATION * sr) + rand_index]
                query = query[rand_index: int(QUERY_DURATION * sr) + rand_index]
                other_label = label
                datapoint = None
            else: 
                rand_index = np.random.randint(query.shape[0] - int(QUERY_DURATION * sr))  # pick random starting point
                query = query[rand_index: int(QUERY_DURATION * sr) + rand_index]
                datapoint = self.df.iloc[self.idxs[np.random.randint(self.length)]]
        else:
            rand_index = np.random.randint(query.shape[0] - int(QUERY_DURATION * sr))  # pick random starting point
            query = query[rand_index: int(QUERY_DURATION * sr) + rand_index]

            if np.random.rand() < self.prob:  # if we want to yield a pair from the same speaker
                speaker = self.speakers_to_rows[label]
                datapoint = self.df.iloc[speaker[np.random.randint(len(speaker))]]
                assert datapoint['id'] == label
            else:
                datapoint = self.df.iloc[self.idxs[np.random.randint(self.length)]]
        
        if datapoint is not None:
            other_label = datapoint['id']
            other_path = datapoint['path']
            other_sr = datapoint['sr']
            _sr, other = wavfile.read(other_path)
            assert other_sr == _sr
            other = torch.tensor(other)
            if other_sr == 16000:  # if this is a vox datapoint
                other = other / 2 ** 15
            rand_index = np.random.randint(other.shape[0] - int(QUERY_DURATION * sr))  # pick random starting point
            other = other[rand_index: int(QUERY_DURATION * sr) + rand_index]

        return (other, query), float(label == other_label)
    
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    