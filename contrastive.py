import numpy as np
import torch
import torch.nn as nn
import pytorch_metric_learning.losses as pml

from trainer import Trainer
from losses import TripletMarginLoss
from pipeline import AudioPipeline
import architectures
import datasets

# -----------------------------------------------------------------------------------------------
# --------------------------------------- THE MODEL ---------------------------------------------
# -----------------------------------------------------------------------------------------------

class ContrastiveModel(nn.Module):
    def __init__(self,
                 in_freq: int,
                 n_layers: int,
                 n_mel: int,
                 emb_dim: int,
                 n_chan: int,
                 n_classes: int):
        super().__init__()
        
        self.emb_mode = False
        self.emb_dim = emb_dim
        
        self.args = {'in_freq': in_freq,
                     'n_layers': n_layers,
                     'n_mel': n_mel,
                     'emb_dim': emb_dim,
                     'n_classes': n_classes,
                     'n_chan': n_chan}
        
        self.pipe = AudioPipeline(n_mel=n_mel, n_fft=1024, input_freq=in_freq, resample_freq=in_freq)
        
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
        if self.emb_mode: return self.emb(*x) # self._triplet_forward(*x)
        else: return self._classify_forward(*x)
    
    @staticmethod
    def load(model_path: str, **kwargs):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path, map_location='cpu')
        model = ContrastiveModel(**params['args'], **kwargs)
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
    
    dataset_name = 'vox'; assert dataset_name in ['vctk', 'vox']
    mode = 'embedding'; assert mode in ['classifier', 'triplet', 'embedding']
    num_speakers = 111   # 10000 to use all speakers
        
    model_name = f'{dataset_name}_emb'
    batch_size = 512
    trainer_args = {'initial_lr': 0.02,
                    'lr_decay_period': 12,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 400,
                    'eval_every': 3,
                    'patience': 5,
                    'num_tries': 20}
    model_args = {'n_layers': 3,
                  'n_mel': 86,
                  'emb_dim': 256,
                  'n_classes': min(num_speakers, 111) if dataset_name == 'VCTK' else min(num_speakers, 1211),
                  'n_chan': 256,
                  'in_freq': 8000 if dataset_name == 'VCTK' else 16000}

    if mode == 'classifier': 
        criterion = nn.CrossEntropyLoss()
        dataset = datasets.VCTKDataset if dataset_name == 'VCTK' else datasets.VoxDataset
        train_dataloader = dataset('train', num_speakers=num_speakers).get_dataloader(batch_size, num_workers=3)
        val_dataloader = dataset('val', num_speakers=num_speakers).get_dataloader(batch_size, num_workers=2)
    elif mode == 'triplet': 
        criterion = TripletMarginLoss(0.1)
        dataset = datasets.VCTKTripletDataset if dataset_name == 'VCTK' else datasets.VoxTripletDataset
        train_dataloader = dataset('train').get_dataloader(batch_size, num_workers=3)
        val_dataloader = dataset('val').get_dataloader(batch_size, num_workers=2)
    else:
        criterion = pml.SupConLoss()
        if dataset_name == 'vox':
            train_dataloader = datasets.VoxContrastiveDataloader('train', batch_size, num_speakers=num_speakers)
            val_dataloader = datasets.VoxContrastiveDataloader('val', batch_size, num_speakers=num_speakers)
        else:
            train_dataloader = datasets.VCTKDataset('train', num_speakers=num_speakers).get_dataloader(batch_size, num_workers=3)
            val_dataloader = datasets.VCTKDataset('val', num_speakers=num_speakers).get_dataloader(batch_size, num_workers=2)

    model = ContrastiveModel(**model_args)
    
    if mode != 'classifier': model.toggle_emb_mode()
    
    t = Trainer(model_name, model, train_dataloader, val_dataloader, criterion=criterion, **trainer_args)
    t.train(**train_args)
    