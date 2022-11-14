import torch
import torch.nn as nn

from contrastive import ContrastiveModel
from utils import exponential_linspace_int
from trainer import Trainer
import datasets

            
class EnergyModel(nn.Module):
    def __init__(self,
                 depth: int,
                 mode: str,
                 freeze_embeddings: bool=True
                 ):
        """
        Downstream model to use embeddings to predict energy of clip pairs

        Parameters
        ----------
        depth : int
            depth of energy head
        mode : str
            which dataset to work with. must be one of `['vctk', 'vox']`
        freeze_embeddings : bool
            whether to freeze the embedding model during training
        """
        super().__init__()
        
        assert mode in ['vctk', 'vox']
        
        self.emb = ContrastiveModel.load(f'checkpoints/models/{mode}_emb.pth').toggle_emb_mode().eval()
        for p in self.emb.parameters():
            p.requires_grad = False
            
        self.emb_dim = self.emb.emb_dim
        self.freeze_embeddings = freeze_embeddings
        
        self.args = {'depth': depth,
                     'mode': mode,
                     'freeze_embeddings': freeze_embeddings}
        
        # use body to bring down to final output dimension
        dims = exponential_linspace_int(2 * self.emb_dim, 1, depth + 1)
        self.model = [nn.BatchNorm1d(2 * self.emb_dim), nn.Dropout(0.1)]
        for i in range(depth):
            in_dim, out_dim = dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.model.extend(layer)
        self.model[-1] = nn.Flatten(0)  # remove last ReLU
        self.model = nn.Sequential(*self.model)

    def forward(self, q1, q2):
        with torch.no_grad():
            self.emb.eval()
            emb1 = self.emb.emb(q1)
            emb2 = self.emb.emb(q2)
        
        # project to final dimension
        cat = torch.cat((emb1, emb2), dim=-1)
        out = self.model(cat)  
        return out

    @staticmethod
    def load(model_path: str, **kwargs):
        """ 
        Load the model from a file.
        """
        params = torch.load(model_path, map_location='cpu')
        model = EnergyModel(**params['args'], **kwargs)
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
    
    model_name = f'{dataset_name}_energy'
    batch_size = 512
    
    trainer_args = {'initial_lr': 0.02,
                    'lr_decay_period': 4,
                    'lr_decay_gamma': 0.6,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 200,
                    'eval_every': 2,
                    'patience': 3,
                    'num_tries': 4}
    model_args = {'depth': 3,
                 'mode': dataset_name,
                 'freeze_embeddings': True}

    train_dataloader = datasets.EnergyDataset('train', 0.3, dataset_name).get_dataloader(batch_size, num_workers=3)
    val_dataloader = datasets.EnergyDataset('val', 0.3, dataset_name).get_dataloader(batch_size, num_workers=2)

    model = EnergyModel(**model_args)
    
    t = Trainer(model_name, model, train_dataloader, val_dataloader, criterion=nn.BCEWithLogitsLoss(), **trainer_args)
    t.train(**train_args)
    