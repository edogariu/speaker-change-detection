from dataset import MozillaDataset, VCTKDataset
from model import SpeakerEmbedding
from trainer import Trainer

if __name__ == '__main__':
    
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'test'
    batch_size = 2048
    trainer_args = {'initial_lr': 0.02,
                    'lr_decay_period': 3,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 60,
                  'eval_every': 1,
                  'patience': 3,
                  'num_tries': 4}
    
    model_args = {'in_dim': 8000,
                  'out_dim': 64,
                  'hidden_dim': 512,
                  'body_type': 'linear',
                  'pooling_type': 'attention',
                  
                  'spec_depth': 4,  # spectrogram head
                  'spec_nchan': 64,
                  'spec_pool_every': 1,
                  'spec_pool_size': 2,
                  
                  'body_depth': 4  # body
                  }

    # --------------------------------------------------------------------------------------------------------

    model = SpeakerEmbedding(**model_args)

    print('preparing datasets')
    train_dataset = VCTKDataset('train')
    val_dataset = VCTKDataset('val')
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)

    print('training')
    trainer = Trainer(model_name, model, train_dataloader, val_dataloader, **trainer_args)
    trainer.train(**train_args)
    