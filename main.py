from dataset import MozillaDataset, VCTKDataset, PairedDataset, ContextDataset, VoxDataset
from model import SpeakerEmbedding, SpeakerEnergy, SpeakerContextModel
from trainer import Trainer

if __name__ == '__main__':
    
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'emb_big'
    batch_size = 64
    trainer_args = {'initial_lr': 0.02,
                    'lr_decay_period': 2,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 60,
                  'eval_every': 1,
                  'patience': 3,
                  'num_tries': 4}
    
    # # embedding
    # model_args = {'in_dim': 8000,
    #               'out_dim': 128,
    #               'hidden_dim': 256,
    #               'body_type': 'linear',
    #               'pooling_type': 'max',
                  
    #               'spec_depth': 7,  # spectrogram head
    #               'spec_nchan': 256,
    #               'spec_pool_every': 1,
    #               'spec_pool_size': 2,
                  
    #               'body_depth': 4  # body
    #               }
    
    # # energy
    # model_args = {'in_dim': 8000,
    #               'hidden_dim': 256,
    #               'body_type': 'linear',
    #               'pooling_type': 'attention',
                  
    #               'spec_depth': 6,  # spectrogram head
    #               'spec_nchan': 128,
    #               'spec_pool_every': 1,
    #               'spec_pool_size': 2,
                  
    #               'body_depth': 6  # body
    #               }

    # context
    model_args = {'hidden_dim': 256,
                  'body_type': 'linear',
                  'pooling_type': 'max',
                  
                  'context_mel_size': 64,
                  'context_depth': 5,
                  'context_nchan': 512,
                  'context_pool_every': 1,
                  'context_pool_size': 2,
                  
                  'query_mel_size': 64,
                  'query_depth': 5,  
                  'query_nchan': 512,
                  'query_pool_every': 1,
                  'query_pool_size': 2,
                  
                  'body_depth': 4
                  }
    # model_args = {'hidden_dim': 128,
    #               'body_type': 'linear',
    #               'pooling_type': 'max',
                  
    #               'context_mel_size': 256,
    #               'context_depth': 7,
    #               'context_nchan': 128,
    #               'context_pool_every': 1,
    #               'context_pool_size': 2,
                  
    #               'query_mel_size': 256,
    #               'query_depth': 7,  
    #               'query_nchan': 128,
    #               'query_pool_every': 1,
    #               'query_pool_size': 2,
                  
    #               'body_depth': 4 
    #               }

    # --------------------------------------------------------------------------------------------------------

    # model = SpeakerEmbedding(**model_args)#; print(model); exit(0)
    # model = SpeakerEnergy(**model_args)
    model = SpeakerContextModel(**model_args)

    # train_dataset = VCTKDataset('train')
    # val_dataset = VCTKDataset('val')
    # train_dataset = PairedDataset('train', 0.3)
    # val_dataset = PairedDataset('val', 0.3)
    # train_dataset = ContextDataset('train', 0.3)
    # val_dataset = ContextDataset('val', 0.3)
    train_dataset = VoxDataset('train', 0.3, 'context')
    val_dataset = VoxDataset('val', 0.3, 'context')
    
    train_dataloader = train_dataset.get_dataloader(batch_size, num_workers=0)
    val_dataloader = val_dataset.get_dataloader(batch_size)

    trainer = Trainer(model_name, model, train_dataloader, val_dataloader, **trainer_args)
    trainer.train(**train_args)
    