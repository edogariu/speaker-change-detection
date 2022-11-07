import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import tqdm

from utils import count_parameters
from losses import ContrastiveLoss, SoftNearestNeighborsLoss, TripletMarginLoss
from pytorch_metric_learning import losses

# CRITERION = SoftNearestNeighborsLoss() 
# CRITERION = nn.BCEWithLogitsLoss() 
# CRITERION = nn.CrossEntropyLoss()
CRITERION = TripletMarginLoss(0.1)
# CRITERION = ContrastiveLoss(temperature=0.1)
# CRITERION = losses.SupConLoss()

class Trainer():
    def __init__(self, 
                 model_name: str,
                 model: nn.Module, 
                 train_dataloader: D.DataLoader,
                 val_dataloader: D.DataLoader,
                 initial_lr: float, 
                 lr_decay_period: int, 
                 lr_decay_gamma: float, 
                 weight_decay: float):
        """
        Trainer object to train a model. Uses Adam optimizer, StepLR learning rate scheduler, and a patience algorithm.

        Parameters
        ------------
        model_name : str
            name to call the model
        model : nn.Module
            model to train
        train_dataloader : D.DataLoader
            dataloader for training data
        val_dataloader : D.DataLoader
            dataloader for validation data. If this is `None`, we do not validate
        initial_lr : float
            learning rate to start with for each model
        lr_decay_period : int
            how many epochs between each decay step for each model
        lr_decay_gamma : float
            size of each decay step for each model
        weight_decay : float
            l2 regularization for each model
        """
        
        self.model_name = model_name
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if val_dataloader is None:
            print('No validation dataloader provided. Skipping validation.')

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Using {} for training'.format(self.device))

        # prep model and optimizer and scheduler and loss function
        self.model.train().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), initial_lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_period, gamma=lr_decay_gamma, verbose=True)
        self.criterion = CRITERION

        # prep statistics
        self.train_losses = {}
        self.val_losses = {}
        self.best_val_loss = (0, float('inf'))  # epoch num and value of best validation loss

    def train_one_epoch(self, epoch_num: int):
        self.model.train()
        print()
        print('-------------------------------------------------------------')
        print('------------------  TRAIN - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        avg_loss = 0.
        i = 0
        pbar = tqdm.tqdm(self.train_dataloader)
        for x, y in pbar:
            x = [_x.to(self.device) for _x in x]
            y = y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(*x)
            loss = self.criterion(out, y)
            loss.backward()
            loss = loss.item()
            avg_loss += loss
            self.optimizer.step()
            i += 1
            pbar.set_postfix({'batch loss': loss})
        avg_loss /= i
        self.train_losses[epoch_num] = avg_loss
        print('avg batch training loss for epoch {}: {}'.format(epoch_num, round(avg_loss, 6)))
        self.scheduler.step()

    def eval(self, epoch_num: int):
        self.model.eval()
        print()
        print('-------------------------------------------------------------')
        print('-------------------  VAL - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        avg_loss = 0.
        i = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_dataloader)
            for x, y in pbar:
                x = [_x.to(self.device) for _x in x]
                y = y.to(self.device)
                out = self.model(*x)
                loss = self.criterion(out, y).item()
                avg_loss += loss
                i += 1
                pbar.set_postfix({'batch loss': loss})

        avg_loss /= i
        self.val_losses[epoch_num] = avg_loss
        print('avg validation batch loss for epoch {}: {}'.format(epoch_num, round(avg_loss, 6)))
        return avg_loss

    def train(self, 
              num_epochs: int, 
              eval_every: int, 
              patience: int, 
              num_tries: int):
        """
        Train the model. Applies patience -- every `eval_every` epochs, we eval on validation data using a metric (corner distance) thats not the loss function. 
        We expect the model to improve in this metric as we train: if after `patience` validation steps the model has still not improved, we reset to the best previous checkpoint.
        We attempt this for `num_tries` attempts before we terminate training early.

        Parameters
        ---------------
        num_epochs : int
            number of epochs to train for
        eval_every : int
            interval (measured in epochs) between valiation evaluations
        patience : int
            number of evaluations without improvement before we reset to best checkpoint
        num_tries : int
            number of checkpoint resets before early stopping

        Returns 
        --------------
        model : nn.Module
            the trained model
        """
        print('Training the following model for {} epochs:\n\n{} with {} parameters'.format(num_epochs, self.model, count_parameters(self.model)))
        patience_counter = 0
        tries_counter = 0
        for e in range(num_epochs):
            try:
                self.train_one_epoch(e)
                if e % eval_every == 0:
                    if self.val_dataloader is not None:
                        val_loss = self.eval(e)
                        if val_loss < self.best_val_loss[1]:  # measure whether our model is improving
                            self.best_val_loss = (e, val_loss)
                            patience_counter = 0
                            self.save_checkpoint()
                            print('Saved checkpoint for epoch num {}'.format(e))
                        else:
                            patience_counter += 1
                            print('Patience {} hit'.format(patience_counter))
                            if patience_counter >= patience:  # if our model has not improved after `patience` evaluations, reset to best checkpoint
                                tries_counter += 1
                                patience_counter = 0
                                self.load_checkpoint()
                                print('Loaded checkpoint from epoch num {}'.format(self.best_val_loss[0]))
                                print('Try {} hit'.format(tries_counter))
                                if tries_counter >= num_tries:  # if our model has reset to best checkpoint `num_tries` times, we are done
                                    print('Stopping training!')
                                    break
                    else:
                        self.save_checkpoint()
                        print('Saved checkpoint for epoch num {}'.format(e))
            except KeyboardInterrupt:
                print('Catching keyboard interrupt!!!')
                self.finish_up(e)
                return self.model
        self.finish_up(num_epochs)
        return self.model

    def finish_up(self, e):
        val_loss = self.eval(e) if self.val_dataloader else -float('inf')
        if val_loss < self.best_val_loss[1]:
            self.save_checkpoint()
            print('Saved checkpoint at the end of training!')
        
        print('\nDone :)')
        
    def save_checkpoint(self):
        self.model.save(f'checkpoints/models/{self.model_name}.pth')
        torch.save(self.optimizer.state_dict(), f'checkpoints/optimizers/{self.model_name}.pth')
        
    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(f'checkpoints/models/{self.model_name}.pth')['state_dict'])
        self.optimizer.load_state_dict(torch.load(f'checkpoints/optimizers/{self.model_name}.pth'))
