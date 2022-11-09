import numpy as np
import torch
import torchaudio.transforms as T

from utils import INPUT_RATE, QUERY_DURATION
from contrastive import ContrastiveModel
from energy import EnergyModel

CONTEXT_WINDOW_SIZE = 3.

def similarity(embA, embB):
    dot = (embA * embB).sum()
    return (dot / (torch.norm(embA) * torch.norm(embB))).item()

class Inferencer():
    def __init__(self, sr: int):
        """
        Class to inference over a context and query window of (hopefully) the right sizes. 

        Parameters
        ----------
        sr : int
            input sampling freq
        """
        self.vox_emb = ContrastiveModel.load('checkpoints/models/vox_emb.pth').toggle_emb_mode().eval()
        self.vox_emb.pipe.resample = T.Resample(INPUT_RATE, INPUT_RATE)  # we will be passing in 8k audio :(
        self.vctk_emb = ContrastiveModel.load('checkpoints/models/vctk_emb.pth').toggle_emb_mode().eval()
        self.energy = EnergyModel.load('checkpoints/models/energy.pth').eval()
        
        self.resample = T.Resample(sr, INPUT_RATE)
        
    def infer(self, before, curr):
        """
        Previous context and current window
        """
        num_searches = 7
        
        with torch.no_grad():
            before = self.resample(torch.tensor(before).float())
            curr = self.resample(torch.tensor(curr).float().reshape(1, -1))
            
            vox_emb_c = self.vox_emb.emb(curr)
            vctk_emb_c = self.vctk_emb.emb(curr)
            
            vox_emb_dists = []
            vctk_emb_dists = []
            energies = []
            for i in np.linspace(0, len(before) - int(QUERY_DURATION * INPUT_RATE), num_searches).astype(int):
                b = before[i: i + int(QUERY_DURATION * INPUT_RATE)].reshape(1, -1)
                vox_emb_dists.append(1 - similarity(self.vox_emb.emb(b), vox_emb_c))
                vctk_emb_dists.append(1 - similarity(self.vctk_emb.emb(b), vctk_emb_c))
                energies.append(1 - torch.sigmoid(self.energy(b, curr)).item())
            
        return np.median(vox_emb_dists), np.median(vctk_emb_dists), np.median(energies)
    