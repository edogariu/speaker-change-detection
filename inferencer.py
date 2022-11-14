import numpy as np
import torch
import torchaudio.transforms as T

from utils import QUERY_DURATION
from energy import EnergyModel

CONTEXT_WINDOW_SIZE = 2.5

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
        self.vctk_energy = EnergyModel.load('checkpoints/models/vctk_energy.pth').eval()
        self.vox_energy = EnergyModel.load('checkpoints/models/vox_energy.pth copy').eval()
        
        self.vctk_resample = T.Resample(sr, 8000)
        self.vox_resample = T.Resample(sr, 16000)
                
    def infer(self, before, curr):
        """
        Previous context and current window. Returns 1 if detected speaker change during `curr` and 0 otherwise, as well as a Tuple of the 4 deciding values
        """
        num_searches = 12
        with torch.no_grad():
            before_vctk = self.vctk_resample(torch.tensor(before).float())
            curr_vctk = self.vctk_resample(torch.tensor(curr).float().reshape(1, -1))
            
            before_vox = self.vox_resample(torch.tensor(before).float())
            curr_vox = self.vox_resample(torch.tensor(curr).float().reshape(1, -1))
            
            vctk_emb = self.vctk_energy.emb.emb(curr_vctk)
            vox_emb = self.vox_energy.emb.emb(curr_vox)

            vctk_energies = []
            vctk_dists = []
            vox_energies = []
            vox_dists = []
            
            # inference over vctk model
            for i in np.linspace(0, len(before_vctk) - int(QUERY_DURATION * 8000), num_searches).astype(int):
                b_vctk = before_vctk[i: i + int(QUERY_DURATION * 8000)].reshape(1, -1)
                vctk_energies.append(1 - torch.sigmoid(self.vctk_energy(b_vctk, curr_vctk)).item())
                vctk_dists.append(1 - similarity(vctk_emb, self.vctk_energy.emb.emb(b_vctk)))
               
            # inference over vox model
            for i in np.linspace(0, len(before_vox) - int(QUERY_DURATION * 16000), num_searches).astype(int): 
                b_vox = before_vox[i: i + int(QUERY_DURATION * 16000)].reshape(1, -1)
                vox_energies.append(1 - torch.sigmoid(self.vox_energy(b_vox, curr_vox)).item())
                vox_dists.append(1 - similarity(vox_emb, self.vox_energy.emb.emb(b_vox)))

        return np.median(vctk_energies) * np.median(vox_energies), (np.median(vctk_energies), np.median(vctk_dists), np.median(vox_energies), np.median(vox_dists))
    