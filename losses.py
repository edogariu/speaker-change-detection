import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature: float=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb: torch.tensor, labels: torch.tensor):
        """
        Computes contrastive loss over batch.

        Parameters
        ----------
        emb : torch.tensor
            batch of embedded inputs
        labels : torch.tensor
            batch of ground truth labels

        Returns
        -------
        torch.tensor
            differentiable loss
        """
        batch_size = emb.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(emb, emb.T), self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # remove diagonal entries (self-contrast)
        mask = torch.eq(labels, labels.T).float().to(emb.device)
        self_contrast_mask = 1 - torch.eye(batch_size)
        mask = mask * self_contrast_mask
      
        # loss is negative log-likelihood over positive entries
        exp_logits = torch.exp(logits) * self_contrast_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = self.temperature * (mask * log_prob).sum(1) / mask.sum(1) 

        return mean_log_prob_pos.mean()

class SoftNearestNeighborsLoss(nn.Module):
    def __init__(self, metric: str = 'cosine', temperature: float = 100):
        """
        Creates module for Soft Nearest Neighbor Loss

        Parameters
        ----------
        metric : str, optional
            which distance metric to use in embedding space. must be one of `['cosine', 'euclidean']`, by default 'cosine'
        temperature : float, optional
            parameter for distance scaling in embedding space, by default 100
        """
        super().__init__()
        
        if metric not in ['cosine', 'euclidean']:
            raise ValueError('metric must be one of [cosine, euclidean]')
        
        self.metric = self.cosine_distance if metric == 'cosine' else self.euclidean_distance
        self.temperature = temperature
        self.EPS = 1e-5  # for stability purposes
    
    @staticmethod
    def euclidean_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def cosine_distance(A, B):
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
        return 1 - prod

    def forward(self, x, y):
        distances = self.metric(x, x)
        pick_probs = torch.exp(-(distances / self.temperature)) - torch.eye(x.shape[0])
        pick_probs /= (self.EPS + pick_probs.sum(axis=1).unsqueeze(1))
        masked_pick_probs = pick_probs * (y == y.unsqueeze(1)).squeeze().to(torch.float32)
        return -torch.log(self.EPS + masked_pick_probs.sum(axis=1)).mean()

if __name__ == '__main__':
    import torch
    x = torch.tensor([[5, 5.5, 4.9], [2, 2.1, 2.], [2., 2., 2.1], [5., 5.1, 4.7], [1, 1, 1]])
    y = torch.tensor([1, 0, 0, 1, 0])
    l = SoftNearestNeighborsLoss('euclidean')
    print(l(x, y))
    