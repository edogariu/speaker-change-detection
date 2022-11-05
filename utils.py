import numpy as np
import torch.nn as nn

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def exponential_linspace_int(start, end, num, divisible_by=1):
    """
    Exponentially increasing values of integers
    """
    base = np.exp(np.log(end / start) / (num - 1))
    return [int(np.round(start * base ** i / divisible_by) * divisible_by) for i in range(num)]

def zero_module(module):
    """
    Helpful method that zeros all the parameters in a nn.Module, used for initialization
    """
    for param in module.parameters():
        param.detach().zero_()
    return module