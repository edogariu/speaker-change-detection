import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ResBlock(nn.Module):
    def __init__(self, chan, stride=1):
        super(ResBlock, self).__init__()
        self.in_conv = nn.Conv2d(chan, chan, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(chan, chan, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(chan)

    def forward(self, x):
        h = self.in_conv(x)
        h = self.relu(h)
        h = self.out_conv(h)
        h = self.bn(h)
        h = self.relu(h + x)  # residual connection
        return h

class MaxPool1D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.MaxPool1d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AvgPool1D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.AvgPool1d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AttentionPool1D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(in_channels, in_channels, 1, bias = False)  # softmax is agnostic to shifts

    def forward(self, x):
        """
        number of features is in channel dim here
        """
        b, _, length = x.shape
        remainder = length % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, self.pool_size - remainder), value = 0)
            mask = torch.zeros((b, 1, length), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, self.pool_size - remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)     

class MaxPool2D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.MaxPool2d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AvgPool2D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.AvgPool2d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AttentionPool2D(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (h p) (w q) -> b d h w p q', p = pool_size, q = pool_size)
        self.to_attn_logits = nn.Conv3d(in_channels, in_channels, 1, bias = False)  # softmax is agnostic to shifts

    def forward(self, x):
        """
        number of features is in channel dim here
        """
        b, _, H, W = x.shape
        remainder_h = H % self.pool_size
        remainder_w = W % self.pool_size
        needs_padding = remainder_h > 0 or remainder_w > 0

        if needs_padding:
            remainder_h = self.pool_size - remainder_h if remainder_h > 0 else remainder_h
            remainder_w = self.pool_size - remainder_w if remainder_w > 0 else remainder_w
            x = F.pad(x, (0, remainder_w, 0, remainder_h), value = 0)
            mask = torch.zeros((b, 1, H, W), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder_w, 0, remainder_h), value = True)

        x = self.pool_fn(x)
        x = x.reshape(*x.shape[:4], self.pool_size ** 2)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            mask_pooled = self.pool_fn(mask)
            mask_pooled = mask_pooled.reshape(*mask_pooled.shape[:4], self.pool_size ** 2)
            logits = logits.masked_fill(mask_pooled, mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1) 