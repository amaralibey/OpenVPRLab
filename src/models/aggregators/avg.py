"""
    AVG refers to the global average pooling aggregator.
    This is the simplest form of aggregation, 
    where the input tensor is averaged over the spatial dimensions.

"""
import torch.nn.functional as F
import torch.nn as nn


class AVGPool(nn.Module):
    """
    AVG pooling module
    
    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x