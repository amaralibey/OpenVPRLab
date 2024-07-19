"""
    Rethinking Visual Geo-localization for Large-Scale Applications
    
    Paper: https://arxiv.org/abs/2204.02287
    GitHub repo: https://github.com/gmberton/CosPlace
    
    Reference:
    @InProceedings{Berton_CVPR_2022_CosPlace,
        author    = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
        title     = {Rethinking Visual Geo-Localization for Large-Scale Applications},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {4878-4888}
    }
    
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class CosPlace(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/cosplace_model/layers.py
    It's a GeM followed by a linear layer. 
    NOTE: In the original paper, CosPlace is trained with a specific training procedure. You may get different results if you train it differently.
    
    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor 
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    m = CosPlace(2048, 512)
    r = m(x)
    print(r.shape)