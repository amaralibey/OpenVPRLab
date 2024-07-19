"""
    Fine-tuning CNN Image Retrieval with No Human Annotation
    
    Paper: https://arxiv.org/abs/2204.02287
    Code repo: https://cmp.felk.cvut.cz/cnnimageretrieval/
    
    Reference:
    @article{radenovic2018fine,
        title={Fine-tuning CNN image retrieval with no human annotation},
        author={Radenovi{\'c}, Filip and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
        journal={IEEE transactions on pattern analysis and machine intelligence},
        volume={41},
        number={7},
        pages={1655--1668},
        year={2018},
        publisher={IEEE}
    }
    
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)