"""
    Conv-AP: this aggregation technique has been published in the following paper:
    
    GSV-Cities: Toward Appropriate Supervised Visual Place Recognition
    
    Paper: https://arxiv.org/abs/2210.10239
    GitHub repo: https://github.com/amaralibey/gsv-cities
    
    
    Reference:
    ----------
    
    @article{ali2022gsv,
        title={GSV-Cities: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

"""
import torch.nn.functional as F
import torch.nn as nn


class ConvAP(nn.Module):
    """
    Conv-AP module
    
    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels=512, s1=2, s2=2):
        super(ConvAP, self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x):
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x