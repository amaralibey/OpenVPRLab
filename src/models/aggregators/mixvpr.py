import torch.nn.functional as F
import torch.nn as nn




"""
    MixVPR: Feature Mixing for Visual Place Recognition
    
    Paper: https://arxiv.org/abs/2303.02190
    GitHub repo: https://github.com/amaralibey/MixVPR
    
    Reference:
    @inproceedings{ali2023mixvpr,
        title={{MixVPR}: Feature Mixing for Visual Place Recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
        pages={2998--3007},
        year={2023}
    }
    
"""

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    """
    MixVPR module

    Args:
        in_channels (int): Number of input channels (depth of input feature maps).
        in_h (int): Height of input feature maps. This has to be fixed accross all images
        in_w (int): Width of input feature maps. This has to be fixed accross all images
        out_channels (int): Number of output channels (depth-wise projection dimension).
        mix_depth (int): Number of stacked FeatureMixers (L). Defaults to 4.
        mlp_ratio (int): Ratio of the mid projection layer in the mixer block. Defaults to 1.
        out_rows (int): Row-wise projection dimension. Defaults to 4.
    """
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x