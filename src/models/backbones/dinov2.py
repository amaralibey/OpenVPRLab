import torch
import torch.nn as nn


class DinoV2(nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=2,
    ):
        """DinoV2 backbone with the ability to keep only the last num_unfrozen_blocks trainable.

        Args:
            backbone_name (str, optional): DinoV2 variant. Defaults to "dinov2_vitb14".
            num_unfrozen_blocks (int, optional): number of blocks to unfreeze. Defaults to 2.

        Raises:
            ValueError: if the backbone_name is not in the available models.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {self.backbone_name} is not recognized!" 
                             f"Supported backbones are: {self.AVAILABLE_MODELS}")
                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze the patch embedding and positional encoding
        self.dino.patch_embed.requires_grad_(False)
        self.dino.pos_embed.requires_grad_(False)
        
        # freeze the first blocks, keep only the last num_unfrozen_blocks trainable
        for i in range(len(self.dino.blocks) - self.num_unfrozen_blocks):
            self.dino.blocks[i].requires_grad_(False)
        
        self.out_channels = self.dino.embed_dim

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.num_unfrozen_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.dino.blocks[-self.num_unfrozen_blocks : ]:
            x = blk(x)
            
        
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        _, _, C = x.shape # we know C == self.dino.embed_dim, but still...
        x = x.permute(0, 2, 1).contiguous().view(B, C, H//14, W//14)
        return x