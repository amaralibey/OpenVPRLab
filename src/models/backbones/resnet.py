import torch.nn as nn
import torch
import torchvision

class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        num_unfrozen_blocks=2,
        crop_last_block=True,
    ):
        """Class representing the resnet backbone used in the pipeline.
        
        Args:
            backbone_name (str): The architecture of the resnet backbone to instantiate.
            pretrained (bool): Whether the model is pretrained or not.
            num_unfrozen_blocks (int): The number of residual blocks to unfreeze (starting from the end).
            crop_last_block (bool): Whether to crop the last residual block.
        
        Raises:
            ValueError: if the backbone_name corresponds to an unknown architecture.
        """
        super().__init__()

        

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        all_layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]
        
        if crop_last_block:
            all_layers.remove(resnet.layer4)
        nb_layers = len(all_layers)

        # Check if the number of unfrozen blocks is valid
        assert (
            isinstance(num_unfrozen_blocks, int) and 0 <= num_unfrozen_blocks <= nb_layers
        ), f"num_unfrozen_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Split the resnet into frozen and unfrozen parts
            self.frozen_layers = nn.Sequential(*all_layers[:nb_layers - num_unfrozen_blocks])
            self.unfrozen_layers = nn.Sequential(*all_layers[nb_layers - num_unfrozen_blocks:])
            
            # this is helful to make PyTorch count the right number of trainable params
            # because it doesn't detect the torch.no_grad() context manager at init time
            self.frozen_layers.requires_grad_(False)
        else:
            # If the model is not pretrained, we keep all layers trainable
            if self.num_unfrozen_blocks > 0:
                print("Warning: num_unfrozen_blocks is ignored when pretrained=False. Setting it to 0.")
                self.num_unfrozen_blocks = 0
            self.frozen_layers = nn.Identity()
            self.unfrozen_layers = nn.Sequential(*all_layers)
        
        # Calculate the output channels from the last conv layer of the model
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = all_layers[-1][-1].conv2.out_channels
        else:
            self.out_channels = all_layers[-1][-1].conv3.out_channels
        
       
    def forward(self, x):
        # We use torch.no_grad() to avoid computing gradients for the frozen layers
        with torch.no_grad():
            x = self.frozen_layers(x)
        
        # Detach the tensor from any computing graph
        x = x.detach()
        
        # Pass the tensor through the unfrozen layers
        x = self.unfrozen_layers(x)
        return x