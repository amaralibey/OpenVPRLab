import torch
from src.models.backbones.resnet import ResNet
from src.models.backbones.dinov2 import DinoV2
from src import _PROJECT_ROOT

# print(_PROJECT_ROOT)

def count_parameters(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters: {total_params/1e6:.3}M")
        print(f"Trainable parameters: {trainable_params/1e6:.3}M")
    return total_params, trainable_params

def main():
    # x = torch.randn(1, 3, 322, 322).cuda()
    # # agg = ResNet(
    # #     backbone_name="resnet50",
    # #     pretrained=True,
    # #     num_unfrozen_blocks=1,
    # #     crop_last_block=True,
    # # )

    # agg = DinoV2(
    #     backbone_name="dinov2_vitb14",
    #     num_unfrozen_blocks=2,
    # ).cuda()
    
    # print(DinoV2.AVAILABLE_MODELS)
    # count_parameters(agg)
    # output = agg(x)
    # print(output.shape)
    
    
    # agg = ResNet(
    #     backbone_name="resnet50",
    #     pretrained=True,
    #     num_unfrozen_blocks=1,
    #     crop_last_block=True,
    # ).cuda()

    
    # print(ResNet.AVAILABLE_MODELS.keys())
    # count_parameters(agg)
    # output = agg(x)
    # print(output.shape)
    from src.core.vpr_datamodule import VPRDataModule
    dm = VPRDataModule(
        batch_size=4,
        num_workers=8,
        print_data_stats=True,
        cities=["PRS", "London"],
        val_set_names=["msls_val"],
    )

    dm.setup("fit")
    
if __name__ == "__main__":
    main()
