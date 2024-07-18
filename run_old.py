import argparse
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework
from src.losses.vpr_losses import VPRLossFunction
from src.models import backbones, aggregators

from rich.traceback import install
install()






TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity', #
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

SEED = 0


def main():
    # init seed for reproducibility
    seed_everything(SEED, workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    # define the argument parser
    parser = argparse.ArgumentParser(description='Train VPRFramework')
    parser.add_argument('--batch-size', type=int, default=120, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizer')
    parser.add_argument('--wd', type=float, default=0.005, help='Weight decay for optimizer')
    parser.add_argument('--warmup', type=float, default=0, help='Warmup steps')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--dev', action='store_true', help='Enable fast development run')
    parser.add_argument('--compile', action='store_true', help='Compile the model using torch.compile()')
    args = parser.parse_args()

    # define a data module
    datamodule = VPRDataModule(
        train_set_name="gsv-cities-light",
        # cities=TRAIN_CITIES,
        cities=None,
        train_image_size=(320, 320),
        batch_size=args.batch_size,
        img_per_place=4,
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=args.num_workers,
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=[
            "msls-val", 
            "pitts30k-val",
        ],
    )
    
    
    # define a backbone
    backbone = backbones.ResNet(
        backbone_name="resnet50",
        pretrained=True,
        num_unfrozen_blocks=1,
        crop_last_block=True
    )

    # backbone = backbones.DinoV2()
    # backbone = backbones.ViT()
    # backbone = backbones.Convnext()
    # backbone = backbones.EfficientNet(
        # backbone_name='efficientnet_b0', 
        # pretrained=True, 
        # layers_to_freeze=2
    # )

    # aggregator = aggregators.MixVPR(
    #     in_channels=backbone.out_channels,
    #     in_h=20,
    #     in_w=20,
    #     out_channels=512,
    #     mix_depth=4,
    #     mlp_ratio=1,
    #     out_rows=4
    # )
    aggregator = aggregators.BoQ(
        in_channels=backbone.out_channels,
        proj_channels=512,
        num_queries=32,
        num_layers=2,
    )
    
    loss_function = VPRLossFunction(
        loss_fn_name="MultiSimilarityLoss", miner_name="MultiSimilarityMiner"
    )
    
    # define the pipeline and the training hyperparameters
    vpr_model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=loss_function,
        
        # ---- Train hyperparameters
        lr=args.lr, # papers use 8e-4 for finetuning
        optimizer=args.optimizer,
        weight_decay=args.wd, # papers use 0.05 for finetuning
        # weight_decay=0.05, # some papers use 0.05 for finetuning
        # momentum=0.9,
        warmup_steps=args.warmup,
        milestones=[5, 10, 15, 20],
        lr_mult=0.3,
        verbose=True,
    )
    
    if args.compile:
        vpr_model = torch.compile(vpr_model)
    
    
    checkpoint_cb = ModelCheckpoint(
        monitor="msls-val/R1",
        filename="epoch({epoch:02d})_step({step:04d})_R1[{msls-val/R1:.4f}]_R5[{msls-val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # progress_bar_cb = RichProgressBar(
    #     leave=False,
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="green1",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="green_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #     ),
    # )
    
    # blue-ish
    # progress_bar_cb = RichProgressBar(
    #     leave=False,
    #     theme=RichProgressBarTheme(
    #         description="bright_magenta",
    #         progress_bar="bright_blue",
    #         progress_bar_finished="bright_green",
    #         progress_bar_pulse="bright_yellow",
    #         batch_progress="bright_red",
    #         time="bright_cyan",
    #         processing_speed="bright_white",
    #         metrics="bright_magenta",
    #     ),
    # )
    
    # green-ish
    # progress_bar_cb = RichProgressBar(
    #     leave=False,
    #     theme=RichProgressBarTheme(
    #         description="bright_blue",
    #         progress_bar="cyan",
    #         progress_bar_finished="bright_cyan",
    #         progress_bar_pulse="magenta",
    #         batch_progress="bright_green",
    #         time="bright_white",
    #         processing_speed="bright_white",
    #         metrics="bright_yellow",
    #     ),
    # )
    # Define colors
    primary_orange = "bold #F05F42"
    light_orange = "#F2765D"
    dark_orange = "#DE3412"
    magenta_accent = "#E12353"
    soft_magenta = "#CC2FAA"

    from rich.console import Console
    from rich.theme import Theme
    from rich.progress import Progress

    # Initialize the RichProgressBar with the custom console
    # progress_bar_cb = RichProgressBar(
    #     leave=False,
    #     theme=RichProgressBarTheme(
    #         description="bold #EE4C2C",        # Task description
    #         progress_bar="#FFA07A",            # Progress bar color
    #         progress_bar_finished="#FF4500",   # Finished bar color
    #         progress_bar_pulse="#FFA07A",      # Pulse bar color
    #         batch_progress="#FF4500",          # Batch progress color
    #         time="bright_white",               # Time color
    #         processing_speed="bright_white",   # Processing speed color
    #         metrics="bright_yellow",           # Metrics color
    #         metrics_text_delimiter=" • ",        # Metrics text delimiter
    #         metrics_format=".2f",
    #     ),
    # )
    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary
    theme_name = "default" # default, cool_modern, vibrant_high_contrast, green_burgundy, magenta
    progress_bar_cb = CustomRichProgressBar(theme_name)    
    model_summary_cb = CustomRRichModelSummary(theme_name)    
    data_summary_cb = DatamoduleSummary(theme_name)
     
    # let's init a Tensorboard Logger
    tb_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}", 
        name=f"{aggregator.__class__.__name__}",
        default_hp_metric=False
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tb_logger,
        num_sanity_val_steps=-1,  # runs N validation steps before stating training (-1 to run all validation steps)
        precision="16-mixed",  # we use half precision to reduce  memory usage
        # precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[
            progress_bar_cb,
            data_summary_cb,
            model_summary_cb,
            checkpoint_cb,
            # backbone_finetuner,
            # accumulator,
        ],
        reload_dataloaders_every_n_epochs=2,  # we reload the dataset to shuffle the order
        log_every_n_steps=10,
        fast_dev_run=args.dev,  # dev mode (only runs one train iteration and one valid iteration, no checkpointing and no performance tracking).
        # gradient_clip_algorithm='value',
        # gradient_clip_val=1.0,
        # profiler="simple",
    )
    
    trainer.fit(
        model=vpr_model,
        datamodule=datamodule,
    )
    

if __name__ == "__main__":
    main()
