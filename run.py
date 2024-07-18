import torch
import yaml
import importlib
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework
from src.losses.vpr_losses import VPRLossFunction

from rich.traceback import install
install() # this is for better traceback formatting

# we mostly use mean and std of ImageNet dataset for normalization
# you can define your own mean and std values and use them
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# list of all cities to be used in "gsv-cities"
# if you want to use a subset cities, you edit the list
# and pass it to the VPRDataModule
ALL_CITIES = [
    'Bangkok', 
    'BuenosAires', 
    'LosAngeles', 
    'MexicoCity',
    'OSL', 
    'Rome', 
    'Barcelona', 
    'Chicago', 
    'Madrid', 
    'Miami',
    'Phoenix', 
    'TRT', 
    'Boston', 
    'Lisbon', 
    'Medellin', 
    'Minneapolis', 
    'PRG', 
    'WashingtonDC', 
    'Brussels',
    'London', 
    'Melbourne', 
    'Osaka', 
    'PRS',
]


def load_config(config_path='model_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**params)


# This is called when the train mode is selected
def train(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    # let's create the VPR DataModule
    datamodule = VPRDataModule(
        train_set_name=config['datamodule']['train_set_name'],
        cities=config['datamodule']['cities'], # if None or "all" then we use all cities
        train_image_size=config['datamodule']['train_image_size'],
        batch_size=config['datamodule']['batch_size'],
        img_per_place=config['datamodule']['img_per_place'],
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=config['datamodule']['num_workers'],
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=config['datamodule']['val_set_names'],
        val_image_size=None, # if None, the same as train_image_size
    )


    # Let's instantiate the backbone, aggregator and loss function. These are the main components of the VPRFramework
    # Make sure the model_config.yaml file is properly configured
    backbone = get_instance(config['backbone']['module'], config['backbone']['class'], config['backbone']['params'])
    
    out_channels = backbone.out_channels
    # most of the time, the aggregator needs to know the number of output channels of the backbone
    # that arguments is passed to the aggregator as a parameter `in_channels`
    config['aggregator']['params']['in_channels'] = out_channels
    aggregator = get_instance(config['aggregator']['module'], config['aggregator']['class'], config['aggregator']['params'])
    loss_function = get_instance(config['loss_function']['module'], config['loss_function']['class'], config['loss_function']['params'])

    vpr_model = VPRFramework(
        backbone=backbone,
        aggregator=aggregator,
        loss_function=loss_function,
        optimizer=config['trainer']['optimizer'],
        lr=config['trainer']['lr'],
        weight_decay=config['trainer']['wd'],
        warmup_steps=config['trainer']['warmup'],
        milestones=config['trainer']['milestones'],
        lr_mult=config['trainer']['lr_mult'],
        verbose= not config["silent"],
        config_dict=config, # pass the config to the framework in order to save it
    )

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)


    # Let's define the TensorBoardLogger
    # We will save under the logs directory 
    # and use the backbone name as the subdirectory
    # e.g. a BoQ model with ResNet50 backbone will be saved under logs/ResNet50/BoQ
    # this makes it easy to compared different aggregators with the same backbone
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=f"{aggregator.__class__.__name__}",
        default_hp_metric=False
    )
    
    # Let's define the checkpointing.
    # We use a callback and give it to the trained
    # The ModelCheckpoint callback saves the best k models based on a validation metric
    # In this example we are using msls-val/R1 as the metric to monitor
    # The checkpoint files will be saved in the logs directory (which we defined in the TensorBoardLogger)
    checkpoint_cb = ModelCheckpoint(
        monitor="msls-val/R1",
        filename="epoch({epoch:02d})_step({step:04d})_R1[{msls-val/R1:.4f}]_R5[{msls-val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Let's define the progress bar, model summary and data summary callbacks
    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary
    # there are multiple themes you can choose from. They are defined in src.utils.callbacks
    # example: default, cool_modern, vibrant_high_contrast, green_burgundy, magenta
    progress_bar_cb = CustomRichProgressBar(config["display_theme"])    
    model_summary_cb = CustomRRichModelSummary(config["display_theme"])    
    data_summary_cb = DatamoduleSummary(config["display_theme"])
     

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        num_sanity_val_steps=0, # is -1 to run one pass on all validation sets before training starts
        precision="16-mixed",
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_cb,
            data_summary_cb,    # this will print the data summary
            model_summary_cb,   # this will print the model summary
            progress_bar_cb,    # this will print the progress bar
            ],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        fast_dev_run=config["dev"], # dev mode (only runs one train iteration and one valid iteration, no checkpointing and no performance tracking).
        enable_model_summary=False, # we are using our own model summary
    )

    # save the config into logs directory
    # with open(f"{tensorboard_logger.log_dir}/custom_config.yaml", 'w') as file:
    #     yaml.dump(config, file)
    
    trainer.fit(model=vpr_model, datamodule=datamodule)

def evaluate(config):
    print("Evaluation mode selected.")
    # Your evaluation logic here

def main():
    from argparser import parse_args
    config = parse_args()
    if config["train"]:
        train(config)
    # elif args.test:
        # evaluate(args, config)
    # else:
        # parser.print_help()

if __name__ == "__main__":
    main()
