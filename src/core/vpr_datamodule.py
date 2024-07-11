# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
# 
# This code is part of OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2  as T2

from src.dataloaders.train.gsv_cities import GSVCitiesDataset
# from src.utils import config_manager
from src.utils.config_manager import ConfigManager
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset

class VPRDataModule(L.LightningDataModule):
    """
    A DataModule for Visual Place Recognition (VPR) tasks.

    This Pytorch Lightning DataModule implementation encapsulates the dataset preparation
    and data-loading code for VPR tasks. It supports GSV-cities and several validation datasets.

    Attributes:
        batch_size: Number of places per batch.
        img_per_place: Number of images per place.
        shuffle_all: If True, shuffles all the data, else shuffle in-city only.
        image_size: Tuple indicating the size of the images (height, width).
        num_workers: Number of worker processes for data loading.
        print_data_stats: If True, prints datasets statistics.
        cities: List of cities to be included in the training dataset (specific to GSV-Cities).
        mean_std: Dictionary indicating the mean and standard deviation of the dataset for normalization.
        batch_sampler: A custom sampler for drawing batches of data.
        random_sample_from_each_place: If True, randomly samples images from each place.
        val_set_names: List of names of validation sets to use.
    """

    def __init__(
        self,
        batch_size=100,
        img_per_place=4,
        shuffle_all=False,
        image_size=(320, 320),
        num_workers=4,
        cities=["London"],
        print_data_stats=True,
        mean_std={"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]},
        batch_sampler=None,
        random_sample_from_each_place=True,
        val_set_names=["pitts30k_val", "msls_val"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.print_data_stats = print_data_stats
        self.cities = cities
        self.mean_std = mean_std
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names

        # check that the validation datasets exists
        for name in self.val_set_names:
            ConfigManager.check_dataset_exists(dataset_type="validation", dataset_name=name)
        
        
            
        self.train_transform = T2.Compose([
            T2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            T2.RandomResizedCrop(
                size=self.image_size,
                scale=(0.8, 1.0),
                ratio=(3/4, 4/3),
                interpolation=T2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T2.RandAugment(num_ops=3, magnitude=15, interpolation=T2.InterpolationMode.BILINEAR),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.valid_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.train_dataset = None
        self.val_datasets = None
        
    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._get_train_dataset()
            self.val_datasets = [self._get_val_dataset(name) for name in self.val_set_names]
        if self.print_data_stats:
            self.print_stats()
    
    def train_dataloader(self):
        # the reason we are initializing the dataset here is because
        # sometimes we want to shuffle the data (in-city only) at each epoch
        # which can only be done when loading the dataset's dataframes
        self.train_dataset = self._get_train_dataset()
        if self.batch_sampler is not None:
            return DataLoader(
                dataset=self.train_dataset,
                num_workers=self.num_workers,
                batch_sampler=self.batch_sampler,
                pin_memory=True,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=self.shuffle_all,
        )

    def val_dataloader(self):
        val_dataloaders = []
        for i,dataset in enumerate(self.val_datasets):
            dl = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    # batch_size=1,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True,
                    shuffle=False
                )
            val_dataloaders.append(dl)
        return val_dataloaders
    
    def _get_train_dataset(self):
        hard_mining = False
        if self.batch_sampler is not None:
            hard_mining = True

        return GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            hard_mining=hard_mining,
        )
    
    def _get_val_dataset(self, name):
        # we need to first verify that the dataset name is valid
        # we can use the config helper for this
        
        if "msls_val" in name.lower():
            return MapillarySLSDataset(
                    dataset_path=ConfigManager.get_dataset_paths_by_type("validation")[name],
                    input_transform=self.valid_transform
                )
        # elif "pitts30k" in name.lower():
            # return PittsburghDataset(which_ds=name, input_transform=self.valid_transform)
        # elif "nordland" in name.lower():
            # return NordlandDataset(input_transform=self.valid_transform)
        # elif "sped" in name.lower():
            # return SPEDDataset(input_transform=self.valid_transform)
        else:
            raise ValueError(f"Unknown dataset name: {name}")


    def print_stats(self):
        from rich.table import Table
        from rich.console import Console
        from rich import box
        from rich.theme import Theme
        custom_theme = Theme({
            "title": "bold not italic underline white",
            "label": "dark_sea_green3",
            "value": "light_steel_blue",
            "border": "grey50",
        })
        # console = Console(theme=custom_theme)
        console = Console()
        console.print("\n")

        def create_table(title):
            return Table(
                # style="dark_magenta",
                title=f"[title]{title}[/title]",
                box=box.SQUARE,
                show_header=False,
                title_justify="left",
                title_style="bold",
            )

        def add_rows(table, data):
            for row in data:
                table.add_row(f"[label]{row[0]}[/label]", f"[value]{row[1]}[/value]")
            console.print(table)
            # console.print("\n")

        # Training dataset stats
        train_table = create_table("Training dataset stats")
        train_data = [
            ("nb. of cities", len(self.cities)),
            ("nb. of places", self.train_dataset.__len__()),
            ("nb. of images", self.train_dataset.total_nb_images)
        ]
        add_rows(train_table, train_data)

        # Validation datasets
        val_table = create_table("Validation datasets")
        val_data = [(f"Validation set {i+1}", name) for i, name in enumerate(self.val_set_names)]
        add_rows(val_table, val_data)

        # Data configuration
        config_table = create_table("Data configuration")
        config_data = [
            ("Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"),
            ("nb. of iterations", self.train_dataset.__len__() // self.batch_size),
            ("Image size", self.image_size)
        ]
        add_rows(config_table, config_data)