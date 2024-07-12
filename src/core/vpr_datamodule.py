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
from src.utils import config_manager
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
        train_image_size: Tuple indicating the size of training images (height, width).
        val_image_size: Tuple indicating the size of validation images (height, width). Default is None. meaning use the same as train_image_size. 
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
        train_set_name="gsv-cities",
        cities=["Osaka"],
        train_image_size=(224, 224),
        batch_size=60,
        img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        val_set_names=["pitts30k_val", "msls_val"],
        val_image_size=None,
        num_workers=4,
        batch_sampler=None,
        mean_std={"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]},
        print_data_stats=True,
    ):
        super().__init__()
        self.train_set_name = train_set_name
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.shuffle_all = shuffle_all
        self.train_image_size = train_image_size
        self.val_image_size = val_image_size if val_image_size is not None else train_image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.print_data_stats = print_data_stats
        self.cities = cities
        self.mean_std = mean_std
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names

        
        # check that the training dataset exists
        # its path is defined in the config.yaml file
        # let's call the config_manager to check this
        self.train_set_path = config_manager.get_dataset_path(dataset_name=self.train_set_name, 
                                                              dataset_type="train")
        
        # check that the validation datasets exist
        # theirs paths are defined in the config.yaml file
        # let's call the config_manager to check this
        self.val_set_paths = {} 
        for ds_name in self.val_set_names:
            ds_path = config_manager.get_dataset_path(dataset_name=ds_name, dataset_type="val")
            # store the paths for later use
            self.val_set_paths[ds_name] = ds_path
        
        # Define the train transformations
        self.train_transform = T2.Compose([
            T2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            T2.RandAugment(num_ops=3, magnitude=15, interpolation=T2.InterpolationMode.BILINEAR),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        # Define the validation transformations
        self.val_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.val_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.train_dataset = None
        self.val_datasets = None
        
    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._get_train_dataset()
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
        if self.print_data_stats:
            self.print_stats()
    
    def train_dataloader(self):
        # the reason we are using `_get_train_dataset` here is because
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
            dataset_path=self.train_set_path,
            cities=self.cities,
            img_per_place=self.img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            hard_mining=hard_mining,
        )
    
    def _get_val_dataset(self, ds_name):        
        if "msls" in ds_name.lower():
            return MapillarySLSDataset(
                    dataset_path=self.val_set_paths[ds_name],
                    input_transform=self.val_transform
            )
        elif "pitts30k" in ds_name.lower():
            pass
            # return PittsburghDataset(which_ds=ds_name, input_transform=self.val_transform)
        # elif "nordland" in ds_name.lower():
            # return NordlandDataset(input_transform=self.val_transform)
        # elif "sped" in ds_name.lower():
            # return SPEDDataset(input_transform=self.val_transform)
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}")


    def print_stats(self):
        from rich.table import Table
        from rich.console import Console
        from rich import box
        from rich.theme import Theme
        from rich.panel import Panel
        from rich.tree import Tree
        
        custom_theme = Theme({
            "title": "not italic white",
            "label": "dark_sea_green3",
            "value": "light_steel_blue",
            "border": "grey50",
        })

        console = Console(theme=custom_theme)
        console.print("\n")

        def create_table():
            return Table(
                box=None,  # Making the table transparent
                show_header=False
            )

        def add_rows(panel_title, table, data):
            for row in data:
                table.add_row(f"[label]{row[0]}[/label]", f"[value]{row[1]}[/value]")
            panel = Panel(table, title=f"[title]{panel_title}[/title]", border_style="border", padding=(1, 1), expand=False)
            console.print(panel)

        def add_tree(panel_title, tree_data):
            tree = Tree(panel_title, hide_root=True)
            for node, children in tree_data.items():
                branch = tree.add(node)
                for child in children:
                    branch.add(child)
            panel = Panel(tree, title=f"[title]{panel_title}[/title]", border_style="border", padding=(0, 1), expand=False)
            console.print(panel)
        # Training dataset stats
        train_table = create_table()
        train_data = [
            ("nb. of cities", len(self.train_dataset.cities)),
            ("nb. of places", self.train_dataset.__len__()),
            ("nb. of images", self.train_dataset.total_nb_images)
        ]
        add_rows("Training dataset stats", train_table, train_data)

        # Validation datasets
        # val_table = create_table()
        # val_data = [(f"val dataset {i+1}  ", name) for i, name in enumerate(self.val_set_names)]
        # add_rows("Validation datasets", val_table, val_data)

         # Validation datasets
        val_tree_data = {
            f"Validation set {i+1}": [
                f"nb. queries: {val_set.num_queries}",
                f"nb. references: {val_set.num_references}"
            ]
            for i, val_set in enumerate(self.val_datasets)
        }
        add_tree("Validation datasets", val_tree_data)

        
        # Data configuration
        config_table = create_table()
        config_data = [
            ("train batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"),
            ("nb. of iter. per epoch", self.train_dataset.__len__() // self.batch_size),
            ("train image size", f"{self.train_image_size[0]}x{self.train_image_size[1]}"),
            ("val image size", f"{self.val_image_size[0]}x{self.val_image_size[1]}")
        ]
        add_rows("Data configuration", config_table, config_data)