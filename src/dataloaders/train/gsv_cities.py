# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
GSV-Cities dataset 
====================

This module implements a PyTorch Dataset class for GSV-Cities dataset from the paper:

"GSV-Cities: Toward Appropriate Supervised Visual Place Recognition" 
by Ali-bey et al., published in Neurocomputing, 2022.


Citation:
    @article{ali2022gsv,
        title={{GSV-Cities}: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

URL: https://arxiv.org/abs/2210.10239
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.utils import config_manager



# First, check if the dataset is downloaded and the path put in the config/data_config.yaml file
# available_train_datasets = ConfigManager.get_dataset_paths_by_type("train")

# assert "gsv_cities" in available_train_datasets, "GSV-Cities dataset not found in the configuration file. Please check `config/data_config.yaml` file."

# Transforms are passed to the dataset, if not, we will use this standard transform
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Now we can define the dataset class
class GSVCitiesDataset(Dataset):
    def __init__(self,
                 dataset_path=None,
                 cities="all", # or None
                 img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 hard_mining=False,
                 ):
        """
        Args:
            cities (list): List of city names to use in the dataset. Default is "all" or None which uses all cities.
            base_path (Path): Base path for the dataset files.
            img_per_place (int): The number of images per place.
            random_sample_from_each_place (bool): Whether to sample images randomly from each place.
            transform (callable): Optional transform to apply on images.
            hard_mining (bool): Whether you are performing hard negative mining or not.
        """
        super().__init__()
        
        # check if the dataset path is provided, if not, use the one in the config/data_config.yaml file
        if dataset_path is None:
            print("No dataset path provided. Using `gsv-cities-light`. We will try to load the one in the config/data_config.yaml file.")
            dataset_path = config_manager.get_dataset_path(
                dataset_name="gsv-cities-light", 
                dataset_type="train")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist. Please check the path.")
            
        self.base_path = Path(dataset_path)
        
        # let's check if the cities are valid
        if cities == "all" or cities is None:
            # get all cities from the Dataframes folder
            cities = [f.name[:-4] for f in self.base_path.glob("Dataframes/*.csv")]
        else:
            for city in cities:
                if not (self.base_path / 'Dataframes' / f'{city}.csv').exists():
                    raise FileNotFoundError(f"Dataframe for city {city} not found. Please check the city name.")

        self.cities = cities
        self.img_per_place = img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.hard_mining = hard_mining
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing one DataFrame
            for each city in self.cities
        '''
        dataframes = []
        for i, city in enumerate(self.cities):
            df = pd.read_csv(self.base_path / 'Dataframes' / f'{city}.csv')
            df['place_id'] += i * 10**5 # to avoid place_id conflicts between cities
            df = df.sample(frac=1) # we always shuffle in city level
            dataframes.append(df)
        
        df = pd.concat(dataframes)
        # keep only places depicted by at least img_per_place images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]
        return df.set_index('place_id')
        
    def __getitem__(self, index):
        if self.hard_mining:
            place_id = index
        else:
            place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample k images
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place) 
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path / 'Images' / row['city_id'] / img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)
    
    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        city = row['city_id']
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return name
