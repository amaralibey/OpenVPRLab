from typing import Optional, Callable, Tuple, Any
import pathlib
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
from src.utils import config_manager


# NOTE: for pitts30k-test and pitts250k-test 
# you need to download them from  the author's website
# https://www.di.ens.fr/willow/research/netvlad/
# 
# For faster loading I hardcoded the image names and ground truth for pitts30k-val (already comes with OpenVPRLav)

REQUIRED_FILES = {
    "pitts30k-val":     ["pitts30k_val_dbImages.npy", "pitts30k_val_qImages.npy", "pitts30k_val_gt_25m.npy"],
    "pitts30k-test":    ["pitts30k_test_dbImages.npy", "pitts30k_test_qImages.npy", "pitts30k_test_gt_25m.npy"],
    "pitts250k-test":   ["pitts250k_test_dbImages.npy", "pitts250k_test_qImages.npy", "pitts250k_test_gt_25m.npy"],
}

class PittsburghDataset(Dataset):
    """
    Pittsburg dataset. It can load pitts30k-val, pitts30k-test and pitts250k-test.

    Args:
        dataset_path (str): Directory containing the dataset. If None, the path in config/data_config.yaml will be used.
        input_transform (callable, optional): Optional transform to be applied on each image.
    """

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        input_transform: Optional[Callable] = None,
    ):
        
        self.input_transform = input_transform

        if dataset_path is None: # use path in config/data_config.yaml
            print("Using the default path of `pitts30k-val` in config/data_config.yaml")
            dataset_path = config_manager.get_dataset_path(dataset_name="pitts30k-val", dataset_type="val")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.is_dir():
                raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")
            
        if "pitts30k-val" in dataset_path.name:
            self.dataset_name = "pitts30k-val"
        elif "pitts30k-test" in dataset_path.name:
            self.dataset_name = "pitts30k-test"
        elif "pitts250k-test" in dataset_path.name:
            self.dataset_name = "pitts250k-test"
        else:
            raise FileNotFoundError(f"Please make sure the dataset name is either `pitts30k-val`, `pitts30k-test` or `pitts250k-test`.")
        
        # make sure required metadata files are in the directory        
        if not all((dataset_path / file).is_file() for file in REQUIRED_FILES[self.dataset_name]):
            raise FileNotFoundError(f"Please make sure all requiered metadata for {dataset_path} are in the directory. i.e. {REQUIRED_FILES[self.dataset_name]}")
        
        self.dataset_path = dataset_path
        self.dbImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][0])
        self.qImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][1])
        self.ground_truth = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][2], allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where image is a PIL image.
        """
        img_path = self.image_paths[index]
        img = Image.open(self.dataset_path / img_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)
