from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from src.utils.config_manager import ConfigManager

class MapillarySLSDataset(Dataset):
    """
    MapillarySLS validation dataset for visual place recognition.

    Args:
        dataset_path (str): Directory containing the dataset.
        input_transform (callable, optional): Optional transform to be applied on each image.
        
    Reference:
        @inProceedings{Warburg_CVPR_2020,
        author    = {Warburg, Frederik and Hauberg, Soren and Lopez-Antequera, Manuel and Gargallo, Pau and Kuang, Yubin and Civera, Javier},
        title     = {Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition},
        booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        year      = {2020},
        month     = {June}
        }
    """

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        input_transform: Optional[Callable] = None,
    ):
        self.input_transform = input_transform

        if dataset_path is None: # use path in config.yaml
            dataset_path = ConfigManager.get_dataset_paths_by_type("validation")["msls_val"]
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.is_dir():
                raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

        # make sure the path contains folders `cph` and `sf` and  the 
        # files `msls_val_ref_image_names.npy`, `msls_val_query_image_names.npy`, 
        # and `msls_val_ground_truth_25m.npy`
        if not (dataset_path / "cph").is_dir() or not (dataset_path / "sf").is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not contain the folders `cph` and `sf`. Please check the path.")
        if not (dataset_path / "msls_val_ref_image_names.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_ref_image_names.npy' does not exist in {dataset_path}. Please check the path.")
        if not (dataset_path / "msls_val_query_image_names.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_query_image_names.npy' does not exist in {dataset_path}. Please check the path.")
        if not (dataset_path / "msls_val_ground_truth_25m.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_ground_truth_25m.npy' does not exist in {dataset_path}. Please check the path.")
        
        self.dataset_path = dataset_path
        # Load image names and ground truth data
        self.ref_image_names = np.load(dataset_path / "msls_val_ref_image_names.npy")
        self.query_image_names = np.load(dataset_path / "msls_val_query_image_names.npy")


        self.ground_truth = np.load(dataset_path / "msls_val_ground_truth_25m.npy", allow_pickle=True)

        # Combine reference and query images
        self.images_pahts = np.concatenate((self.ref_image_names, self.query_image_names))
        self.num_references = len(self.ref_image_names)
        self.num_queries = len(self.query_image_names)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where image is a PIL image.
        """
        img_path = self.images_pahts[index]
        img = Image.open(self.dataset_path / img_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.images_pahts)