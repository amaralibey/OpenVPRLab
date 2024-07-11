from typing import Optional, Callable, Tuple, Any
import pathlib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


# NOTE: you need to download the Pittsburg dataset from  the author's website
# https://www.di.ens.fr/willow/research/netvlad/
# when downloaded, put the folders in a directory and inti 'root='YOUR_PATH'
# I hardcoded the image names and ground truth for faster evaluation (which I include in the repo)


class PittsburghDataset(Dataset):
    """
    Pittsburg dataset. It contains pitts30k_val, pitts30k_test and pitts250k_test.

    Args:
        which_ds (str): Which of the three datasets to use.
        input_transform (callable, optional): Optional transform to be applied on each image.
        root (str, optional): Directory with image folders of Nordland.
        gt_root (str, optional): Directory with ground truth data (provided by GSV-Cities).
    """

    def __init__(
        self,
        which_ds: str = "pitts30k_test",  # which of the three datasets to use
        input_transform: Optional[Callable] = None,
        root: str = "/home/YOUR_DIR/datasets/Pittsburgh/",  # this points to image folders of MapillarySLS
        gt_root: str = "/home/YOUR_DIR/gsv-cities/datasets/",  # this is hard coded ground truth from GSV-Cities framework
        # root: str = "/run/media/amar/Storage/pitts30k-val",  # this points to image folders of MapillarySLS
        # gt_root: str = "/run/media/amar/Storage/pitts30k-val",  # this is hard coded ground truth from GSV-Cities framework
    ):
        assert which_ds.lower() in ["pitts30k_val", "pitts30k_test", "pitts250k_test"]
        self.input_transform = input_transform

        gt_root_path = pathlib.Path(gt_root)
        if not gt_root_path.is_dir():
            raise FileNotFoundError(
                f"The ground truth directory {gt_root} does not exist. Please check the path. Make sure to use the ground truth from GSV-Cities framework."
            )
        root_path = pathlib.Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Please check the path to the Pittsburg dataset.")

        self.root_path = root_path
        self.dbImages = np.load(
            gt_root_path.joinpath(f"Pittsburgh/{which_ds}_dbImages.npy")
        )
        self.qImages = np.load(
            gt_root_path.joinpath(f"Pittsburgh/{which_ds}_qImages.npy")
        )

        self.ground_truth = np.load(
            gt_root_path.joinpath(f"Pittsburgh/{which_ds}_gt.npy"), allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where image is a PIL image.
        """
        img_path = self.root_path.joinpath(self.images[index])
        img = Image.open(img_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.images)
