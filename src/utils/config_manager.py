import yaml
from pathlib import Path
from src import _PROJECT_ROOT


CONFIG_FILE_PATH = _PROJECT_ROOT / 'config.yaml'


def get_dataset_path(dataset_name, dataset_type, config_path=CONFIG_FILE_PATH):
    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError("Configuration file not found. Please check if `config.yaml` exists.")
    
    datasets = config.get('datasets', {})
    if dataset_type not in datasets:
        raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file `config.yaml`."
                         " Have you downloaded the datasets?" 
                         "\nYou can download val datasets with `scripts/datasets_downloader.py`" )
    
    if dataset_name not in datasets[dataset_type]:
        raise ValueError(f"Dataset '{dataset_name}' not found in the configuration file. Please check `config.yaml` file."
                         "\nYou can download datasets with `scripts/datasets_downloader.py`")
        
    if datasets[dataset_type][dataset_name] is None:
        raise ValueError(f"Dataset path for '{dataset_name}' not found in the configuration file. Please check `config.yaml` file. Or download the dataset with `scripts/datasets_downloader.py`")
    
    # check if the path exists
    dataset_path = datasets[dataset_type][dataset_name]
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{dataset_name}' not found. Please check `config.yaml` file.")
    return dataset_path


# class ConfigManager:
#     config_path = _PROJECT_ROOT / 'config.yaml'
    
#     @classmethod
#     def _load_config(cls):
#         with open(cls.config_path, 'r') as file:
#             cls.config = yaml.safe_load(file)
    
    
#     @classmethod
#     def dataset_exists(cls, dataset_name, dataset_type):
#         """
#         Check if a dataset exists in the configuration file.

#         Args:
#             dataset_name (str): Name of the dataset.
#             dataset_type (str): Type of dataset ('train' or 'val').
            
#         Returns:
#             bool: True if the dataset exists, False otherwise
#         """
#         cls._load_config()
#         datasets = cls.config.get('datasets', {})
#         if cls.config is None:
#             raise ValueError("Configuration file not found. Please check if `config.yaml` exists. and datasets are downloaded")
            
#         if dataset_type not in datasets:
#             raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file `config.yaml`")
        
        
#         return dataset_name in datasets[dataset_type]
    
#     @classmethod
#     def get_dataset_paths_by_type(cls, dataset_type):
#         """
#         Retrieve all dataset paths of a specific type (train or validation).

#         :param dataset_type: Type of dataset ('train' or 'val')
#         :return: Dictionary of dataset paths for the specified type with names as keys and paths as values.
#         """
#         cls._load_config()
#         datasets = cls.config.get('datasets', {})
#         if dataset_type not in datasets:
#             raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file. "
#                              f"Please use 'train' or 'validation'.")
        
#         valid_paths = {}
#         for name, path in datasets[dataset_type].items():
#             if path is None:
#                 raise ValueError(f"Dataset path for '{name}' not found in the configuration file. Please check `config.yaml` file.")
#             dataset_path = Path(path)
#             if dataset_path.exists():
#                 valid_paths[name] = dataset_path
#             else:
#                 raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{name}' not found. Please check `config.yaml` file.")
#         return valid_paths

#     @classmethod
#     def check_dataset_exists(cls, dataset_type, dataset_name):
#         """
#         Check if a specific dataset path exists.

#         :param dataset_type: Type of dataset ('train' or 'val')
#         :param dataset_name: Name of the dataset.
#         :return: Boolean value indicating path existence.
#         """
#         cls._load_config()
#         dataset_paths = cls.config.get('datasets_paths', {})
#         if dataset_type not in dataset_paths:
#             raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file. "
#                              f"Please use 'train' or 'validation'.")
        
#         if dataset_name not in dataset_paths[dataset_type]:
#             raise ValueError(f"Dataset '{dataset_name}' not found in the configuration file. Please check `config.yaml` file.")
        
#         if dataset_paths[dataset_type][dataset_name] is None:
#             raise ValueError(f"Dataset path for '{dataset_name}' not found in the configuration file. Please check `config.yaml` file.")
        
#         return True
    
#     @classmethod
#     def get_all_datasets(cls):
#         """
#         Retrieve all dataset paths for both train and validation datasets.

#         :return: Dictionary with dataset types as keys and dictionaries of dataset paths as values.
#         """
#         cls._load_config()
#         all_paths = {}
#         dataset_paths = cls.config.get('datasets_paths', {})
#         for dataset_type, paths in dataset_paths.items():
#             valid_paths = {}
#             for name, path in paths.items():
#                 dataset_path = Path(path)
#                 if dataset_path.exists():
#                     valid_paths[name] = dataset_path
#                 else:
#                     raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{name}' not found. Please check `config.yaml` file.")
#             all_paths[dataset_type] = valid_paths
#         return all_paths

#     @classmethod
#     def check_all_paths_exist(cls):
#         """
#         Check if all dataset paths in the configuration exist.

#         :return: A dictionary with dataset names as keys and boolean values indicating path existence.
#         """
#         cls._load_config()
#         path_status = {}
#         dataset_paths = cls.config.get('datasets_paths', {})
#         for dataset_type, paths in dataset_paths.items():
#             for name, path in paths.items():
#                 dataset_path = Path(path)
#                 path_status[f"{dataset_type}/{name}"] = dataset_path.exists()
#         return path_status
