import yaml
from pathlib import Path
from src import _PROJECT_ROOT

class ConfigManager:
    config_path = _PROJECT_ROOT / 'config.yaml'
    
    @classmethod
    def _load_config(cls):
        """
        Load the configuration file.

        :return: Parsed YAML configuration as a dictionary.
        """
        with open(cls.config_path, 'r') as file:
            cls.config = yaml.safe_load(file)
    
    @classmethod
    def get_dataset_paths_by_type(cls, dataset_type):
        """
        Retrieve all dataset paths of a specific type (train or validation).

        :param dataset_type: Type of dataset ('train' or 'val')
        :return: Dictionary of dataset paths for the specified type with names as keys and paths as values.
        """
        cls._load_config()
        dataset_paths = cls.config.get('datasets_paths', {})
        if dataset_type not in dataset_paths:
            raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file. "
                             f"Please use 'train' or 'validation'.")
        
        valid_paths = {}
        for name, path in dataset_paths[dataset_type].items():
            if path is None:
                raise ValueError(f"Dataset path for '{name}' not found in the configuration file. Please check `config.yaml` file.")
            dataset_path = Path(path)
            if dataset_path.exists():
                valid_paths[name] = dataset_path
            else:
                raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{name}' not found. Please check `config.yaml` file.")
        return valid_paths

    @classmethod
    def check_dataset_exists(cls, dataset_type, dataset_name):
        """
        Check if a specific dataset path exists.

        :param dataset_type: Type of dataset ('train' or 'val')
        :param dataset_name: Name of the dataset.
        :return: Boolean value indicating path existence.
        """
        cls._load_config()
        dataset_paths = cls.config.get('datasets_paths', {})
        if dataset_type not in dataset_paths:
            raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file. "
                             f"Please use 'train' or 'validation'.")
        
        if dataset_name not in dataset_paths[dataset_type]:
            raise ValueError(f"Dataset '{dataset_name}' not found in the configuration file. Please check `config.yaml` file.")
        
        if dataset_paths[dataset_type][dataset_name] is None:
            raise ValueError(f"Dataset path for '{dataset_name}' not found in the configuration file. Please check `config.yaml` file.")
        
        return True
    
    @classmethod
    def get_all_datasets(cls):
        """
        Retrieve all dataset paths for both train and validation datasets.

        :return: Dictionary with dataset types as keys and dictionaries of dataset paths as values.
        """
        cls._load_config()
        all_paths = {}
        dataset_paths = cls.config.get('datasets_paths', {})
        for dataset_type, paths in dataset_paths.items():
            valid_paths = {}
            for name, path in paths.items():
                dataset_path = Path(path)
                if dataset_path.exists():
                    valid_paths[name] = dataset_path
                else:
                    raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{name}' not found. Please check `config.yaml` file.")
            all_paths[dataset_type] = valid_paths
        return all_paths

    @classmethod
    def check_all_paths_exist(cls):
        """
        Check if all dataset paths in the configuration exist.

        :return: A dictionary with dataset names as keys and boolean values indicating path existence.
        """
        cls._load_config()
        path_status = {}
        dataset_paths = cls.config.get('datasets_paths', {})
        for dataset_type, paths in dataset_paths.items():
            for name, path in paths.items():
                dataset_path = Path(path)
                path_status[f"{dataset_type}/{name}"] = dataset_path.exists()
        return path_status
