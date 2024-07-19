# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import yaml
from pathlib import Path
from src import _PROJECT_ROOT


CONFIG_FILE_PATH = _PROJECT_ROOT / 'config/data/config.yaml'


def get_dataset_path(dataset_name, dataset_type, config_path=CONFIG_FILE_PATH):
    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError("Configuration file not found. Please check if `config/data/config.yaml` exists.")
    
    datasets = config.get('datasets', {})
    if dataset_type not in datasets:
        raise ValueError(f"Dataset type '{dataset_type}' not found in the configuration file `config/data/config.yaml`."
                         " Have you downloaded the datasets?" 
                         "\nYou can download val datasets with `scripts/datasets_downloader.py`" )
    
    if dataset_name not in datasets[dataset_type]:
        raise ValueError(f"Dataset '{dataset_name}' not found in the configuration file. Please check `config/data/config.yaml` file."
                         "\nYou can download datasets with `scripts/datasets_downloader.py`")
        
    if datasets[dataset_type][dataset_name] is None:
        raise ValueError(f"Dataset path for '{dataset_name}' not found in the configuration file. Please check `config/data/config.yaml` file. Or download the dataset with `scripts/datasets_downloader.py`")
    
    # check if the path exists
    dataset_path = datasets[dataset_type][dataset_name]
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' for '{dataset_name}' not found. Please check `config/data/config.yaml` file.")
    return dataset_path