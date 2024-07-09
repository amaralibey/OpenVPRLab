import yaml
from pathlib import Path
from src import _PROJECT_ROOT

def load_config():
    """Load the configuration file.
    from the project root directory.

    Returns:
        config (dict): The configuration file as a dictionary.
    """
    config_path = _PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open('r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config):
    """Save the configuration file.

    Args:
        config (dict): The configuration file as a dictionary.
    """
    config_path = _PROJECT_ROOT / 'config.yaml'
    with config_path.open('w') as file:
        yaml.dump(config, file)
