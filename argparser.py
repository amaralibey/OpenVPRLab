# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import yaml
from typing import Dict, Any

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='VPR Framework Training and Evaluation')

    # General arguments
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--train', action='store_true', help='Run mode: train or evaluate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--silent', action='store_true', help='Disable verbose output')
    parser.add_argument('--compile', action='store_true', help='Compile the model using torch.compile()')
    parser.add_argument('--dev', action='store_true', help='Enable fast development run')
    parser.add_argument('--display_theme', type=str, help='Theme for the console display')

    # Datamodule arguments
    parser.add_argument('--train_set', type=str, help='Name of the training dataset')
    parser.add_argument('--val_sets', nargs='+', help='Names of the validation datasets')
    parser.add_argument('--train_image_size', type=int, nargs=2, help='Training image size (height width)')
    parser.add_argument('--val_image_size', type=int, nargs=2, help='Validation image size (height width). Dafault is None (same as training size)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--img_per_place', type=int, help='Number of images per place')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--backbone', type=str, help='Backbone model name')
    parser.add_argument('--aggregator', type=str, help='Aggregator model name')
    parser.add_argument('--loss_function', type=str, help='Loss function name')

    # Trainer arguments
    parser.add_argument('--optimizer', type=str, help='Optimizer name')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--warmup', type=int, help='Number of warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, help='Milestones for learning rate scheduler')
    parser.add_argument('--lr_mult', type=float, help='Learning rate multiplier for scheduler')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')

    args = parser.parse_args()

    # If a config file is provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("No config file provided. Using command-line arguments and default values.")
        config = {}

    # Update config with command-line arguments and default values
    config = update_config_with_args_and_defaults(config, args)

    return config

def update_config_with_args_and_defaults(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update the configuration dictionary with command-line arguments and default values.
    Priority: Command-line args > Config file values > Default values
    """
    # These are the default values for the framework. 
    # We will use these values if the user does not provide them in the config file or as command-line arguments.
    # If you want to change the default values, you can do so here. But it's recommended that you 
    # write them in a YAML config file and load them using the --config argument. 
    default_config = {
        'seed': 42,
        'silent': False,
        'compile': False,
        'dev': False,
        'display_theme': "default",
        'train': True,
        'datamodule': {
            'train_set_name': "gsv-cities-light",
            'cities': "all",
            'val_set_names': ["msls-val"],
            'train_image_size': [320, 320],
            'val_image_size': None,
            'batch_size': 60,
            'img_per_place': 4,
            'num_workers': 8,
        },
        'backbone': {
            'module': 'src.models.backbones',
            'class': 'ResNet',
            'params': {},
        },
        'aggregator': {
            'module': 'src.models.aggregators',
            'class': 'MixVPR',
            'params': {},
        },
        'loss_function': {
            'module': 'src.losses',
            'class': 'VPRLossFunction',
            'params': {},
        },
        'trainer': {
            'optimizer': "adamw",
            'lr': 0.0002,
            'wd': 0.001,
            'warmup': 0,
            'milestones': [10, 20, 30],
            'lr_mult': 0.1,
            'max_epochs': 40,
        },
    }

    # Helper function to update nested dictionaries
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Update config with default values for missing keys
    config = update_nested_dict(default_config, config)

    # Update with command-line arguments if provided
    arg_dict = vars(args)
    
    # Update datamodule config
    if arg_dict['train_set'] is not None:
        config['datamodule']['train_set_name'] = arg_dict['train_set']
    if arg_dict['val_sets'] is not None:
        config['datamodule']['val_set_names'] = arg_dict['val_sets']
    if arg_dict['train_image_size'] is not None:
        config['datamodule']['train_image_size'] = arg_dict['train_image_size']
    if arg_dict['val_image_size'] is not None:
        config['datamodule']['val_image_size'] = arg_dict['val_image_size']
    if arg_dict['batch_size'] is not None:
        config['datamodule']['batch_size'] = arg_dict['batch_size']
    if arg_dict['img_per_place'] is not None:
        config['datamodule']['img_per_place'] = arg_dict['img_per_place']
    if arg_dict['num_workers'] is not None:
        config['datamodule']['num_workers'] = arg_dict['num_workers']

    # Update model config
    if arg_dict['backbone'] is not None:
        config['backbone']['class'] = arg_dict['backbone']
    if arg_dict['aggregator'] is not None:
        config['aggregator']['class'] = arg_dict['aggregator']
    if arg_dict['loss_function'] is not None:
        config['loss_function']['class'] = arg_dict['loss_function']

    # Update trainer config
    if arg_dict['optimizer'] is not None:
        config['trainer']['optimizer'] = arg_dict['optimizer']
    if arg_dict['lr'] is not None:
        config['trainer']['lr'] = arg_dict['lr']
    if arg_dict['wd'] is not None:
        config['trainer']['wd'] = arg_dict['wd']
    if arg_dict['warmup'] is not None:
        config['trainer']['warmup'] = arg_dict['warmup']
    if arg_dict['milestones'] is not None:
        config['trainer']['milestones'] = arg_dict['milestones']
    if arg_dict['lr_mult'] is not None:
        config['trainer']['lr_mult'] = arg_dict['lr_mult']
    if arg_dict['max_epochs'] is not None:
        config['trainer']['max_epochs'] = arg_dict['max_epochs']

    # Update other general config
    if arg_dict['seed'] is not None:
        config['seed'] = arg_dict['seed']
    if arg_dict['silent']:
        config['silent'] = arg_dict['silent']
    if arg_dict['compile']:
        config['compile'] = arg_dict['compile']
    if arg_dict['display_theme'] is not None:
        config['display_theme'] = arg_dict['display_theme']
    if arg_dict['dev']:
        config['dev'] = arg_dict['dev']
    if arg_dict['train']:
        config['train'] = arg_dict['train']

    return config

if __name__ == "__main__":
    config = parse_args()