# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
# 
# This script is part of OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

# import os
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple
from rich.console import Console
from rich.prompt import Prompt
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
import yaml


# Dictionary mapping dataset names to their corresponding script filenames, sizes and type
AVAILABLE_DATASETS: Dict[str, Tuple[str, str, str]] = {
    "gsv-cities":       ("download_gsv_cities.sh",          "23.4 GB",  "train"),
    "gsv-cities-light": ("download_gsv_cities_light.sh",    "1.2 GB",   "train"),
    "pitts30k-val":     ("download_pitts30k_val.sh",        "0.7 GB",   "val"),
    "msls-val":         ("download_msls_val.sh",            "1.2 GB",   "val"),
}

console = Console()



def print_available_datasets() -> None:
    """Print the list of available datasets with their sizes."""
    console.print("\nWelcome to OpenVPRLab Datasets Downloader!\n", style="bold green")
    
    table = Table.grid(expand=False)
    table.add_column("Number", min_width=5, justify="center")
    table.add_column("Dataset", min_width=25)
    table.add_column("Type")
    table.add_column("Size", justify="right", min_width=10)

    table.add_row("", "Dataset", "Type", "Size", style="italic red")
    # table.add_row("", "----------------", "----", "----", style="bold green")
    for i, item in enumerate(AVAILABLE_DATASETS.items(), 1):
        dataset, (_, size, type) = item
        table.add_row(f"[bold]{str(i)}.[/bold]", dataset, type, size)

    table.add_row(f"[bold]{str(len(AVAILABLE_DATASETS) + 1)}.[/bold]", "Download All Datasets", "", 
                  f"{sum_dataset_sizes()}")

    panel = Panel.fit(
        table,
        title="Available Datasets to Download",
        border_style="bold",
        padding=(1, 1)
    )
    # console.print(panel)
    console.print(table)


def sum_dataset_sizes() -> str:
    """Calculate the total size of all datasets."""
    total_size = sum(float(size.split()[0]) for _, (_, size, _) in AVAILABLE_DATASETS.items())
    return f"{total_size:.1f} GB"


def choose_dataset() -> Tuple[str, List[str]]:
    """Prompt the user to choose a dataset."""
    
    print_available_datasets()

    while True:
        choice = Prompt.ask("\nEnter your choice of dataset to download (1-5 or press Enter to exit)", default="", show_default=False)
        if not choice:
            return None
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(AVAILABLE_DATASETS)+1:
                return choice_num
            else:
                console.print("Invalid choice. Please enter a valid number.", style="bold red")
        except ValueError:
            console.print("Invalid input. Please enter a number.", style="bold red")



def execute_script(script_path: str, dataset_name: str, dataset_type: str) -> None:
    """Execute the chosen .sh script and update config if successful."""
    try:
        result = subprocess.run(['bash', script_path], check=True, capture_output=True, text=True)
        console.print(f"Successfully executed {script_path}", style="bold green")
        if result.stdout:
            console.print("Output:", style="dim")
            console.print(result.stdout)

        # downloaded datasets are stored in the folder data/ in the project root
        dataset_path = Path(__file__).parent.parent / "data" / dataset_type / dataset_name
        update_config_yaml(dataset_name, dataset_type, dataset_path)
        
    except subprocess.CalledProcessError as e:
        console.print(f"\nAn error occurred while executing {script_path}:", style="bold red")
        if e.stdout:
            console.print("Standard output:", style="dim")
            console.print(e.stdout)
        if e.stderr:
            console.print("Error output:", style="dim")
            console.print(e.stderr)


def update_config_yaml(dataset_name: str, dataset_type: str, dataset_path: str) -> None:
    """Update the config/data_config.yaml file with the path of the downloaded dataset."""
    config_path = Path(__file__).parent.parent / "config/data_config.yaml" # Assuming the config file is in the project root
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        console.print(f"Config file not found at {config_path}. Creating a new one.", style="bold yellow")
        config = {}
    
    if config is None:
        config = {}
        
    if 'datasets' not in config:
        config['datasets'] = {}
    
    if dataset_type not in config['datasets']:
        config['datasets'][dataset_type] = {}
        
    config['datasets'][dataset_type][dataset_name] = dataset_path.as_posix()
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    console.print(f"Updated config/data_config.yaml with path for {dataset_name}", style="bold green")





def main() -> None:
    script_directory = Path(__file__).parent
    
    choice_num = choose_dataset()

    if choice_num is None:
        console.print("\nNo dataset selected. Exiting.", style="bold yellow")
        return
    
    # Download all datasets
    elif choice_num == len(AVAILABLE_DATASETS) + 1:
        for dataset_name, (script_filename, dataset_size, dataset_type) in AVAILABLE_DATASETS.items():
            script_path = script_directory / script_filename
            if not script_path.is_file():
                console.print(f"\nThe script {script_path} does not exist. Skipping.", style="bold yellow")
                continue
            console.print(f"\nDownloading {dataset_name} dataset using {script_filename}...", style="bold green")
            execute_script(script_path, dataset_name, dataset_type)
    # Download a single dataset
    else:
        dataset_name, (script_filename, dataset_size, dataset_type) = list(AVAILABLE_DATASETS.items())[choice_num - 1]
        script_path = script_directory / script_filename
        if not script_path.is_file():
            console.print(f"\nThe script {script_path} does not exist. Exiting.", style="bold yellow")
            return
        console.print(f"\nDownloading {dataset_name} dataset using {script_filename}...", style="bold green")
        execute_script(script_path, dataset_name, dataset_type)
        

    console.print("\nAll selected downloads completed.", style="bold green")

if __name__ == "__main__":
    main()