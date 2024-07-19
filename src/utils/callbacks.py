# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import RichModelSummary

from rich.theme import Theme
from typing import List, Dict, Tuple
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree


THEMES = {
    "default": {
        "title": "bold #fb8500",  
        "header": "bold #fb8500",
        "text": "#2ec4b6",
        "label": "#2ec4b6", 
        "value": "bold #2ec4b6",
        "border": "#fb8500",
        "progress_bar": "green1",
        "progress_bar_pulse": "green1",
        "progress_bar_finished": "green1",
        "batch_progress": "green_yellow",
        "processing_speed": "#fb8500",
        # "metrics": "grey82",
    },
    "cool_modern": {
        "title": "bold #4A90E2",  # Blue
        "header": "bold #50E3C2",  # Turquoise
        "text": "#D1E8E2",  # Light Cyan
        "label": "#50E3C2",  # Turquoise
        "value": "#B8D8D8",  # Pale Blue
        "border": "#7B8D8E",  # Gray Blue
        "progress_bar": "#4A90E2",  # Blue
        "progress_bar_pulse": "#4A90E2",
        "progress_bar_finished": "#417B82",  # Darker Blue
        "batch_progress": "#50E3C2",  # Turquoise
    },
    "vibrant_high_contrast": {
        "title": "bold #FF6347",  # Tomato Red
        "header": "bold #FFD700",  # Gold
        "text": "#FFFFFF",  # White
        "label": "#FFD700",  # Gold
        "value": "#1E90FF",  # Dodger Blue
        "border": "#FF4500",  # Orange Red
        "progress_bar": "#FF6347",  # Tomato Red
        "progress_bar_pulse": "#FF6347",
        "progress_bar_finished": "#FF4500",  # Orange Red
        "batch_progress": "#FFD700",  # Gold
    },
    "magenta": {
        "title": "bold #FF69B4",  # Hot Pink
        "header": "bold #FF69B4",  # Deep Pink
        "text": "#FFFFFF",  # White
        "label": "#FF69B4",  # Hot Pink
        "value": "grey70",  # Light Gray
        "border": "#8B008B",  # Dark Magenta
        "progress_bar": "#FF1493",  # Deep Pink
        "progress_bar_pulse": "#FF69B4",  # Hot Pink
        "progress_bar_finished": "#C71585",  # Medium Violet Red
        "batch_progress": "#C71585",  # Medium Violet Red
    },
    "green_burgundy": {
        "title": "bold #556B2F",  # Dark Olive Green
        "header": "bold #6B8E23",  # Olive Drab
        "text": "#FFFFFF",  # White
        "label": "#6B8E23",  # Olive Drab
        "value": "grey70",  # Light Gray
        "border": "#8B0000",  # Dark Red
        "progress_bar": "#556B2F",  # Dark Olive Green
        "progress_bar_pulse": "#6B8E23",  # Olive Drab
        "progress_bar_finished": "#8B0000",  # Dark Red
        "batch_progress": "#8B0000",  # Dark Red
    },
}


class DatamoduleSummary(Callback):
    def __init__(self, theme_name=None):
        if theme_name:
            self.console = Console(theme=Theme(THEMES[theme_name]))
        else:
            self.console = Console(theme=Theme(THEMES["default"]))

    def on_fit_start(self, trainer, pl_module):
        if pl_module.verbose:
            self.display_data_stats(trainer.datamodule)
        
        
    def create_table(self) -> Table:
        return Table(box=None, show_header=False, min_width=32)

    def add_rows(self, panel_title: str, table: Table, data: List[Tuple[str, str]]) -> None:
        for row in data:
            table.add_row(f"[label]{row[0]}[/label]", f"[value]{row[1]}[/value]")
        panel = Panel(table, title=f"[title]{panel_title}[/title]", border_style="border", padding=(1, 1), expand=False)
        
        # for row in data:
        #     table.add_row(row[0], row[1])
        # panel = Panel(table, title=panel_title, padding=(1, 1), expand=False)
        
        self.console.print(panel)

    def add_tree(self, panel_title: str, tree_data: dict) -> None:
        tree = Tree(panel_title, hide_root=True, guide_style="border")
        for node, children in tree_data.items():
            branch = tree.add(f"[label]{node}[/label]")
            for child in children:
                branch.add(f"[value]{child}[/value]")
        panel = Panel(tree, title=f"[title]{panel_title}[/title]", border_style="border", padding=(1, 2), expand=False)
        
        # tree = Tree(panel_title, hide_root=True)
        # for node, children in tree_data.items():
        #     branch = tree.add(node)
        #     for child in children:
        #         branch.add(child)
        # panel = Panel(tree, title=panel_title, padding=(1, 2), expand=False)

        self.console.print(panel)




    def display_data_stats(self, datamodule):
        self.console.print("\n")

        # Training dataset stats
        train_table = self.create_table()
        train_data = [
            ("number of cities", str(len(datamodule.train_dataset.cities))),
            ("number of places", str(datamodule.train_dataset.__len__())),
            ("number of images", str(datamodule.train_dataset.total_nb_images))
        ]
        self.add_rows("Training dataset stats", train_table, train_data)

        # Validation datasets
        val_tree_data = {
            f"{datamodule.val_set_names[i]}": [
                f"queries     {val_set.num_queries}",
                f"references  {val_set.num_references}"
            ]
            for i, val_set in enumerate(datamodule.val_datasets)
        }
        self.add_tree("Validation datasets", val_tree_data)

        # Data configuration
        config_table = self.create_table()
        config_data = [
            ("iterations per epoch", str(datamodule.train_dataset.__len__() // datamodule.batch_size)),
            ("train batch size (PxK)", f"{datamodule.batch_size}x{datamodule.img_per_place}"),
            ("training image size", f"{datamodule.train_image_size[0]}x{datamodule.train_image_size[1]}"),
            ("validation image size", f"{datamodule.val_image_size[0]}x{datamodule.val_image_size[1]}")
        ]
        self.add_rows("Data configuration", config_table, config_data)

        self.console.print("\n")
    



class CustomRichProgressBar(RichProgressBar):
    def __init__(self, theme_name="default"):
        if theme_name is None or theme_name not in THEMES:
            print(f"Theme '{theme_name}' not found.")
            super().__init__(leave=False)
        else:
            super().__init__(
                leave=False,
                theme=RichProgressBarTheme(
                    description=THEMES[theme_name]["title"],        # Task description
                    progress_bar=THEMES[theme_name]["progress_bar"],            # Progress bar color
                    progress_bar_finished=THEMES[theme_name]["progress_bar_finished"],   # Finished bar color
                    progress_bar_pulse=THEMES[theme_name]["progress_bar"],      # Pulse bar color
                    batch_progress=THEMES[theme_name]["batch_progress"],          # Batch progress color
                    time=THEMES[theme_name]["text"],               # Time color
                    processing_speed=THEMES[theme_name]["text"],   # Processing speed color
                    metrics=THEMES[theme_name]["label"],           # Metrics color
                    metrics_text_delimiter="\n",      # Metrics text delimiter
                    metrics_format=".2f",
                ),
            )


class CustomRRichModelSummary(RichModelSummary):
    def __init__(self, theme_name="default"):
        if theme_name is None or theme_name not in THEMES:
            print(f"Theme '{theme_name}' not found.")
            super().__init__(max_depth=1)
        else:
            super().__init__(max_depth=1, header_style=THEMES[theme_name]["header"])
