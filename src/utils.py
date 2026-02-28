from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.box import ROUNDED
from omegaconf import OmegaConf, DictConfig

def print_config(cfg: DictConfig):
    table = Table(title="Hydra Configuration", box=ROUNDED)

    # Styling options
    table.row_styles = [Style(color="cyan", dim=True), Style(color="magenta", dim=True)]  # Zebra style

    # Collect all keys
    keys = set()
    for key, value in cfg.items():
        keys.add(key)
        if isinstance(value, dict):
            for nested_key in value.keys():
                keys.add(nested_key)

    # Sort keys for consistent column ordering
    keys = sorted(keys)

    # Add columns for each key
    for key in keys:
        table.add_column(key, style="cyan")

    # Populate values in respective columns
    row = {}
    for key, value in cfg.items():
        if isinstance(value, dict): # What we want here is to recursively resolve into tables
            for nested_key, nested_value in value.items():
                row[nested_key] = OmegaConf.to_yaml(nested_value)
        else:
            row[key] = OmegaConf.to_yaml(value)

    # Add a row with values in respective columns
    table.add_row(*[row.get(key, "") for key in keys])

    console = Console()
    console.print(table)