"""Central config loader. Every phase imports `load_config()` from here so
paths, thresholds, and hyperparameters stay in one place (configs/config.yaml).
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict, key_path: str) -> Path:
    """Resolve a dotted path key (e.g. 'paths.manifest_csv') relative to REPO_ROOT."""
    node = cfg
    for key in key_path.split("."):
        node = node[key]
    return REPO_ROOT / node
