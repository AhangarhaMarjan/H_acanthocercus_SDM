"""Load config/config.yaml and resolve project and raw-data paths."""
from pathlib import Path
import os

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# codex_20261112: opt-in configuration keeps the original run and outputs intact.
CONFIG_FILE = Path(os.environ.get("HEMISCORPIUS_CONFIG", PROJECT_ROOT / "config" / "config.yaml"))
if not CONFIG_FILE.is_absolute():
    CONFIG_FILE = PROJECT_ROOT / CONFIG_FILE


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Read the YAML configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_dir(cfg: dict, key: str) -> Path:
    """Absolute path of a raw-data folder, e.g. raw_dir(cfg, 'terraclimate')."""
    raw = cfg["paths"]["raw"]
    return Path(raw["root"]) / raw[key]


def project_path(cfg: dict, key: str) -> Path:
    """Absolute path of a project folder ('processed' or 'results')."""
    return PROJECT_ROOT / cfg["paths"][key]


def processed_step(cfg: dict, step: str) -> Path:
    """Resolve an explicitly isolated output stage, otherwise use the original data."""
    override = cfg["paths"].get("processed_overrides", {}).get(step)
    return PROJECT_ROOT / override if override else project_path(cfg, "processed") / step


def qa_dir(cfg: dict, step_name: str) -> Path:
    """Folder for the QA report of one step, created if needed."""
    d = project_path(cfg, "results") / "qa" / step_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_tbd(value) -> bool:
    """True if a config value is still undecided."""
    return value is None or str(value).strip().upper() == "TBD"
