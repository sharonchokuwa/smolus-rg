"""
Loads a YAML experiment config and resolves dynamic variables
so paths work on both Colab and local.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)

def _detect_drive_root():
    """
    Auto-detect the Google Drive mount point when running on Colab.
    Falls back to the current working directory on local machines.
    returns str
    """
    colab_drive = "/content/drive/MyDrive"
    if Path(colab_drive).exists():
        logger.info(f"Colab environment detected. DRIVE_ROOT = {colab_drive}")
        return colab_drive
    local_fallback = str(Path.cwd())
    logger.info(f"Local environment detected. DRIVE_ROOT = {local_fallback}")
    return local_fallback


def _interpolate(obj, variables: dict):
    """Recursively replace {KEY} placeholders in all string values."""
    if isinstance(obj, str):
        for key, val in variables.items():
            obj = obj.replace("{" + key + "}", str(val))
        return obj
    if isinstance(obj, dict):
        return {k: _interpolate(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(item, variables) for item in obj]
    return obj


def load_config(config_path: str,
                drive_root: str | None = None,
                overrides: dict | None = None): 
    """
    Load and fully resolve a YAML experiment config.
    Args:
        config_path: Path to the YAML file.
        drive_root:  Override the auto-detected DRIVE_ROOT.
        overrides:  Optional dictionary of config values to override.
    Returns:
        A plain Python dict with all placeholders resolved.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

     # APPLY OVERRIDES ON THE RAW CFG, BEFORE ANY INTERPOLATION 
    if overrides:
        for dotted_key, value in overrides.items():           
            keys = dotted_key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            logger.info(f"Override (pre-interpolation): {dotted_key} = {value!r}")

    root = drive_root or _detect_drive_root()

    # resolve top-level scalar vars used by experiment.name
    dataset_format = cfg.get("dataset_format", "")
    if dataset_format == "raw":
        dataset_cfg = "csv_dataset"
    else:
        dataset_cfg = "dataset"
    
    top_level_vars = {
        "DRIVE_ROOT": root,
        "model_size": cfg.get("model_size", ""),
        "anatomy": cfg.get("anatomy", ""),
        "dataset_format":  dataset_format,
        "suffix": cfg.get(dataset_cfg, {}).get("suffix", ""),
        "format": dataset_format,
        "model.init_strategy": cfg.get("model", {}).get("init_strategy", ""),

    }
    cfg = _interpolate(cfg, top_level_vars)

    # experiment.name is resolved; expose it for downstream paths
    experiment_name = cfg.get("experiment", {}).get("name", "unnamed")
    full_vars = {
        **top_level_vars,
        "experiment.name": experiment_name,
        "anatomy.capitalize": str(cfg.get("anatomy", "")).capitalize(),
    }
    cfg = _interpolate(cfg, full_vars)

    logger.info(f"Config loaded: {config_path} | experiment={experiment_name}")
    return cfg

def re_interpolate_paths(cfg: dict, drive_root: str):
    """
    Re-run interpolation after runtime overrides have been applied.
    Called immediately after _apply_overrides() whenever experiment.name,
    anatomy, or any path-affecting key is overridden. 
    Only re-interpolates fields that reference {experiment.name} or {anatomy}.
    All other cfg values are unchanged.
    returns dict
    """
    dataset_format = cfg.get("dataset_format", "")
    dataset_cfg_key = "csv_dataset" if dataset_format == "raw" else "dataset"

    top_level_vars = {
        "DRIVE_ROOT":          drive_root,
        "model_size":          cfg.get("model_size", ""),
        "anatomy":             cfg.get("anatomy", ""),
        "dataset_format":      dataset_format,
        "suffix":              cfg.get(dataset_cfg_key, {}).get("suffix", ""),
        "format":              dataset_format,
        "model.init_strategy": cfg.get("model", {}).get("init_strategy", ""),
    }

    # Use the CURRENT (post-override) experiment name
    experiment_name = cfg.get("experiment", {}).get("name", "unnamed")
    full_vars = {
        **top_level_vars,
        "experiment.name":     experiment_name,
        "anatomy.capitalize":  str(cfg.get("anatomy", "")).capitalize(),
    }

    # Only re-interpolate the fields that contain path templates
    for path_field in ["output_dir", "drive_backup_dir"]:
        if path_field in cfg.get("experiment", {}):
            cfg["experiment"][path_field] = _interpolate(
                cfg["experiment"][path_field], full_vars)
    if "drive_backup_dir" in cfg.get("checkpointing", {}):
        cfg["checkpointing"]["drive_backup_dir"] = _interpolate(
            cfg["checkpointing"]["drive_backup_dir"], full_vars)
    # Also re-interpolate the ClearML task name which references experiment.name
    if "task_name" in cfg.get("clearml", {}):
        cfg["clearml"]["task_name"] = _interpolate(
            cfg["clearml"]["task_name"], full_vars)

    logger.info(
        f"Paths re-interpolated after override. "
        f"output_dir={cfg['experiment'].get('output_dir', '?')}"
    )
    return cfg


def get_active_checkpoint_id(cfg: dict):
    """Return the HuggingFace model ID / path for the configured init strategy as str."""
    strategy = cfg["model"]["init_strategy"]
    checkpoint_id = cfg["model"]["checkpoints"][strategy]
    logger.info(f"Model init strategy: '{strategy}' → {checkpoint_id}")
    return checkpoint_id


def make_output_dirs(cfg: dict):
    """Create output and backup directories if they don't exist."""
    for path_key in ["output_dir"]:
        p = Path(cfg["experiment"][path_key])
        p.mkdir(parents=True, exist_ok=True)
    if cfg["checkpointing"].get("drive_backup"):
        Path(cfg["checkpointing"]["drive_backup_dir"]).mkdir(parents=True, exist_ok=True)
