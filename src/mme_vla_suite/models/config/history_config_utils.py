"""Utilities for loading, saving, and extracting history configuration."""

import logging
import os
import pathlib
from typing import Any

import etils.epath as epath
import omegaconf

HISTORY_CONFIG_DEFAULT_PATH = "src/mme_vla_suite/models/config/robomme"


def load_history_config(history_config_path: pathlib.Path | str) -> omegaconf.DictConfig:
    """Load history config from YAML file.

    Args:
        history_config_path: Path to history_config.yaml file.

    Returns:
        Loaded history config as OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    history_config_path = pathlib.Path(history_config_path)
    if not history_config_path.exists():
        raise FileNotFoundError(f"Expected history config at {history_config_path}")

    logging.info(f"Loading history config from {history_config_path}")
    return omegaconf.OmegaConf.load(history_config_path)


def save_history_config(history_config: dict | omegaconf.DictConfig | None, ckpt_dir: pathlib.Path | str) -> None:
    """Save history config to YAML file.

    Args:
        history_config: History config dictionary or OmegaConf DictConfig.
        ckpt_dir: Checkpoint directory where history_config.yaml will be saved.
    """
    ckpt_dir = epath.Path(ckpt_dir)
    output_path = ckpt_dir / "history_config.yaml"
    logging.info(f"Saving history config to {output_path}")

    if history_config is None:
        logging.warning("No history config provided. Skipping save.")
        return

    if isinstance(history_config, dict):
        history_config = omegaconf.OmegaConf.create(history_config)

    # OmegaConf.save accepts a plain string path/file-like; epath.Path is not supported.
    omegaconf.OmegaConf.save(history_config, str(output_path))


def get_history_config_path(checkpoint_dir: pathlib.Path | str) -> pathlib.Path:
    """Get the expected path for history config YAML file given a checkpoint directory.

    Args:
        checkpoint_dir: Path to a checkpoint directory (e.g., runs/ckpts/.../9999).
                       The history config is expected in the parent directory.

    Returns:
        Path to history_config.yaml (checkpoint_dir.parent / "history_config.yaml").
    """
    return pathlib.Path(checkpoint_dir).parent / "history_config.yaml"


def get_history_config(history_config: str | omegaconf.DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        history_config_path = os.path.join(
            os.environ.get("HISTORY_CONFIG_PATH", HISTORY_CONFIG_DEFAULT_PATH), history_config
        )
        history_config = omegaconf.OmegaConf.load(history_config_path)
        return history_config
    if isinstance(history_config, omegaconf.DictConfig):
        return history_config
    if history_config is None:
        return None
    raise ValueError(f"Invalid history config: {history_config}")


DEFAULT_MAX_TOKEN_LEN = 64
_AUX_LANG_TASK_MAX_TOKEN_LEN: dict[str, int] = {
    "subtask": 128,
    "summary": 128,
    "subtask_summary": 180,
}


def resolve_max_token_len(
    history_config: str | omegaconf.DictConfig | None,
    base: int = DEFAULT_MAX_TOKEN_LEN,
    *,
    train: bool = True,
) -> int:
    """Resolve the absolute max token length for a given history config."""
    loaded_config = get_history_config(history_config)
    if loaded_config is None:
        return base

    aux_lang_task = getattr(loaded_config, "aux_lang_task", None)
    if aux_lang_task in _AUX_LANG_TASK_MAX_TOKEN_LEN:
        if not train:
            return base
        return _AUX_LANG_TASK_MAX_TOKEN_LEN[aux_lang_task]

    return base


def extract_visualization_filename_parts(history_config: omegaconf.DictConfig) -> dict[str, Any]:
    """Extract configuration values needed for visualization output filename.

    Args:
        history_config: Loaded history config.

    Returns:
        Dictionary with keys: representation_type, integration_type, mask_or_pad,
        sampling_type, mem_sink_len.
    """
    representation_type = history_config.representation_type
    integration_type = history_config.integration_type
    memory_type = f"{representation_type}_memory"
    mask_or_pad = getattr(history_config, memory_type).mask_or_pad
    sampling_type = getattr(history_config, memory_type).type
    mem_sink_len = getattr(history_config, memory_type).mem_sink_len

    return {
        "representation_type": representation_type,
        "integration_type": integration_type,
        "mask_or_pad": mask_or_pad,
        "sampling_type": sampling_type,
        "mem_sink_len": mem_sink_len,
    }


def build_visualization_filename(
    episode_idx: int,
    history_config: omegaconf.DictConfig,
) -> str:
    """Build visualization output filename from episode index and history config.

    Args:
        episode_idx: Episode index.
        history_config: Loaded history config.

    Returns:
        Filename string (without path or extension).
    """
    parts = extract_visualization_filename_parts(history_config)
    return (
        f"attention_visualization_{episode_idx}_{parts['representation_type']}_"
        f"{parts['integration_type']}_{parts['mask_or_pad']}_"
        f"{parts['mem_sink_len']}_sink_{parts['sampling_type']}"
    )
