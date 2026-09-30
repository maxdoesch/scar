import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

from mme_vla_suite.models.config.history_config_utils import get_history_config_path
from mme_vla_suite.models.config.history_config_utils import load_history_config
import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config
import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.MME_VLA_Policy:
    repack_transforms = repack_transforms or transforms.Group()

    if train_config.model.use_history:
        logging.info("Checking history config")
        history_config_path = get_history_config_path(checkpoint_dir)
        history_config = load_history_config(history_config_path)

        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, history_config=history_config, use_history=True),
        )
    else:
        logging.info("Model does not use history; skipping history config")

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model, train=False)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    return _policy.MME_VLA_Policy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
