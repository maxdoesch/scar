import dataclasses
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import tqdm
import tyro

import mme_vla_suite.training.config as _config
from mme_vla_suite.training.dataset_lerobot import RoboMMEDatasetLerobot
import openpi.shared.normalize as normalize
from openpi.training.data_loader import TorchDataLoader
from openpi.training.data_loader import TransformedDataset
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class KeepKeys(transforms.DataTransformFn):
    def __init__(self, keys: list[str]):
        self._keys = tuple(keys)

    def __call__(self, x: dict) -> dict:
        # Only retain leaves used for stats to keep batch pytree structure stable.
        return {k: x[k] for k in self._keys}


def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: DictConfig,
    action_horizon: int,
    batch_size: int,
    num_batches: int | None = None,
    num_workers: int = 0,
    compute_norm_stats: bool = False,
    seed: int = 0,
):
    resolved_root = Path(dataset_path)
    dataset = RoboMMEDatasetLerobot(
        root=resolved_root,
        data_config=data_config,
        history_config=history_config,
        action_horizon=action_horizon,
        compute_norm_stats=compute_norm_stats,
    )

    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            KeepKeys(["state", "actions"]),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    print(f"Dataset length: {len(dataset)}, batch size: {batch_size}")

    num_batches = len(dataset) // batch_size
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=None,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
    )
    return data_loader, num_batches


def main(
    config_name: str = "mme_vla_suite", repo_id: str = "robomme", dataset_path: str = "data/robomme_preprocessed_data"
):
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_data_loader(
        dataset_path=dataset_path,
        data_config=data_config,
        history_config=None,
        action_horizon=config.model.action_horizon,
        batch_size=128,
        num_workers=4,
        compute_norm_stats=True,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    print(f"norm_stats: {norm_stats}")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
