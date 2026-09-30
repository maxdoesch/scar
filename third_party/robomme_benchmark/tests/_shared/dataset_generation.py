"""
Test-fixture shim around robomme.env_record_wrapper.dataset_generation.

The reusable per-episode generation logic (DatasetCase, run_one_episode,
run_episode_with_retry) lives in src/. This module adds the
session-cache-on-disk behavior used by the pytest fixtures.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.env_record_wrapper import (  # noqa: E402
    DatasetCase,
    FailsafeTimeout,
    RobommeRecordWrapper,
    run_episode_with_retry,
    run_one_episode,
)
from robomme.env_record_wrapper.dataset_generation import (  # noqa: E402
    DATASET_RRT_MAX_ATTEMPTS,
    DATASET_SCREW_MAX_ATTEMPTS,
    MAX_SEED_ATTEMPTS,
    _patch_planner_screw_to_rrt,
    _tensor_to_bool,
)


# Backwards-compatible aliases for tests that import the underscore-prefixed names.
_run_one_episode = run_one_episode
_run_episode_with_retry = run_episode_with_retry


@dataclass(frozen=True)
class GeneratedDataset:
    case: DatasetCase
    work_dir: Path
    raw_h5_path: Path
    resolver_dataset_dir: Path
    resolver_h5_path: Path
    used_seed: int


def _write_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def generate_dataset_case(case: DatasetCase, cache_root: Path) -> GeneratedDataset:
    case_dir = cache_root / case.cache_key()
    work_dir = case_dir / "work"
    resolver_dataset_dir = case_dir / "resolver_dataset"
    resolver_h5_path = resolver_dataset_dir / f"record_dataset_{case.env_id}.h5"
    meta_path = case_dir / "meta.json"

    if meta_path.exists():
        meta = _read_meta(meta_path)
        raw_h5_path = Path(meta["raw_h5_path"])
        if raw_h5_path.exists() and resolver_h5_path.exists():
            return GeneratedDataset(
                case=case,
                work_dir=work_dir,
                raw_h5_path=raw_h5_path,
                resolver_dataset_dir=resolver_dataset_dir,
                resolver_h5_path=resolver_h5_path,
                used_seed=int(meta["used_seed"]),
            )

    case_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    resolver_dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_h5_path, used_seed = run_episode_with_retry(case=case, output_dir=work_dir)
    shutil.copy2(raw_h5_path, resolver_h5_path)

    payload = {
        "case": asdict(case),
        "used_seed": used_seed,
        "raw_h5_path": str(raw_h5_path),
        "resolver_h5_path": str(resolver_h5_path),
    }
    _write_meta(meta_path, payload)

    return GeneratedDataset(
        case=case,
        work_dir=work_dir,
        raw_h5_path=raw_h5_path,
        resolver_dataset_dir=resolver_dataset_dir,
        resolver_h5_path=resolver_h5_path,
        used_seed=used_seed,
    )


class DatasetFactoryCache:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self._memo: dict[str, GeneratedDataset] = {}

    def get(self, case: DatasetCase) -> GeneratedDataset:
        key = case.cache_key()
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        generated = generate_dataset_case(case, self.cache_root)
        self._memo[key] = generated
        return generated


DatasetFactory = Callable[[DatasetCase], GeneratedDataset]


__all__ = [
    "DATASET_RRT_MAX_ATTEMPTS",
    "DATASET_SCREW_MAX_ATTEMPTS",
    "DatasetCase",
    "DatasetFactory",
    "DatasetFactoryCache",
    "FailsafeTimeout",
    "GeneratedDataset",
    "MAX_SEED_ATTEMPTS",
    "RobommeRecordWrapper",
    "_patch_planner_screw_to_rrt",
    "_run_episode_with_retry",
    "_run_one_episode",
    "_tensor_to_bool",
    "generate_dataset_case",
    "run_episode_with_retry",
    "run_one_episode",
]
