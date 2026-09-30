"""
Generate per-task RoboMME HDF5 datasets driven by env_metadata JSON.

For each selected task, the script iterates every (episode, seed, difficulty)
triplet listed in src/robomme/env_metadata/<dataset>/record_dataset_<Task>_metadata.json
(filtered by --difficulty), records each episode via RobommeRecordWrapper, and
merges the per-episode HDF5 into a single record_dataset_<Task>.h5 whose schema
matches the published robomme_data_h5 dataset (root-level episode_<N>/ groups,
each with setup/ + timestep_<T>/{obs,action,info}/ subgroups).

Re-running the script is idempotent: episodes already present in the aggregated
file are skipped.

Example:
    python scripts/generate_dataset.py \\
        --dataset train \\
        --task-id All \\
        --difficulty easy \\
        --output-dir runs/generated_dataset
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional, Tuple

import h5py
import tyro

from robomme.env_record_wrapper import (
    BenchmarkEnvBuilder,
    DatasetCase,
    run_episode_with_retry,
)


TaskID = Literal[
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
    "All",
]
DatasetType = Literal["train", "test", "val", "debug", "train_400"]
Difficulty = Literal["easy", "medium", "hard", "all"]


def _select_metadata_records(
    builder: BenchmarkEnvBuilder,
    difficulty: Difficulty,
) -> list[tuple[int, int, Optional[str]]]:
    """Return [(episode_idx, base_seed, difficulty)] for the chosen task and difficulty."""
    selected: list[tuple[int, int, Optional[str]]] = []
    for (task, episode), record in builder.metadata_index.items():
        if task != builder.env_id:
            continue
        rec_difficulty = record.get("difficulty")
        if difficulty != "all" and rec_difficulty != difficulty:
            continue
        seed = record.get("seed")
        if seed is None:
            continue
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            continue
        selected.append((episode, seed_int, rec_difficulty))
    selected.sort(key=lambda r: r[0])
    return selected


def _existing_episodes(aggregated_h5: Path) -> set[int]:
    if not aggregated_h5.exists():
        return set()
    out: set[int] = set()
    with h5py.File(aggregated_h5, "r") as f:
        for key in f.keys():
            if key.startswith("episode_"):
                try:
                    out.add(int(key.split("_", 1)[1]))
                except ValueError:
                    continue
    return out


def _merge_episode_into_aggregate(
    per_episode_h5: Path,
    aggregated_h5: Path,
    episode_idx: int,
) -> None:
    """Copy episode_<idx> group from the per-episode file into the aggregated per-task file."""
    group_name = f"episode_{episode_idx}"
    with h5py.File(per_episode_h5, "r") as src, h5py.File(aggregated_h5, "a") as dst:
        if group_name not in src:
            raise KeyError(f"{per_episode_h5} missing expected group {group_name}")
        if group_name in dst:
            del dst[group_name]
        src.copy(src[group_name], dst, name=group_name)


def _generate_for_task(
    task_id: str,
    dataset: DatasetType,
    difficulty: Difficulty,
    output_dir: Path,
    max_seed_attempts: int,
    save_video: bool,
    video_variant: str,
    failure_injection_episodes: int,
    failure_injection_z_episodes: int,
) -> tuple[int, int]:
    """Generate the per-task aggregated h5. Return (n_succeeded, n_attempted)."""
    builder = BenchmarkEnvBuilder(env_id=task_id, dataset=dataset)
    records = _select_metadata_records(builder, difficulty)

    if not records:
        print(
            f"[{task_id}] no metadata records match difficulty={difficulty}, skipping."
        )
        return 0, 0

    aggregated_h5 = output_dir / f"record_dataset_{task_id}.h5"
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = _existing_episodes(aggregated_h5)
    if already_done:
        print(
            f"[{task_id}] resuming: {len(already_done)} episode(s) already in {aggregated_h5.name}"
        )

    staging_dir = output_dir / "_staging" / task_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    n_succeeded = 0
    n_attempted = 0
    for episode_idx, base_seed, rec_difficulty in records:
        if episode_idx in already_done:
            n_succeeded += 1
            continue

        n_attempted += 1
        case = DatasetCase(
            env_id=task_id,
            episode=episode_idx,
            base_seed=base_seed,
            difficulty=rec_difficulty,
            save_video=save_video,
            video_variant=video_variant,
            mode_tag="cli",
            failure_injection_max_episode=failure_injection_episodes - 1,
            failure_injection_z_max_episode=failure_injection_z_episodes - 1,
        )
        print(
            f"[{task_id}] generating episode {episode_idx} "
            f"(base_seed={base_seed}, difficulty={rec_difficulty})"
        )
        try:
            per_episode_h5, used_seed = run_episode_with_retry(
                case=case,
                output_dir=staging_dir,
                max_seed_attempts=max_seed_attempts,
            )
        except RuntimeError as exc:
            print(f"[{task_id}] episode {episode_idx}: {exc}")
            continue

        _merge_episode_into_aggregate(per_episode_h5, aggregated_h5, episode_idx)
        try:
            per_episode_h5.unlink()
        except OSError:
            pass
        n_succeeded += 1
        print(
            f"[{task_id}] episode {episode_idx}: merged into "
            f"{aggregated_h5.name} (used_seed={used_seed})"
        )

    # Move sidecar videos out of staging before it is wiped.
    staged_videos_dir = staging_dir / "videos"
    if staged_videos_dir.exists():
        final_videos_dir = output_dir / "videos"
        final_videos_dir.mkdir(parents=True, exist_ok=True)
        for video_file in staged_videos_dir.iterdir():
            shutil.move(str(video_file), str(final_videos_dir / video_file.name))

    # Clean up staging dir for this task if empty (and parent _staging if all tasks done).
    shutil.rmtree(staging_dir, ignore_errors=True)
    parent_staging = output_dir / "_staging"
    if parent_staging.exists() and not any(parent_staging.iterdir()):
        parent_staging.rmdir()

    return n_succeeded, len(records)


def main(
    dataset: DatasetType = "train",
    task_id: Tuple[TaskID, ...] = ("All",),
    difficulty: Difficulty = "all",
    output_dir: str = "runs/generated_dataset",
    max_seed_attempts: int = 30,
    save_video: bool = False,
    video_variant: Literal["simple", "grounded", "hybrid", "all"] = "all",
    failure_injection_episodes: int = 6,
    failure_injection_z_episodes: int = 3,
) -> None:
    """
    Generate one record_dataset_<Task>.h5 per task by iterating the metadata JSON
    for the chosen split and difficulty.

    Args:
        dataset: Metadata split to source episodes from
            (train / test / val / debug / train_400).
        task_id: One or more tasks. Pass "All" to expand to every task. Multiple
            values can be supplied space-separated, e.g. --task-id BinFill RouteStick.
        difficulty: easy / medium / hard / all (filter applied to metadata records).
        output_dir: Directory where record_dataset_<Task>.h5 files are written.
        max_seed_attempts: Per-episode retry budget. Each attempt uses base_seed + attempt.
        save_video: If true, RobommeRecordWrapper also writes sidecar success videos.
        video_variant: Which variant is drawn on debug mp4 overlays when
            --save-video is set: simple / grounded / hybrid / all (default all,
            draws every variant). HDF5 datasets always contain all variants.
        failure_injection_episodes: Number of leading episodes (per task) that
            get an intentional grasp failure injected for failure-recovery training.
            Episodes 0..(N-1) are affected. Set to 0 to disable.
        failure_injection_z_episodes: Of those, how many leading episodes use mode
            "z" (vertical drop). The remainder use mode "xy" (horizontal slip).
            Must be <= failure_injection_episodes.
    """
    if failure_injection_episodes < 0:
        raise ValueError("--failure-injection-episodes must be >= 0")
    if failure_injection_z_episodes < 0:
        raise ValueError("--failure-injection-z-episodes must be >= 0")
    if failure_injection_z_episodes > failure_injection_episodes:
        raise ValueError(
            "--failure-injection-z-episodes must be <= --failure-injection-episodes "
            f"(got z={failure_injection_z_episodes}, total={failure_injection_episodes})"
        )

    if "All" in task_id:
        task_ids = BenchmarkEnvBuilder.get_task_list()
    else:
        # Preserve user-given order, drop duplicates.
        seen: set[str] = set()
        task_ids = [t for t in task_id if not (t in seen or seen.add(t))]
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, tuple[int, int]] = {}
    for tid in task_ids:
        print(f"\n=== Task: {tid} ===")
        n_ok, n_total = _generate_for_task(
            task_id=tid,
            dataset=dataset,
            difficulty=difficulty,
            output_dir=out,
            max_seed_attempts=max_seed_attempts,
            save_video=save_video,
            video_variant=video_variant,
            failure_injection_episodes=failure_injection_episodes,
            failure_injection_z_episodes=failure_injection_z_episodes,
        )
        summary[tid] = (n_ok, n_total)

    print("\n=== Summary ===")
    print(f"{'Task':<22} {'Succeeded':>10} / {'Requested':>9}")
    for tid, (n_ok, n_total) in summary.items():
        print(f"{tid:<22} {n_ok:>10} / {n_total:>9}")
    print(f"\nDataset written to: {out}")


if __name__ == "__main__":
    tyro.cli(main)
