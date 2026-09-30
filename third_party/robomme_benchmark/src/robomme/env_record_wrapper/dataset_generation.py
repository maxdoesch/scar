"""
Dataset generation helpers: drive a single recording episode end-to-end with the
ManiSkill motion-planner and the RobommeRecordWrapper.

These helpers are reused by both the test fixtures (tests/_shared/dataset_generation.py)
and the user-facing scripts/generate_dataset.py CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from .RecordWrapper import FailsafeTimeout, RobommeRecordWrapper
from ..robomme_env import *  # noqa: F401,F403  -- registers gym envs
from ..robomme_env.utils.SceneGenerationError import SceneGenerationError
from ..robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)


DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3
MAX_SEED_ATTEMPTS = 30


@dataclass(frozen=True)
class DatasetCase:
    env_id: str
    episode: int
    base_seed: int
    difficulty: Optional[str]
    save_video: bool
    mode_tag: str
    # Failure-injection cutoffs (inclusive) on the episode index. Defaults match
    # the original hard-coded behavior (eps 0-5 inject, eps 0-2 use mode "z",
    # eps 3-5 use mode "xy"). Set to -1 to disable failure injection entirely.
    failure_injection_max_episode: int = 5
    failure_injection_z_max_episode: int = 2
    video_variant: str = "all"

    def cache_key(self) -> str:
        difficulty = self.difficulty if self.difficulty else "none"
        return (
            f"{self.env_id}_ep{self.episode}_{difficulty}_"
            f"{self.base_seed}_{int(self.save_video)}_{self.video_variant}_{self.mode_tag}"
        )


def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _patch_planner_screw_to_rrt(planner) -> None:
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        for _ in range(DATASET_SCREW_MAX_ATTEMPTS):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result

        for _ in range(DATASET_RRT_MAX_ATTEMPTS):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def run_one_episode(
    case: DatasetCase,
    seed: int,
    output_dir: Path,
) -> bool:
    """
    Build env + RobommeRecordWrapper, drive task_list[*]["solve"] callbacks with the
    motion planner, return whether the episode reached success.

    The HDF5 is written to <output_dir>/hdf5_files/<env_id>_ep<episode>_seed<seed>.h5
    only on success (handled inside RobommeRecordWrapper.close()).
    """
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=case.difficulty,
    )
    if case.episode <= case.failure_injection_max_episode:
        env_kwargs["robomme_failure_recovery"] = True
        env_kwargs["robomme_failure_recovery_mode"] = (
            "z" if case.episode <= case.failure_injection_z_max_episode else "xy"
        )

    env = gym.make(case.env_id, **env_kwargs)
    env = RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=case.env_id,
        episode=case.episode,
        seed=seed,
        save_video=case.save_video,
        video_variant=case.video_variant,
    )

    episode_successful = False
    try:
        env.reset()
        is_stick = case.env_id in ("PatternLock", "RouteStick")
        if is_stick:
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        _patch_planner_screw_to_rrt(planner)

        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        for task_entry in tasks:
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure:
                screw_failed = True
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout:
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            if _tensor_to_bool(success_flag):
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(fail_flag):
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )
    except SceneGenerationError:
        episode_successful = False
    finally:
        try:
            env.close()
        except Exception:
            pass

    return episode_successful


def run_episode_with_retry(
    case: DatasetCase,
    output_dir: Path,
    max_seed_attempts: int = MAX_SEED_ATTEMPTS,
) -> tuple[Path, int]:
    """
    Try seeds base_seed + 0, +1, ..., +(max_seed_attempts-1).
    Return (h5_path, used_seed) on the first successful episode.
    Raise RuntimeError if no attempt succeeds.
    """
    for attempt in range(max_seed_attempts):
        seed = case.base_seed + attempt
        try:
            success = run_one_episode(case=case, seed=seed, output_dir=output_dir)
        except Exception:
            continue
        if not success:
            continue

        h5_path = (
            output_dir / "hdf5_files" / f"{case.env_id}_ep{case.episode}_seed{seed}.h5"
        )
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing expected HDF5: {h5_path}")
        return h5_path, seed
    raise RuntimeError(
        f"[{case.env_id}] Failed to generate successful record in {max_seed_attempts} attempts."
    )
