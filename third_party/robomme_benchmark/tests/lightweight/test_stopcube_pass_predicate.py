"""Unit tests for StopCube's pass-count predicate."""

import torch

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.robomme_env.utils.subgoal_evaluate_func import is_obj_passed_target  # noqa: E402


class _Pose:
    """Minimal pose stub."""

    def __init__(self, xyz):
        self.p = torch.tensor([xyz], dtype=torch.float32)


class _Actor:
    """Minimal actor stub."""

    def __init__(self, xyz):
        self.pose = _Pose(xyz)


class _Env:
    """Minimal env stub."""

    def __init__(self, cube_half_size=0.02):
        self.cube_half_size = cube_half_size


def test_is_obj_passed_target_counts_crossings_without_double_counting():
    env = _Env()
    target = _Actor([0.0, 0.0, 0.01])
    cube = _Actor([0.5, 0.5, 0.01])

    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=1) is False

    cube.pose = _Pose([0.0, 0.0, 0.01])
    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=1) is True
    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=2) is False
    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=2) is False

    cube.pose = _Pose([0.5, 0.5, 0.01])
    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=2) is False
    cube.pose = _Pose([0.0, 0.0, 0.01])
    assert is_obj_passed_target(env, obj=cube, target=target, pass_index=2) is True
