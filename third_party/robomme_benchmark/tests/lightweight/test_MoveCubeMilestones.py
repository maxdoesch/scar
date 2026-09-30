from __future__ import annotations

import importlib.util
import types

import numpy as np
import pytest
import torch

from tests._shared.repo_paths import find_repo_root


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


move_cube_mod = _load_module(
    "robomme.robomme_env.move_cube_under_test",
    "src/robomme/robomme_env/MoveCube.py",
)


class _Pose:
    def __init__(self, p, yaw: float = 0.0):
        self._p = np.asarray(p, dtype=np.float32)
        self._yaw = float(yaw)
        self.p = torch.tensor([self._p], dtype=torch.float32)
        self.q = torch.tensor(
            [[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]], dtype=torch.float32
        )
        self.sp = types.SimpleNamespace(p=self._p.copy())

    def to_transformation_matrix(self):
        c = np.cos(self._yaw)
        s = np.sin(self._yaw)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        matrix[:3, 3] = self._p
        return torch.tensor([matrix], dtype=torch.float32)


class _Actor:
    def __init__(self, p, yaw: float = 0.0):
        self.pose = _Pose(p, yaw=yaw)


class _Robot:
    def __init__(self, qpos=None):
        self._qpos = torch.tensor(
            [qpos or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]],
            dtype=torch.float32,
        )

    def get_qpos(self):
        return self._qpos


class _Agent:
    def __init__(self, tcp_p=(0.0, 0.0, 0.05), qpos=None):
        self.tcp = types.SimpleNamespace(pose=_Pose(tcp_p))
        self.robot = _Robot(qpos=qpos)

    def is_grasping(self, _obj):
        return False


def _capture_tasks(way: str):
    captured = {}

    def _fake_seq(self, tasks, allow_subgoal_change_this_timestep):
        captured["tasks"] = tasks
        return False, "pending", False, None

    original = move_cube_mod.sequential_task_check
    move_cube_mod.sequential_task_check = _fake_seq
    try:
        dummy = types.SimpleNamespace(
            elapsed_steps=0,
            use_demonstrationwrapper=False,
            demonstration_record_traj=False,
            cube_half_size=0.02,
            obj_flag=-1,
            peg_head=_Actor([0.0, 0.06, 0.02]),
            peg_tail=_Actor([0.0, -0.06, 0.02]),
            cube=_Actor([0.0, 0.0, 0.02]),
            goal_site=_Actor([0.08, 0.0, 0.0]),
            cube_init_pose=_Pose([0.0, 0.05, 0.02]),
            cube_init_pose_2=_Pose([0.0, -0.05, 0.02]),
            goal_site_1_pose_p=np.array([[0.08, -0.05, 0.0]], dtype=np.float32),
            goal_site_2_pose_p=np.array([[0.08, 0.05, 0.0]], dtype=np.float32),
            way=way,
            agent=_Agent(),
        )
        move_cube_mod.MoveCube.evaluate(dummy, solve_complete_eval=False)
    finally:
        move_cube_mod.sequential_task_check = original
    return captured["tasks"]


@pytest.mark.parametrize(
    ("way", "first_slice", "second_slice", "expected_names"),
    [
        (
            "peg_push",
            slice(0, 3),
            slice(5, 8),
            [
                "Pick up the peg",
                "Position the peg against the red cube",
                "Push the cube with the peg toward the target",
            ],
        ),
        (
            "gripper_push",
            slice(0, 2),
            slice(4, 6),
            [
                "Close the gripper on the cube",
                "Push the cube with the gripper toward the target",
            ],
        ),
        (
            "grasp_putdown",
            slice(0, 2),
            slice(4, 6),
            ["Pick up the cube", "place the cube onto the target"],
        ),
    ],
    ids=["peg-push", "gripper-push", "grasp-putdown"],
)
def test_movecube_milestone_sequences(way, first_slice, second_slice, expected_names):
    tasks = _capture_tasks(way)

    assert [task["name"] for task in tasks[first_slice]] == expected_names
    assert [task["name"] for task in tasks[second_slice]] == expected_names

    if way == "peg_push":
        assert tasks[1]["choice_label"] == "position the peg against the red cube"
        assert tasks[1]["failure_func"] is None
