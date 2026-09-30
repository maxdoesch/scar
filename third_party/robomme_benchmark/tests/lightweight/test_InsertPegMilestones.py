from __future__ import annotations

import importlib.util
import types

import numpy as np
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


insertpeg_mod = _load_module(
    "robomme.robomme_env.insertpeg_under_test",
    "src/robomme/robomme_env/InsertPeg.py",
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
        self.dof = 0

    def set_pose(self, pose):
        p = np.asarray(pose.p, dtype=np.float32).reshape(-1)
        self.pose = _Pose(p)

    def set_qpos(self, _qpos):
        return None

    def set_qvel(self, _qvel):
        return None


class _Agent:
    def __init__(self):
        self.robot = types.SimpleNamespace(pose=_Pose([-0.615, 0.0, 0.0]))
        self.tcp = types.SimpleNamespace(pose=_Pose([0.0, 0.12, 0.05]))

    def reset(self, _qpos):
        return None


class _TableScene:
    def initialize(self, _env_idx):
        return None


def test_insertpeg_picks_then_inserts_without_align_step():
    dummy = _build_dummy_env()
    insertpeg_mod.InsertPeg._initialize_episode(dummy, torch.tensor([0]), {})

    pick_name = (
        f"Pick up the {dummy.peg_position_label} peg "
        f"by grasping the {dummy.grasp_target_distance} end"
    )
    assert dummy.task_list[0]["name"] == pick_name
    assert (
        dummy.task_list[1]["name"]
        == f"Insert the peg from the {dummy.insert_way} side of the box"
    )

    for task in dummy.task_list:
        assert "Align" not in task.get("name", "")
        assert "align" not in task.get("choice_label", "")


def _build_dummy_env():
    pegs = [
        _Actor([0.0, -0.15, 0.0]),
        _Actor([0.1, -0.25, 0.0]),
        _Actor([-0.1, -0.05, 0.0]),
    ]
    peg_heads = [
        _Actor([0.03, -0.15, 0.0]),
        _Actor([0.13, -0.25, 0.0]),
        _Actor([-0.07, -0.05, 0.0]),
    ]
    peg_tails = [
        _Actor([-0.03, -0.15, 0.0]),
        _Actor([0.07, -0.25, 0.0]),
        _Actor([-0.13, -0.05, 0.0]),
    ]
    return types.SimpleNamespace(
        device="cpu",
        end_steps=None,
        reset_in_proecess=False,
        table_scene=_TableScene(),
        _hb_generator=torch.Generator().manual_seed(0),
        radius=0.01,
        length=0.05,
        pegs=pegs,
        peg_heads=peg_heads,
        peg_tails=peg_tails,
        # pegs[0]/peg_heads[0]/peg_tails[0] are the same objects as peg/peg_head/peg_tail
        # so `id(self.peg) in {id(p) for p in self.pegs}` holds for label lookup.
        peg=pegs[0],
        peg_head=peg_heads[0],
        peg_tail=peg_tails[0],
        box=_Actor([0.0, 0.0, 0.04], yaw=np.pi / 2),
        agent=_Agent(),
    )
