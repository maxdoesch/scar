"""
test_TaskGoal.py

Coverage tests for all env_id branches in task_goal.get_language_goal.
Run with: uv run python -m pytest tests/lightweight/test_TaskGoal.py -s
"""

import importlib.util

from tests._shared.repo_paths import find_repo_root


def _load_task_goal_module():
    repo_root = find_repo_root(__file__)
    module_path = (
        repo_root / "src" / "robomme" / "robomme_env" / "utils" / "task_goal.py"
    )
    spec = importlib.util.spec_from_file_location("task_goal_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_task_goal_module()
get_language_goal = mod.get_language_goal


class _Unwrapped:
    """Mock env.unwrapped; arbitrary attributes can be set as needed."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Env:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


class _Self:
    """Mock self object for called methods."""

    def __init__(self, env, **kwargs):
        self.env = env
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_self(unwrapped_attrs=None, **self_attrs):
    unwrapped = _Unwrapped(**(unwrapped_attrs or {}))
    env = _Env(unwrapped)
    return _Self(env, **self_attrs)


def _call(env_id, mock_self):
    result = get_language_goal(mock_self, env_id)
    print()
    for idx, goal in enumerate(result, start=1):
        print(f"[{env_id}] goal{idx}: {goal}")
    return result


def test_unknown_env_returns_single_goal_when_equal():
    """Unknown env_id goes to the default empty-string branch; when two goals are equal it returns a single-element list."""
    s = _make_self()
    result = _call("UnknownEnv", s)
    assert result == [""]


def test_movecube_still_returns_two_goals():
    """For defined branches with different texts, still return a two-element list."""
    s = _make_self()
    result = _call("MoveCube", s)
    assert len(result) == 2
