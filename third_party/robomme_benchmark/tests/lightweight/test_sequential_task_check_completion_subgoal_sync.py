"""Unit tests for completion-state grounding sync."""

import pytest

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.robomme_env.utils.subgoal_evaluate_func import sequential_task_check  # noqa: E402


class _Env:
    """Minimal env stub."""


@pytest.mark.parametrize(
    ("case", "expected_name"),
    [
        ("last-task-completes", "All tasks completed"),
        ("timestep-past-end", "All tasks completed"),
        ("empty-task-list", "No tasks"),
    ],
)
def test_completion_states_clear_stale_grounded_subgoal(case, expected_name):
    env = _Env()
    sentinel_segment = object()
    tasks = []

    if case == "last-task-completes":
        tasks = [
            {
                "func": lambda: True,
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "segment": sentinel_segment,
                "demonstration": False,
            }
        ]
    elif case == "timestep-past-end":
        tasks = [
            {
                "func": lambda: False,
                "name": "press the button",
                "subgoal_segment": "press the button at <>",
                "segment": sentinel_segment,
                "demonstration": False,
            }
        ]
        env.timestep = len(tasks)
    env.current_subgoal_segment = "press the button at <>"
    env.current_segment = sentinel_segment
    env.current_subgoal_segment_online = "press the button at <>"
    env.current_segment_online = sentinel_segment

    all_completed, current_task_name, task_failed, _ = sequential_task_check(
        env, tasks, allow_subgoal_change_this_timestep=True
    )

    assert all_completed is True
    assert task_failed is False
    assert current_task_name == expected_name
    assert env.current_task_name == expected_name
    assert env.current_subgoal_segment == expected_name
    assert env.current_segment is None
    assert env.current_task_name_online == expected_name
    assert env.current_subgoal_segment_online == expected_name
    assert env.current_segment_online is None
    assert env.current_subgoal_segment_hybrid == expected_name
    assert env.current_subgoal_segment_hybrid_online == expected_name
