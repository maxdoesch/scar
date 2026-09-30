"""Unit tests for summary slot replacement and summary_on_complete hook wiring."""

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.robomme_env.utils.subgoal_evaluate_func import (  # noqa: E402
    sequential_task_check,
)


class _Env:
    """Minimal env stub."""


def test_sequential_task_check_calls_summary_on_complete_once_on_advance():
    env = _Env()
    completed = {"count": 0}

    def _on_complete():
        completed["count"] += 1
        return ("task_one", "Task one done.", "Task one done <1,2>.")

    tasks = [
        {
            "func": lambda: True,
            "name": "task one",
            "demonstration": False,
            "summary_on_complete": _on_complete,
        },
        {
            "func": lambda: False,
            "name": "task two",
            "demonstration": False,
        },
    ]
    sequential_task_check(env, tasks, allow_subgoal_change_this_timestep=True)
    assert completed["count"] == 1
    assert env.summary_simple == "Task one done."


def test_summary_on_complete_suppressed_for_demonstration_task():
    """Demonstration tasks must not populate the summary even when they complete and advance."""
    env = _Env()
    completed = {"count": 0}

    def _on_complete():
        completed["count"] += 1
        return ("task_one", "Task one done.", "Task one done <1,2>.")

    tasks = [
        {
            "func": lambda: True,
            "name": "task one",
            "demonstration": True,
            "summary_on_complete": _on_complete,
        },
        {
            "func": lambda: False,
            "name": "task two",
            "demonstration": False,
        },
    ]
    all_completed, current_task_name, task_failed, _ = sequential_task_check(
        env, tasks, allow_subgoal_change_this_timestep=True
    )
    assert env.timestep == 1
    assert current_task_name == "task two"
    assert completed["count"] == 0
    assert not hasattr(env, "summary_simple") or env.summary_simple == ""


def test_four_tuple_summary_on_complete_sets_hybrid_slot():
    env = _Env()

    def _on_complete():
        return ("k", "Simple.", "Grounded <1,2>.", "Hybrid.")

    tasks = [
        {
            "func": lambda: True,
            "name": "t1",
            "demonstration": False,
            "summary_on_complete": _on_complete,
        },
        {"func": lambda: False, "name": "t2", "demonstration": False},
    ]
    sequential_task_check(env, tasks, allow_subgoal_change_this_timestep=True)
    assert env.summary_simple == "Simple."
    assert env.summary_hybrid == "Hybrid."


def test_hybrid_subgoal_segment_copied_to_env():
    env = _Env()
    tasks = [
        {
            "func": lambda: False,
            "name": "t1",
            "demonstration": False,
            "subgoal_segment": "pick up the cube at <>",
            "subgoal_segment_hybrid": "Pick up the cube.",
        },
    ]
    sequential_task_check(env, tasks, allow_subgoal_change_this_timestep=True)
    assert env.current_subgoal_segment_hybrid == "Pick up the cube."
    assert env.current_subgoal_segment_hybrid_online == "Pick up the cube."
