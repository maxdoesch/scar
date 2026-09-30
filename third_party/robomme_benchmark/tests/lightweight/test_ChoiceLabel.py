import importlib.util

import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = [pytest.mark.lightweight, pytest.mark.gpu]


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


matcher_mod = _load_module(
    "oracle_action_matcher_under_test",
    "src/robomme/robomme_env/utils/oracle_action_matcher.py",
)


def test_map_action_text_to_option_label_strict_exact():
    options = [
        {"label": "a", "action": "pick up the cube"},
        {"label": "b", "action": "put it down"},
    ]

    assert (
        matcher_mod.map_action_text_to_option_label("pick up the cube", options) == "a"
    )
    assert (
        matcher_mod.map_action_text_to_option_label("unknown action", options) is None
    )
    assert matcher_mod.map_action_text_to_option_label(None, options) is None
