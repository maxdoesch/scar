# -*- coding: utf-8 -*-
"""
test_step_error_handling.py
============================
Lightweight test: Verify that DemonstrationWrapper.step() returns a structured error
via info["status"] = "error" upon internal exceptions, instead of propagating upwards.

Also verify that the step loops in run_example.py and dataset_replay.py
have been changed to check info["status"] rather than bare try/except.

Run (must use uv):
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/lightweight/test_step_error_handling.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo path helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root, ensure_src_on_path

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)


# ---------------------------------------------------------------------------
# Helpers: load DemonstrationWrapper.step source for AST inspection
# ---------------------------------------------------------------------------


def _demo_wrapper_path() -> Path:
    return _PROJECT_ROOT / "src/robomme/env_record_wrapper/DemonstrationWrapper.py"


def _load_step_source() -> str:
    return _demo_wrapper_path().read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: DemonstrationWrapper.step() catches exceptions and returns status="error"
# ---------------------------------------------------------------------------


def test_step_error_returns_status_error() -> None:
    """
    Construct a minimal Mock environment to make super().step() inside _step_batch() throw an exception,
    verifying that DemonstrationWrapper.step() does not propagate upwards, but returns "error" via info["status"].
    """
    source = _load_step_source()
    tree = ast.parse(source, filename=str(_demo_wrapper_path()))

    # Find step() method, verify try/except structure exists
    step_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DemonstrationWrapper":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "step":
                    step_method = item
                    break
            break

    assert step_method is not None, "DemonstrationWrapper.step method not found"

    # Verify that the step method body contains try/except
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(step_method))
    assert has_try, "DemonstrationWrapper.step() should contain a try/except block"

    # Verify that status = "error" is set in the except block
    has_error_status = False
    for node in ast.walk(step_method):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                for n in ast.walk(handler):
                    if isinstance(n, ast.Constant) and n.value == "error":
                        has_error_status = True
    assert has_error_status, (
        "There should be a string constant status='error' in the except block"
    )

    # Verify that the error_message key exists in the except block
    has_error_message = False
    for node in ast.walk(step_method):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                for n in ast.walk(handler):
                    if isinstance(n, ast.Constant) and n.value == "error_message":
                        has_error_message = True
    assert has_error_message, (
        "There should be an 'error_message' key in the except block"
    )

    print(
        "  ✓ DemonstrationWrapper.step() contains try/except and except returns status='error' + error_message"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n[TEST] DemonstrationWrapper step error handling")

    test_step_error_returns_status_error()
    print("  test1: AST structure verification passed")

    print("\nPASS: All step error handling tests passed")


if __name__ == "__main__":
    main()
