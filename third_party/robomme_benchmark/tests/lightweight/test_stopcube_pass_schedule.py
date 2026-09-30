import pytest

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.robomme_env.StopCube import build_stopcube_pass_schedule  # noqa: E402


@pytest.mark.parametrize(
    ("stop_time", "move_interval", "expected"),
    [
        (
            2,
            60,
            [
                (
                    1,
                    30.0,
                    "remain static",
                    "remain static",
                    "Remain static.",
                )
            ],
        ),
        (
            4,
            80,
            [
                (
                    1,
                    40.0,
                    "remain static",
                    "remain static",
                    "Remain static.",
                ),
                (
                    2,
                    120.0,
                    "remain static",
                    "remain static",
                    "Remain static.",
                ),
                (
                    3,
                    200.0,
                    "remain static",
                    "remain static",
                    "Remain static.",
                ),
            ],
        ),
    ],
)
def test_build_stopcube_pass_schedule_returns_expected_entries(
    stop_time, move_interval, expected
):
    assert (
        build_stopcube_pass_schedule(stop_time=stop_time, move_interval=move_interval)
        == expected
    )
