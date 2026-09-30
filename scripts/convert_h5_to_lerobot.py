"""Convert RoboMME HDF5 datasets to LeRobot format with SigLIP embeddings.

Examples:
    # 8x8 embeddings for front camera (default):
    uv run python scripts/convert_h5_to_lerobot.py

    # 4x4 embeddings for both cameras:
    uv run python scripts/convert_h5_to_lerobot.py --resolution 4x4 \\
        --embedding_views front_rgb wrist_rgb

    # Cap at 5 episodes per file, custom paths:
    uv run python scripts/convert_h5_to_lerobot.py \\
        --raw_data_path data/robomme_data_h5 \\
        --output_path data/robomme_lerobot_4x4 \\
        --resolution 4x4 \\
        --max_episodes 5

    # Reconvert all datasets, ignoring existing outputs:
    uv run python scripts/convert_h5_to_lerobot.py --force

    # Use the hybrid text variant for subgoal/video_description/summary:
    uv run python scripts/convert_h5_to_lerobot.py \
        --raw_data_path ~/robomme_benchmark/runs/generated_dataset/train \
        --output_path data/robomme_lerobot_hybrid \
        --language_variant hybrid

    # Use the simple (non-grounded) text variant for subgoal/video_description/summary:
    uv run python scripts/convert_h5_to_lerobot.py --language_variant simple
"""

import dataclasses
import time

import tyro

from mme_vla_suite.dataset_builder.build_lerobot_dataset import LeRobotConverter


@dataclasses.dataclass
class Args:
    raw_data_path: str = "data/robomme_data_h5"
    """Directory containing *.h5 files."""

    output_path: str = "data/robomme_lerobot"
    """Root output directory; one sub-directory per H5 file."""

    resolution: str = "8x8"
    """SigLIP spatial pooling resolution. Choices: 2x2, 4x4, 8x8."""

    embedding_views: list[str] = dataclasses.field(default_factory=lambda: ["front_rgb"])
    """Camera views to embed. Any subset of: front_rgb, wrist_rgb."""

    fps: int = 30
    """Frame rate for LeRobot video encoding."""

    max_episodes: int | None = None
    """Cap episodes per H5 file (default: all)."""

    language_variant: str = "grounded"
    """Text variant to read for subgoal/video_description/summary. Choices: grounded, simple, hybrid."""

    force: bool = False
    """Reconvert all datasets even if they appear fully converted."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    t0 = time.perf_counter()
    converter = LeRobotConverter(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        resolution=args.resolution,
        embedding_views=args.embedding_views,
        fps=args.fps,
        max_episodes=args.max_episodes,
        language_variant=args.language_variant,
    )
    converter.run(force=args.force)
    print(f"\nTotal time: {(time.perf_counter() - t0) / 60:.2f} minutes")
