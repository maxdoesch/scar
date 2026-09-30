"""Single entrypoint to run dataset builders (RoboMME).

Build RoboMME preprocessed pickle data from raw HDF5 data.
```
uv run python scripts/build_dataset.py --dataset_type robomme_pkl
```

"""

import argparse
import time

from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw HDF5 dataset for training")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="robomme_pkl",
        choices=["robomme_pkl"],
        help="Dataset type to build",
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/robomme_data_h5",
        help="Raw HDF5 directory",
    )
    parser.add_argument(
        "--preprocessed_data_path",
        type=str,
        default="data/robomme_preprocessed_data",
        help="Output directory",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap episodes per file (default: all)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write visualization MP4s",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t0 = time.perf_counter()

    if args.dataset_type == "robomme_pkl":
        processor = DatasetProcessor(
            raw_data_path=args.raw_data_path,
            preprocessed_data_path=args.preprocessed_data_path,
            visualize=args.visualize,
            max_episodes=args.max_episodes,
        )
        processor.run()
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    print(f"Time taken: {(time.perf_counter() - t0) / 60:.2f} minutes")
