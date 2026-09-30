# src/mme_vla_suite/training/dataset_lerobot.py
"""RoboMMEDatasetLerobot: MultiLeRobotDataset wrapper for the lerobot parquet format."""

import bisect
from functools import partial
import json
import logging
import math
from pathlib import Path
import random

from lerobot.datasets.multi_dataset import MultiLeRobotDataset
import ml_dtypes
import numpy as np
from omegaconf import DictConfig
import pandas as pd

from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from openpi.training import config as _config
from openpi.training.data_loader import Dataset

random.seed(0)
logger = logging.getLogger(__name__)


def discover_repo_ids(root: str | Path) -> list[str]:
    """Discover two-level repo_ids (``org/task``) under *root*.

    A directory is considered a valid lerobot dataset if it contains
    ``meta/info.json``.  Returns a sorted list for reproducibility.

    Raises:
        ValueError: if no valid datasets are found under *root*.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Discovery root '{root}' does not exist or is not a directory.")
    found: list[str] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "meta" / "info.json").exists():
            # One-level layout: root/task/meta/info.json → repo_id = "task"
            found.append(sub.name)
        else:
            # Two-level layout: root/org/task/meta/info.json → repo_id = "org/task"
            for task_dir in sorted(sub.iterdir()):
                if task_dir.is_dir() and (task_dir / "meta" / "info.json").exists():
                    found.append(f"{sub.name}/{task_dir.name}")
    return found


class RoboMMEDatasetLerobot(Dataset):
    """MultiLeRobotDataset wrapper exposing a torch ``Dataset`` interface.

    Reads state/action/text from LeRobot parquet files and loads per-episode
    bfloat16 SigLIP embeddings via memory-mapped numpy arrays.
    """

    def __init__(
        self,
        root: str | Path,
        data_config: _config.DataConfig,
        history_config: DictConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
    ):
        self._root = Path(root)
        repo_ids = discover_repo_ids(self._root)
        if not repo_ids:
            raise ValueError(
                f"No valid lerobot datasets found under '{root}'. "
                "Each dataset must contain 'meta/info.json' at 'root/org/task/meta/info.json'."
            )
        print(f"      Auto-discovered {len(repo_ids)} datasets under '{self._root}': {repo_ids}")

        self.history_config = history_config
        self.action_horizon = action_horizon
        self._repo_ids = repo_ids

        self.fps = self._read_dataset_fps(self._root, repo_ids)

        # Build MultiLeRobotDataset
        delta_timestamps = {"action": [i / self.fps for i in range(action_horizon)]}
        self.multi_dataset = MultiLeRobotDataset(
            repo_ids,
            root=self._root,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )

        # Per-dataset metadata
        self._emb_roots: list[Path] = []
        self._emb_infos: list[dict] = []
        self._summary_segments: list[dict[int, list[tuple[int, int, str]]]] = []
        self._video_description_lookup: list[dict[int, str]] = []
        self._frame_offsets: list[dict[int, int]] = []

        for repo_id in repo_ids:
            ds_root = self._root / repo_id
            with open(ds_root / "embeddings" / "info.json") as f:
                emb_info = json.load(f)
            self._emb_infos.append(emb_info)
            self._emb_roots.append(ds_root / "embeddings")
            self._summary_segments.append(self._load_summary_segments(ds_root / "meta" / "subtask_summaries.parquet"))
            self._video_description_lookup.append(
                self._load_video_description_lookup(ds_root / "meta" / "video_descriptions.parquet")
            )

            offsets_path = ds_root / "embeddings" / "frame_offsets.json"
            if offsets_path.exists():
                with open(offsets_path) as f:
                    raw = json.load(f)
                self._frame_offsets.append({int(k): v for k, v in raw.items()})
            else:
                self._frame_offsets.append({})

        # History config setup
        if self.history_config is not None:
            self._setup_history(self._emb_infos)
        else:
            logger.info("=== Do not use history ===")

        self.compute_norm_stats = compute_norm_stats
        if not compute_norm_stats:
            self.state_norm_stats = data_config.norm_stats["state"]
            self.use_quantiles = data_config.use_quantile_norm

        self.is_first_call = True

        self._mmap_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_dataset_fps(root: Path, repo_ids: list[str]) -> int:
        """Read fps from each dataset's ``meta/info.json`` and validate consistency.

        fps is stored in the LeRobot dataset metadata (the single source of truth). All
        co-trained datasets must share the same fps, since ``delta_timestamps`` (and thus the
        action-chunk time spacing) is a single schedule applied across the concatenated dataset.
        """
        fps_by_repo: dict[str, int] = {}
        for repo_id in repo_ids:
            info_path = root / repo_id / "meta" / "info.json"
            with open(info_path) as f:
                fps_by_repo[repo_id] = int(json.load(f)["fps"])
        unique_fps = set(fps_by_repo.values())
        if len(unique_fps) > 1:
            raise ValueError(f"Inconsistent fps across co-trained datasets: {fps_by_repo}")
        return next(iter(unique_fps))

    @staticmethod
    def _load_summary_segments(
        path: Path,
    ) -> dict[int, list[tuple[int, int, str]]]:
        """Load subtask_summaries.parquet into {episode_idx: sorted [(from, to, text)]}."""
        summaries_df = pd.read_parquet(path)
        segments: dict[int, list[tuple[int, int, str]]] = {}
        for row in summaries_df.itertuples(index=False):
            ep = int(row.episode_index)
            segments.setdefault(ep, []).append((int(row.from_frame_index), int(row.to_frame_index), row.summary))
        for segs in segments.values():
            segs.sort(key=lambda x: x[0])
        return segments

    @staticmethod
    def _load_video_description_lookup(path: Path) -> dict[int, str]:
        """Load video_descriptions.parquet into {video_description_index: text}."""

        if not path.exists():
            return {}

        df = pd.read_parquet(path)
        if "video_description_index" not in df.columns:
            return {}

        lookup: dict[int, str] = {}
        for text, idx in zip(df.index.tolist(), df["video_description_index"].tolist()):
            lookup[int(idx)] = str(text)
        return lookup

    def _setup_history(self, emb_infos: list[dict]) -> None:
        """Validate history_config against embedding files and initialize MemoryBuffer + pos_emb."""
        hc = self.history_config
        self.img_emb_dim = hc.memory_feature.img.input_dim
        self.pos_emb_dim = hc.memory_feature.pos.input_dim
        use_state_emb = getattr(hc, "use_state_emb", True)
        self.state_emb_dim = hc.memory_feature.state.input_dim if use_state_emb else 0
        assert self.state_emb_dim == 0, (
            "state_emb_dim > 0 (state history) is not supported for the lerobot dataset format"
        )
        self.num_views = hc.num_views
        self.streaming_obs_horizon = hc.streaming_obs_horizon

        aux_lang_task = getattr(hc, "aux_lang_task", None)
        if aux_lang_task is not None:
            assert aux_lang_task in ("subtask", "summary", "subtask_summary"), (
                f"aux_lang_task='{aux_lang_task}' is not supported for lerobot format; "
                "supported values: 'subtask', 'summary', 'subtask_summary'"
            )

        if hc.representation_type in ("perceptual", "selective"):
            memory_type_attr = f"{hc.representation_type}_memory"
            memory_cfg_type = getattr(hc, memory_type_attr).type
            token_per_image = hc.token_per_image

            for emb_info in emb_infos:
                assert len(emb_info["views"]) >= self.num_views, (
                    f"history_config.num_views={self.num_views} exceeds available views {emb_info['views']} in dataset"
                )
                if memory_cfg_type == "token_dropping":
                    assert emb_info["num_tokens"] == 64, (
                        f"token_dropping requires 8x8 embeddings (64 num_tokens), "
                        f"but dataset has num_tokens={emb_info['num_tokens']}"
                    )
                else:
                    assert emb_info["num_tokens"] == token_per_image, (
                        f"embedding num_tokens={emb_info['num_tokens']} does not match "
                        f"history_config.token_per_image={token_per_image}"
                    )

            spatial_size = int(math.sqrt(token_per_image))
            self._spatial_key = f"{spatial_size}x{spatial_size}"
            # Use first emb_info's view list (all datasets must have compatible views)
            self._selected_views = emb_infos[0]["views"][: self.num_views]
            self._pos_emb_dict = self._init_pos_emb_dict(self._spatial_key, spatial_size)
            self.mem_buffer = MemoryBuffer(
                num_views=self.num_views,
                img_emb_dim=self.img_emb_dim,
                pos_emb_dim=self.pos_emb_dim,
                state_emb_dim=self.state_emb_dim,
                token_drop_stride=self.streaming_obs_horizon // 2,
                prepare_buffer=False,
            )
        else:
            # no MemoryBuffer needed
            self.mem_buffer = None

    def _init_pos_emb_dict(self, spatial_key: str, spatial_size: int) -> dict[str, np.ndarray]:
        """Pre-compute positional embeddings for the needed spatial resolution."""
        from mme_vla_suite.shared.posemb_3d import PosEmb3D

        max_steps = 4096
        pos_embedder = PosEmb3D(dim=self.pos_emb_dim)
        ranges = np.arange(max_steps)
        return {spatial_key: np.array(pos_embedder(ranges, spatial_size))}

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def _load_emb_frames(
        self,
        dataset_idx: int,
        epis_idx: int,
        view: str,
        indices,
    ) -> np.ndarray:
        """Memory-map episode embedding file and return only the requested rows.

        Returns shape (len(indices), num_tokens, emb_dim) as bfloat16.
        """
        indices = np.asarray(indices)
        path = str(self._emb_roots[dataset_idx] / f"episode_{epis_idx}_{view}.npy")
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode="r")
        mmap_arr = self._mmap_cache[path]
        return mmap_arr[indices].view(ml_dtypes.bfloat16)

    def _frame_offset(self, dataset_idx: int, epis_idx: int) -> int:
        """Demo-prefix offset for an episode."""

        return self._frame_offsets[dataset_idx].get(epis_idx, 0)

    # ------------------------------------------------------------------
    # Summary lookup
    # ------------------------------------------------------------------

    def _lookup_summary(self, dataset_idx: int, epis_idx: int, step_idx: int) -> str | None:
        """Binary-search for the summary segment containing step_idx."""
        segments = self._summary_segments[dataset_idx].get(epis_idx, [])
        from_frames = [s[0] for s in segments]
        pos = bisect.bisect_right(from_frames, step_idx) - 1
        if pos >= 0 and segments[pos][0] <= step_idx <= segments[pos][1]:
            return segments[pos][2]
        return None

    @staticmethod
    def _as_scalar_int(value) -> int | None:
        """Best-effort conversion of scalar-like values into int."""

        if value is None:
            return None

        try:
            # Works for tensors, numpy scalars, and Python scalars.
            return int(value.item()) if hasattr(value, "item") else int(value)
        except (ValueError, TypeError):
            pass

        arr = np.asarray(value)
        if arr.size == 0:
            return None

        try:
            return int(arr.reshape(-1)[0])
        except (ValueError, TypeError):
            return None

    def _lookup_video_description(self, dataset_idx: int, item: dict) -> str | None:
        """Decode video_description_index into the original episode-level string."""

        vd_idx = self._as_scalar_int(item.get("video_description_index"))
        if vd_idx is None or vd_idx < 0:
            return None
        return self._video_description_lookup[dataset_idx].get(vd_idx)

    # ------------------------------------------------------------------
    # History feature gathering
    # ------------------------------------------------------------------

    def _gather_history_feat(self, dataset_idx: int, indices_to_load: list[int], epis_idx: int) -> dict[int, dict]:
        """Load embedding frames for requested step indices from mmap arrays."""

        # Load only the needed rows per view
        view_frames: dict[str, np.ndarray] = {
            view: self._load_emb_frames(dataset_idx, epis_idx, view, indices_to_load) for view in self._selected_views
        }
        # view_frames[view] shape: (len(indices_to_load), num_tokens, emb_dim) bfloat16

        history_feats: dict[int, dict] = {}
        for i, idx in enumerate(indices_to_load):
            history_feats[idx] = {
                f"image_emb_{self._spatial_key}": np.stack(
                    [view_frames[v][i] for v in self._selected_views], axis=0
                ),  # (num_views, num_tokens, emb_dim) bfloat16
                f"pos_emb_{self._spatial_key}": self._pos_emb_dict[self._spatial_key][
                    idx * self.num_views : (idx + 1) * self.num_views
                ],  # (num_views, num_tokens, pos_dim) float32
                "state_emb": np.zeros(self.state_emb_dim, dtype=np.float32),
            }
        return history_feats

    # ------------------------------------------------------------------
    # Aux_lang_task helpers
    # ------------------------------------------------------------------

    def _apply_multi_field_aux_lang_task(self, data: dict, fields: list[str], ratio: float) -> None:
        if random.uniform(0, 1) <= ratio:
            if self.is_first_call:
                print(
                    f"=== Using aux_lang_task fields={fields}, ratio={ratio}. "
                    f"Sampled: { {f: data[f] for f in fields} } ===",
                    flush=True,
                )
                self.is_first_call = False
        else:
            for f in fields:
                data[f] = None

    # ------------------------------------------------------------------
    # Static memory
    # ------------------------------------------------------------------

    def _apply_static_memory(self, data: dict, dataset_idx: int, epis_idx: int, step_idx: int) -> None:
        hc = self.history_config
        memory_type_attr = f"{hc.representation_type}_memory"
        memory_cfg_type = getattr(hc, memory_type_attr).type

        if memory_cfg_type == "frame_sampling":
            img_emb, pos_emb, state_emb, mask = self.mem_buffer.prepare_frame_sampling(
                step_idx,
                hc.dataset_token_budget,
                hc.token_per_image,
                partial(self._gather_history_feat, dataset_idx),
                epis_idx=epis_idx,
            )
        elif memory_cfg_type == "frame_sampling_every_n":
            img_emb, pos_emb, state_emb, mask = self.mem_buffer.prepare_frame_sampling_every_n(
                step_idx,
                hc.dataset_token_budget,
                hc.frame_sampling_step_size,
                hc.token_per_image,
                partial(self._gather_history_feat, dataset_idx),
                epis_idx=epis_idx,
            )
        elif memory_cfg_type == "token_dropping":
            kept_indices_path = (
                self._emb_roots[dataset_idx].parent / "features" / f"episode_{epis_idx}" / "kept_indices.json"
            )
            with open(kept_indices_path) as f:
                kept_indices = json.load(f)
            img_emb, pos_emb, state_emb, mask = self.mem_buffer.prepare_token_dropping(
                step_idx,
                hc.dataset_token_budget,
                partial(self._gather_history_feat, dataset_idx),
                kept_indices=kept_indices,
                epis_idx=epis_idx,
            )
        else:
            raise ValueError(
                f"Unsupported memory type '{memory_cfg_type}' for representation_type '{hc.representation_type}'"
            )

        data["static_image_emb"] = img_emb
        data["static_pos_emb"] = pos_emb
        data["static_state_emb"] = state_emb
        data["static_mask"] = mask

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.multi_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.multi_dataset[idx]

        epis_idx = int(item["episode_index"].item())
        step_idx = int(item["frame_index"].item())
        dataset_idx = int(item["dataset_index"].item())
        abs_step_idx = step_idx + self._frame_offset(dataset_idx, epis_idx)

        state = item["observation.state"]
        if hasattr(state, "numpy"):
            state = state.numpy()

        actions = item["action"][: self.action_horizon]
        if hasattr(actions, "numpy"):
            actions = actions.numpy()

        task_prompt = item.get("task")
        if isinstance(task_prompt, str):
            task_prompt = task_prompt.lower()
        video_description = self._lookup_video_description(dataset_idx, item)

        data: dict = {
            "actions": actions,
            "state": state,
            "epis_idx": np.array([epis_idx], dtype=np.int32),
            "step_idx": np.array([abs_step_idx], dtype=np.int32),
            "image": item.get("observation.images.front_rgb"),
            "wrist_image": item.get("observation.images.wrist_rgb"),
            "grounded_subgoal": item.get("subtask"),
            "simple_subgoal": None,
            "prompt": task_prompt,
            "video_description": video_description,
            "summary": self._lookup_summary(dataset_idx, epis_idx, step_idx),
        }

        # Aux lang task sampling
        hc = self.history_config
        aux_lang_task = getattr(hc, "aux_lang_task", None) if hc is not None else None
        aux_lang_task_sample_ratio = getattr(hc, "aux_lang_task_sample_ratio", 0) if hc is not None else 0

        _MULTI_FIELD_AUX_LANG_TASK = {
            "subtask": ["grounded_subgoal"],
            "summary": ["summary"],
            "subtask_summary": ["grounded_subgoal", "summary", "video_description"],
        }
        if hc is not None and aux_lang_task is not None:
            self._apply_multi_field_aux_lang_task(
                data, _MULTI_FIELD_AUX_LANG_TASK[aux_lang_task], aux_lang_task_sample_ratio
            )

        # Static memory
        if hc is not None and hc.representation_type in ("perceptual", "selective"):
            self._apply_static_memory(data, dataset_idx, epis_idx, abs_step_idx)

        # Fill missing keys with None
        for key in (
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            "simple_subgoal",
            "grounded_subgoal",
            "prompt",
            "video_description",
            "summary",
        ):
            if key not in data:
                data[key] = None

        return data
