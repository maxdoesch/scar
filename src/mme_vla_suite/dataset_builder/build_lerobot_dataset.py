"""Convert raw HDF5 RoboMME datasets to LeRobot format with SigLIP embeddings.

Usage:
    uv run python scripts/convert_h5_to_lerobot.py --resolution 8x8

SigLIP embeddings are stored as bfloat16 .npy files (one per episode per view)
in a sibling embeddings/ directory, separate from the LeRobot Parquet/video data.
"""

from __future__ import annotations

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import shutil
import traceback

import h5py
import jax
import jax.numpy as jnp
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd

from mme_vla_suite.shared.data_utils import pool_tokens_to_size
from openpi.shared import image_tools

RESOLUTION_TO_NUM_TOKENS: dict[str, int] = {
    "8x8": 64,
    "4x4": 16,
    "2x2": 4,
}

SIGLIP_EMB_DIM = 2048
IMAGE_SHAPE = (256, 256, 3)
STATE_DIM = 8  # 7 joint + 1 gripper
ACTION_DIM = 8  # joint_action has dim 8

LANGUAGE_VARIANTS = ("grounded", "simple", "hybrid")


def _decode_h5_string(raw) -> str:
    """Decode an h5py scalar dataset value that may be ``bytes`` or already ``str``."""
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _read_optional_setup_string(episode_data: h5py.Group, key: str) -> str | None:
    """Read an optional scalar string dataset from ``episode_data['setup']``.

    Returns ``None`` if the key is absent (some episodes may not have a
    video description).
    """
    setup = episode_data["setup"]
    if key not in setup:
        return None
    return _decode_h5_string(setup[key][()])


def read_episode_frames(episode_data: h5py.Group, language_variant: str) -> tuple[str, str | None, list[dict]]:
    """Extract all frames from an H5 episode group.

    Args:
        episode_data: HDF5 group for one episode (e.g. ``f["episode_0"]``).
        language_variant: One of ``"grounded"``, ``"simple"``, or ``"hybrid"``.
            Selects which h5 text variant is read for the subgoal, video
            description, and summary fields (only the selected variant is ever
            read/stored).

    Returns:
        A tuple of:
        - ``task_goal``: the decoded task description string.
        - ``video_description``: decoded episode-level video description for
          the selected variant, or ``None`` if this episode has none (only
          "Video*" task types record this field).
        - ``frames``: list of frame dicts, one per timestep, each with keys
          ``front_rgb``, ``wrist_rgb``, ``state``, ``action``,
          ``is_video_demo``, ``subgoal``, ``summary``.
    """
    task_goal: str = episode_data["setup/task_goal"][0].decode()
    video_description = _read_optional_setup_string(episode_data, f"video_description_{language_variant}")

    subgoal_key = f"{language_variant}_subgoal"
    summary_key = f"summary_{language_variant}"

    timestep_keys = sorted(
        (k for k in episode_data.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )

    frames: list[dict] = []
    last_subgoal: str = ""
    last_summary: str = ""

    for ts_key in timestep_keys:
        ts = episode_data[ts_key]

        front_rgb: np.ndarray = ts["obs/front_rgb"][()]  # (256, 256, 3) uint8
        wrist_rgb: np.ndarray = ts["obs/wrist_rgb"][()]  # (256, 256, 3) uint8
        joint_state: np.ndarray = ts["obs/joint_state"][()].astype(np.float32)  # (7,)
        gripper: np.ndarray = ts["obs/gripper_state"][()].astype(np.float32)[:1]  # (1,)
        state = np.concatenate([joint_state, gripper], axis=0)  # (8,) float32
        action = ts["action/joint_action"][()].astype(np.float32)  # (8,) float32

        is_video_demo: bool = bool(ts["info/is_video_demo"][()])

        if subgoal_key in ts["info"]:
            last_subgoal = _decode_h5_string(ts["info"][subgoal_key][()])
        # else: last_subgoal retains the previous timestep's value (fallback)

        if summary_key in ts["info"]:
            last_summary = _decode_h5_string(ts["info"][summary_key][()])
        # else: last_summary retains the previous timestep's value (fallback)

        frames.append(
            {
                "front_rgb": front_rgb,
                "wrist_rgb": wrist_rgb,
                "state": state,
                "action": action,
                "is_video_demo": is_video_demo,
                "subgoal": last_subgoal,
                "summary": last_summary,
            }
        )

    return task_goal, video_description, frames


def _load_episode_frames(h5_path: str, ep_key: str, language_variant: str) -> tuple[str, str | None, list[dict]]:
    """Open *h5_path* in a fresh handle and return fully-loaded episode data.

    Designed for use in a prefetch thread — each call opens and closes its own
    h5py handle so the main thread's file object is not shared across threads.
    """
    with h5py.File(h5_path, "r") as f:
        return read_episode_frames(f[ep_key], language_variant)


def build_features() -> dict:
    """Build the LeRobot feature schema dict (no SigLIP — stored separately as .npy).

    Returns:
        LeRobot-compatible features dict.
    """
    return {
        "observation.images.front_rgb": {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "subtask_index": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["subtask_index"],
        },
        "video_description_index": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["video_description_index"],
        },
    }


def embed_frames(
    images: np.ndarray,
    resolution: str,
    vision_enc,
) -> np.ndarray:
    """Embed a batch of images with SigLIP and pool to the target resolution.

    Args:
        images: (T, H, W, 3) uint8 numpy array.
        resolution: One of ``"2x2"``, ``"4x4"``, ``"8x8"``.
        vision_enc: Callable (k, v, 224, 224, 3) jnp → (k, v, 256, 2048) bfloat16.
                    Typically ``jax.jit(SigLipTokenizer().__call__)``.

    Returns:
        bfloat16 numpy array of shape (T, num_tokens, SIGLIP_EMB_DIM).

    Raises:
        ValueError: If resolution is not supported.
    """
    if resolution not in RESOLUTION_TO_NUM_TOKENS:
        raise ValueError(f"Unsupported resolution '{resolution}'. Choose from: {list(RESOLUTION_TO_NUM_TOKENS)}")
    num_tokens = RESOLUTION_TO_NUM_TOKENS[resolution]

    # Process in fixed-size chunks to avoid JAX recompilation per episode length.
    # JAX JIT traces by shape; variable T causes recompilation. BOTH vision_enc
    # and resize_with_pad are @jax.jit, so every unique chunk size triggers TWO
    # recompilations and accumulates XLA compilation artifacts in GPU memory.
    # Fix: always pad the last chunk to chunk_size with zeros and slice results
    # afterward — JAX sees exactly ONE shape (chunk_size, H, W, 3) for every call.
    chunk_size = 64
    T = images.shape[0]
    chunks_out: list[np.ndarray] = []

    for start in range(0, T, chunk_size):
        chunk = images[start : start + chunk_size]  # (<=chunk_size, H, W, 3)
        actual_size = len(chunk)

        # Pad to chunk_size so JAX always sees the same shape (no recompilation).
        if actual_size < chunk_size:
            pad = np.zeros((chunk_size - actual_size, *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)

        # Normalize and resize to 224×224
        img_f = chunk.astype(np.float32) / 255.0 * 2.0 - 1.0
        img_jnp = jnp.array(img_f)
        img_jnp = image_tools.resize_with_pad(img_jnp, 224, 224)  # (chunk_size, 224, 224, 3)

        # SigLipTokenizer expects (k, v, 224, 224, 3); we pass k=chunk_size, v=1
        img_jnp_kv = img_jnp[:, None, ...]  # (chunk_size, 1, 224, 224, 3)
        output_emb = vision_enc(img_jnp_kv)  # (chunk_size, 1, 256, 2048)

        # Pool to target resolution, then discard padding rows
        pooled = pool_tokens_to_size(output_emb, target_size=num_tokens)  # (chunk_size, 1, num_tokens, 2048)
        pooled_np = jax.device_get(pooled)[:actual_size, 0, :, :]  # (actual_size, num_tokens, 2048)
        chunks_out.append(pooled_np)

    return np.concatenate(chunks_out, axis=0)  # (T, num_tokens, 2048) bfloat16


def _run_length_encode_summaries(summaries: list[str]) -> list[tuple[int, int, str]]:
    """Run-length encode a per-frame summary sequence into contiguous segments.

    Args:
        summaries: Per-frame summary strings, one per non-demo frame, indexed
            0..N-1 (matching LeRobot's own per-episode ``frame_index``).

    Returns:
        List of ``(from_frame_index, to_frame_index, text)`` tuples with an
        inclusive range, one per contiguous run of identical text. Runs whose
        text is the empty string are omitted, so frames before the first
        summary is produced correctly resolve to ``None`` at lookup time
        instead of an invented placeholder.
    """
    segments: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(summaries) + 1):
        if i == len(summaries) or summaries[i] != summaries[start]:
            if summaries[start] != "":
                segments.append((start, i - 1, summaries[start]))
            start = i
    return segments


class LeRobotConverter:
    """Convert a directory of HDF5 RoboMME files to per-file LeRobot datasets.

    Each H5 file becomes one LeRobot dataset at ``output_path/<h5_stem>/``.
    SigLIP embeddings are stored as bfloat16 .npy files in a sibling
    ``embeddings/`` directory (one file per episode per selected view), NOT
    inside the LeRobot Parquet data (bfloat16 is unsupported there).

    A single ``subgoal`` and a single episode-level ``video_description`` are
    read per episode, selected by ``language_variant`` (``"grounded"``,
    ``"simple"``, or ``"hybrid"``) — only that variant is ever read or stored,
    never both.
    They are stored following the LeRobot v3 convention used for
    ``task``/``tasks.parquet``:
    - ``subtask_index`` integer feature per frame, mapping into
      ``meta/subtasks.parquet``.
    - ``video_description_index`` integer feature per frame (constant within
      an episode, since video descriptions are episode-level), mapping into
      ``meta/video_descriptions.parquet``. Episodes without a video
      description use index ``-1`` (not present in the lookup table).

    The framewise ``summary`` field (also selected by ``language_variant``)
    is run-length-encoded per episode and written directly to
    ``meta/subtask_summaries.parquet`` as ``(episode_index, from_frame_index,
    to_frame_index, summary)`` rows — the schema already consumed by
    ``mme_vla_suite.training.dataset_lerobot.RoboMMEDatasetLerobot``. Frames
    whose summary is the empty string (i.e. before the first summary is
    produced) are omitted so lookups for those frames correctly resolve to
    ``None`` instead of an invented placeholder.

    Output layout::

        output_path/<dataset_name>/
            videos/               ← LeRobot MP4 videos
            data/                 ← LeRobot Parquet (state, action, *_index features)
            meta/
                tasks.parquet                ← task strings (managed by LeRobot)
                subtasks.parquet              ← subtask strings (managed by converter)
                video_descriptions.parquet    ← video description strings (managed by converter)
                subtask_summaries.parquet     ← per-episode summary segments (managed by converter)
                ...
            embeddings/
                info.json
                episode_0_front_rgb.npy   ← (T, N, 2048) bfloat16
                ...

    Args:
        raw_data_path: Directory containing ``*.h5`` files.
        output_path: Root output directory; one sub-dir per H5 file.
        resolution: Spatial pooling resolution: ``"2x2"``, ``"4x4"``, or ``"8x8"``.
        embedding_views: Camera names to embed. Subset of ``["front_rgb", "wrist_rgb"]``.
        fps: Frame rate for LeRobot video encoding.
        max_episodes: Cap episodes per H5 file (``None`` = all).
        language_variant: ``"grounded"``, ``"simple"``, or ``"hybrid"``.
            Selects which h5 text variant is used for subgoal,
            video_description, and summary.
    """

    def __init__(
        self,
        raw_data_path: str = "data/robomme_data_h5",
        output_path: str = "data/robomme_lerobot",
        resolution: str = "8x8",
        embedding_views: list[str] | None = None,
        fps: int = 30,
        max_episodes: int | None = None,
        language_variant: str = "grounded",
    ) -> None:
        if resolution not in RESOLUTION_TO_NUM_TOKENS:
            raise ValueError(f"Unsupported resolution '{resolution}'. Choose from: {list(RESOLUTION_TO_NUM_TOKENS)}")
        if language_variant not in LANGUAGE_VARIANTS:
            raise ValueError(f"Unsupported language_variant '{language_variant}'. Choose from: {LANGUAGE_VARIANTS}")
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.resolution = resolution
        self.embedding_views: list[str] = embedding_views or ["front_rgb"]
        self.fps = fps
        self.max_episodes = max_episodes
        self.language_variant = language_variant

    def _is_fully_converted(self, h5_path: str, out_dir: str) -> bool:
        """Return True only if ``out_dir`` contains a complete, valid conversion.

        Checks performed:
        - Output directory and ``embeddings/info.json`` exist.
        - Recorded resolution, embedding views, and language_variant match current settings.
        - Number of recorded episodes matches expected count from the H5 file.
        - All per-episode ``.npy`` embedding files are present.
        - LeRobot ``meta/info.json`` exists with a matching ``total_episodes``.
        - ``meta/subtasks.parquet``, ``meta/video_descriptions.parquet``, and
          ``meta/subtask_summaries.parquet`` all exist.
        """
        out_path = Path(out_dir)
        if not out_path.exists():
            return False

        emb_info_path = out_path / "embeddings" / "info.json"
        if not emb_info_path.exists():
            return False

        try:
            with open(emb_info_path) as fh:
                info = json.load(fh)
        except Exception:
            return False

        if info.get("resolution") != self.resolution:
            return False
        if set(info.get("views", [])) != set(self.embedding_views):
            return False
        if info.get("language_variant") != self.language_variant:
            return False

        try:
            with h5py.File(h5_path, "r") as fh:
                ep_keys = [k for k in fh.keys() if k.startswith("episode_")]
                expected = len(ep_keys)
                if self.max_episodes is not None:
                    expected = min(expected, self.max_episodes)
        except Exception:
            return False

        if info.get("num_episodes", -1) != expected:
            return False

        emb_dir = out_path / "embeddings"
        for ep_idx in range(expected):
            for view in self.embedding_views:
                if not (emb_dir / f"episode_{ep_idx}_{view}.npy").exists():
                    return False

        meta_info_path = out_path / "meta" / "info.json"
        if not meta_info_path.exists():
            return False

        try:
            with open(meta_info_path) as fh:
                meta = json.load(fh)
            if meta.get("total_episodes", -1) != expected:
                return False
        except Exception:
            return False

        for meta_name in (
            "subtasks.parquet",
            "video_descriptions.parquet",
            "subtask_summaries.parquet",
        ):
            if not (out_path / "meta" / meta_name).exists():
                return False

        if not (out_path / "embeddings" / "frame_offsets.json").exists():
            return False

        return True

    def run(self, force: bool = False) -> None:
        """Convert all H5 files found in ``raw_data_path``.

        Args:
            force: When ``True``, reconvert every dataset even if it appears
                   complete.  When ``False`` (default), skip datasets that pass
                   :meth:`_is_fully_converted`.
        """
        from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer  # deferred: triggers JAX/CUDA init

        h5_files = sorted(f for f in os.listdir(self.raw_data_path) if f.endswith(".h5"))
        if not h5_files:
            print("No .h5 files found in", self.raw_data_path)
            return

        pending: list[tuple[str, str, str]] = []
        for fname in h5_files:
            h5_path = os.path.join(self.raw_data_path, fname)
            dataset_name = Path(fname).stem
            out_dir = os.path.join(self.output_path, dataset_name)
            if not force and self._is_fully_converted(h5_path, out_dir):
                print(f"  [skip] {fname} — already fully converted at {out_dir}")
            else:
                pending.append((fname, h5_path, out_dir))

        if not pending:
            print("All datasets are up-to-date. Use --force to reconvert.")
            return

        print(f"\n{len(pending)} dataset(s) to convert (skipped {len(h5_files) - len(pending)}).")
        print("Initializing SigLIP tokenizer...")
        tokenizer = SigLipTokenizer()
        vision_enc = jax.jit(tokenizer.__call__)

        for fname, h5_path, out_dir in pending:
            dataset_name = Path(fname).stem
            print(f"\nConverting {fname} → {out_dir}")
            self._convert_file(h5_path, dataset_name, out_dir, vision_enc)

    def _convert_file(
        self,
        h5_path: str,
        dataset_name: str,
        out_dir: str,
        vision_enc,
    ) -> None:
        """Convert one H5 file to a LeRobot dataset + sibling embeddings/."""
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        features = build_features()
        dataset = LeRobotDataset.create(
            repo_id=f"robomme/{dataset_name}",
            fps=self.fps,
            features=features,
            root=out_dir,
            robot_type="robomme",
            use_videos=True,
        )

        # Create embeddings directory after LeRobot dataset dir exists
        emb_dir = os.path.join(out_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        # subtask_registry maps subgoal string -> integer index (LeRobot v3 convention)
        subtask_registry: dict[str, int] = {}
        # video_description_registry: same convention; episodes without one use index -1 (unregistered).
        video_description_registry: dict[str, int] = {}
        # Accumulates (episode_index, from_frame_index, to_frame_index, summary) rows,
        # written to meta/subtask_summaries.parquet at the end.
        summary_rows: list[tuple[int, int, int, str]] = []

        # Get episode keys without holding the file open for the whole conversion.
        with h5py.File(h5_path, "r") as f:
            episode_keys = sorted(
                (k for k in f.keys() if k.startswith("episode_")),
                key=lambda k: int(k.split("_")[1]),
            )
            if self.max_episodes is not None:
                episode_keys = episode_keys[: self.max_episodes]

        if not episode_keys:
            print("  No episodes found (max_episodes=0?), skipping.")
            return

        failed_episodes: list[str] = []
        frame_offsets: dict[int, int] = {}
        pending_saves: list[Future] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Prefetch first episode while the loop hasn't started yet.
            prefetch: Future = executor.submit(_load_episode_frames, h5_path, episode_keys[0], self.language_variant)

            for ep_idx, ep_key in enumerate(episode_keys):
                task_goal, video_description, frames = prefetch.result()

                # Immediately kick off prefetch for the next episode.
                if ep_idx + 1 < len(episode_keys):
                    prefetch = executor.submit(
                        _load_episode_frames, h5_path, episode_keys[ep_idx + 1], self.language_variant
                    )

                try:
                    demo_offset, save_futs = self._convert_episode(
                        task_goal,
                        video_description,
                        frames,
                        dataset,
                        vision_enc,
                        ep_key,
                        ep_idx,
                        emb_dir,
                        subtask_registry,
                        video_description_registry,
                        summary_rows,
                        executor,
                    )
                    frame_offsets[ep_idx] = demo_offset
                    pending_saves.extend(save_futs)
                except Exception as exc:
                    print(f"    ERROR in {ep_key}: {exc}")
                    traceback.print_exc()
                    dataset.clear_episode_buffer()
                    failed_episodes.append(ep_key)

            # Flush all async .npy writes before writing metadata.
            for fut in pending_saves:
                fut.result()

        if failed_episodes:
            print(f"  WARNING: {len(failed_episodes)} episode(s) failed: {failed_episodes}")

        def _write_registry_parquet(registry: dict[str, int], index_name: str, column_name: str, filename: str) -> None:
            """Write a string→index registry as a LeRobot-style dedup parquet (mirrors tasks.parquet).

            The integer id is a plain ``int64`` column and the string is the named
            row index (``str``), matching ``docs/metadata_fields.md``. dtypes are
            pinned explicitly so an *empty* registry (0 rows) still produces an
            ``int64`` column / ``str`` index instead of pandas' default
            ``float64`` / ``object`` for empty containers. The ids are passed as a
            plain numpy array (not a pandas Series) to avoid index-alignment
            against the string index, which would coerce the column to ``float64``.
            """
            registry_df = pd.DataFrame(
                {column_name: np.array(list(registry.values()), dtype=np.int64)},
                index=pd.Index(list(registry.keys()), name=index_name, dtype="str"),
            )
            path = Path(out_dir) / "meta" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            registry_df.to_parquet(path)

        _write_registry_parquet(subtask_registry, "subtask", "subtask_index", "subtasks.parquet")
        _write_registry_parquet(
            video_description_registry, "video_description", "video_description_index", "video_descriptions.parquet"
        )

        # Write per-episode summary segments (matches the schema read by
        # mme_vla_suite.training.dataset_lerobot.RoboMMEDatasetLerobot._load_summary_segments).
        summaries_df = pd.DataFrame(
            summary_rows,
            columns=["episode_index", "from_frame_index", "to_frame_index", "summary"],
        )
        summaries_path = Path(out_dir) / "meta" / "subtask_summaries.parquet"
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        summaries_df.to_parquet(summaries_path, index=False)

        # Write embedding metadata
        info = {
            "resolution": self.resolution,
            "num_tokens": RESOLUTION_TO_NUM_TOKENS[self.resolution],
            "embedding_dim": SIGLIP_EMB_DIM,
            "dtype": "bfloat16",
            "views": self.embedding_views,
            "num_episodes": len(episode_keys),
            "language_variant": self.language_variant,
        }
        with open(os.path.join(emb_dir, "info.json"), "w") as fh:
            json.dump(info, fh, indent=2)

        with open(os.path.join(emb_dir, "frame_offsets.json"), "w") as fh:
            json.dump({str(k): v for k, v in frame_offsets.items()}, fh, indent=2)

        dataset.finalize()
        total = dataset.meta.total_episodes
        print(
            f"  Done. {total} episodes, {len(subtask_registry)} unique subtasks, "
            f"{len(video_description_registry)} unique video descriptions, "
            f"{len(summary_rows)} summary segments → {out_dir}"
        )

    def _convert_episode(
        self,
        task_goal: str,
        video_description: str | None,
        frames: list[dict],
        dataset: LeRobotDataset,
        vision_enc,
        ep_key: str,
        ep_idx: int,
        emb_dir: str,
        subtask_registry: dict[str, int],
        video_description_registry: dict[str, int],
        summary_rows: list[tuple[int, int, int, str]],
        save_executor: ThreadPoolExecutor,
    ) -> tuple[int, list[Future]]:
        """Convert one episode: save .npy embeddings for all frames, add only non-demo frames to LeRobot.

        Returns:
            A tuple of:
            - ``demo_offset``: number of leading video-demo frames. The embedding index
              for LeRobot frame j is ``j + demo_offset``.
            - ``save_futures``: futures for the async .npy writes (caller must ``.result()`` them).

        Side effects:
            Appends ``(episode_index, from_frame_index, to_frame_index, summary)``
            tuples to ``summary_rows`` for every non-empty run-length-encoded
            summary segment in this episode (local frame indices, matching the
            non-demo frames actually written to the LeRobot dataset).
        """
        demo_offset = next((i for i, f in enumerate(frames) if not f["is_video_demo"]), len(frames))
        non_demo_frames = [f for f in frames if not f["is_video_demo"]]

        # Verify the prefix invariant: demo frames must form a contiguous prefix
        actual_demo_count = len(frames) - len(non_demo_frames)
        if demo_offset != actual_demo_count:
            raise ValueError(
                f"{ep_key}: video-demo frames are not a contiguous prefix "
                f"(prefix length={demo_offset}, total demo frames={actual_demo_count}). "
                "Interleaved video-demo frames are not supported."
            )

        print(
            f"    {ep_key}: {len(frames)} frames total "
            f"({demo_offset} video-demo, {len(non_demo_frames)} non-demo), embedding...",
            flush=True,
        )

        # Embed ALL frames (including video-demo) — batch all views into one GPU call.
        # Stack: (num_views * T, H, W, 3), embed once, then split by view.
        T = len(frames)
        view_images = [np.stack([f[view] for f in frames]) for view in self.embedding_views]
        all_images = np.concatenate(view_images, axis=0)  # (num_views * T, H, W, 3)
        all_embs = embed_frames(all_images, self.resolution, vision_enc)  # (num_views * T, num_tokens, 2048)

        save_futs: list[Future] = []
        for i, view in enumerate(self.embedding_views):
            emb = all_embs[i * T : (i + 1) * T]
            npy_path = os.path.join(emb_dir, f"episode_{ep_idx}_{view}.npy")
            save_futs.append(save_executor.submit(np.save, npy_path, emb.copy()))
            print(f"    {ep_key}: queued {view} embedding save {emb.shape}", flush=True)

        gc.collect()

        # Register subtasks only from non-demo frames
        for frame in non_demo_frames:
            sg = frame["subgoal"]
            if sg not in subtask_registry:
                subtask_registry[sg] = len(subtask_registry)

        # Video descriptions are episode-level (constant across the episode); register once
        # and reuse the same index for every frame. Absent -> sentinel index -1 (unregistered).
        if video_description is not None:
            if video_description not in video_description_registry:
                video_description_registry[video_description] = len(video_description_registry)
            video_description_index = video_description_registry[video_description]
        else:
            video_description_index = -1

        # Run-length encode the framewise summary over non-demo frames (local frame index
        # 0..N-1) and record segments for meta/subtask_summaries.parquet.
        for from_idx, to_idx, summary_text in _run_length_encode_summaries(
            [frame["summary"] for frame in non_demo_frames]
        ):
            summary_rows.append((ep_idx, from_idx, to_idx, summary_text))

        # Guard: skip save if no non-demo frames to write
        if not non_demo_frames:
            print(
                f"    {ep_key}: SKIPPING save — all {len(frames)} frames are video-demo, no non-demo frames to write.",
                flush=True,
            )
            return demo_offset, save_futs

        print(f"    {ep_key}: writing {len(non_demo_frames)} non-demo frames...", flush=True)
        for frame in non_demo_frames:
            dataset.add_frame(
                {
                    "observation.images.front_rgb": frame["front_rgb"],
                    "observation.images.wrist_rgb": frame["wrist_rgb"],
                    "observation.state": frame["state"],
                    "action": frame["action"],
                    "subtask_index": np.array([subtask_registry[frame["subgoal"]]], dtype=np.int32),
                    "video_description_index": np.array([video_description_index], dtype=np.int32),
                    "task": task_goal,
                }
            )

        print(f"    {ep_key}: saving episode (encoding video)...", flush=True)
        dataset.save_episode(parallel_encoding=True)
        print(f"    {ep_key}: done. task='{task_goal}'", flush=True)

        return demo_offset, save_futs
