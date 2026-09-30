"""Unit tests for RoboMMEDatasetLerobot."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import ml_dtypes
import numpy as np
import pandas as pd
import pytest

NUM_TOKENS = 16  # 4x4
EMB_DIM = 2048
NUM_FRAMES = 20
STATE_DIM = 8
ACTION_DIM = 8


@pytest.fixture
def fake_ds_root(tmp_path: Path) -> Path:
    """Create a minimal fake LeRobot dataset on disk."""
    repo_id = "fake_dataset"
    ds_root = tmp_path / repo_id

    # meta/info.json
    (ds_root / "meta").mkdir(parents=True)
    (ds_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "total_episodes": 2,
                "total_frames": NUM_FRAMES * 2,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                    "action": {"dtype": "float32", "shape": [ACTION_DIM]},
                    "subtask_index": {"dtype": "int32", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": []},
                },
            }
        )
    )

    # meta/subtask_summaries.parquet
    summaries_df = pd.DataFrame(
        [
            {"episode_index": 0, "from_frame_index": 0, "to_frame_index": 9, "summary": "first half"},
            {"episode_index": 0, "from_frame_index": 10, "to_frame_index": 19, "summary": "second half"},
            {"episode_index": 1, "from_frame_index": 0, "to_frame_index": 19, "summary": "whole ep1"},
        ]
    )
    summaries_df.to_parquet(ds_root / "meta" / "subtask_summaries.parquet", index=False)

    # embeddings/frame_offsets.json — offset 0 for both episodes (no video-demo prefix)
    (ds_root / "embeddings").mkdir()
    (ds_root / "embeddings" / "frame_offsets.json").write_text(json.dumps({"0": 0, "1": 0}))

    # embeddings/info.json
    (ds_root / "embeddings" / "info.json").write_text(
        json.dumps(
            {
                "resolution": "4x4",
                "num_tokens": NUM_TOKENS,
                "embedding_dim": EMB_DIM,
                "dtype": "bfloat16",
                "views": ["front_rgb"],
                "num_episodes": 2,
            }
        )
    )

    # embeddings/episode_{i}_front_rgb.npy  — bfloat16 stored as void bytes
    rng = np.random.default_rng(42)
    for ep in range(2):
        data = rng.standard_normal((NUM_FRAMES, NUM_TOKENS, EMB_DIM)).astype(np.float32)
        bf16 = data.astype(ml_dtypes.bfloat16)
        np.save(ds_root / "embeddings" / f"episode_{ep}_front_rgb.npy", bf16)

    return tmp_path  # return root containing repo_id subdir


def _make_dataset_no_history(fake_ds_root, tmp_path):
    """Helper: build RoboMMEDatasetLerobot with no history_config."""
    from mme_vla_suite.training.dataset_lerobot import RoboMMEDatasetLerobot

    mock_data_config = MagicMock()
    mock_data_config.norm_stats = {"state": MagicMock(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM))}
    mock_data_config.use_quantile_norm = False

    with patch("mme_vla_suite.training.dataset_lerobot.MultiLeRobotDataset") as MockMulti:
        mock_multi = MagicMock()
        mock_multi.__len__ = MagicMock(return_value=NUM_FRAMES * 2)
        MockMulti.return_value = mock_multi
        ds = RoboMMEDatasetLerobot(
            root=fake_ds_root,
            data_config=mock_data_config,
            history_config=None,
            action_horizon=4,
        )
    return ds


def test_delta_timestamps_use_fps_from_dataset_metadata(fake_ds_root):
    """fps for delta_timestamps must be read from the dataset's meta/info.json, not a config."""
    from mme_vla_suite.training.dataset_lerobot import RoboMMEDatasetLerobot

    # Override the fixture's recorded fps to a distinct value to prove it is sourced from metadata.
    info_path = fake_ds_root / "fake_dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = 50
    info_path.write_text(json.dumps(info))

    mock_data_config = MagicMock()
    mock_data_config.norm_stats = {"state": MagicMock(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM))}
    mock_data_config.use_quantile_norm = False

    with patch("mme_vla_suite.training.dataset_lerobot.MultiLeRobotDataset") as MockMulti:
        MockMulti.return_value.__len__ = MagicMock(return_value=NUM_FRAMES * 2)
        ds = RoboMMEDatasetLerobot(
            root=fake_ds_root,
            data_config=mock_data_config,
            history_config=None,
            action_horizon=4,
        )

    assert ds.fps == 50
    _, kwargs = MockMulti.call_args
    dt = kwargs["delta_timestamps"]
    assert dt["action"] == [i / 50 for i in range(4)]


def test_load_emb_frames_returns_correct_rows(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)

    indices = [0, 5, 10]
    frames = ds._load_emb_frames(0, 0, "front_rgb", indices)

    assert frames.shape == (3, NUM_TOKENS, EMB_DIM)
    assert frames.dtype == ml_dtypes.bfloat16

    # Verify row values match what was written
    rng = np.random.default_rng(42)
    full = rng.standard_normal((NUM_FRAMES, NUM_TOKENS, EMB_DIM)).astype(np.float32).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(frames[0], full[0])
    np.testing.assert_array_equal(frames[1], full[5])
    np.testing.assert_array_equal(frames[2], full[10])


def test_load_emb_frames_applies_demo_offset(tmp_path):
    """_load_emb_frames must shift indices by the demo_offset for the episode."""
    import json

    import ml_dtypes
    import numpy as np

    NUM_TOKENS = 16
    EMB_DIM = 2048
    N_DEMO = 5
    N_NONDEMO = 10
    T_TOTAL = N_DEMO + N_NONDEMO

    repo_id = "offset_dataset"
    ds_root = tmp_path / repo_id
    (ds_root / "meta").mkdir(parents=True)
    (ds_root / "meta" / "info.json").write_text(
        json.dumps({"fps": 20, "total_episodes": 1, "total_frames": N_NONDEMO, "features": {}})
    )
    summaries_df = pd.DataFrame(
        [{"episode_index": 0, "from_frame_index": 0, "to_frame_index": N_NONDEMO - 1, "summary": "all"}]
    )
    summaries_df.to_parquet(ds_root / "meta" / "subtask_summaries.parquet", index=False)

    (ds_root / "embeddings").mkdir()
    (ds_root / "embeddings" / "frame_offsets.json").write_text(json.dumps({"0": N_DEMO}))
    (ds_root / "embeddings" / "info.json").write_text(
        json.dumps(
            {
                "resolution": "4x4",
                "num_tokens": NUM_TOKENS,
                "embedding_dim": EMB_DIM,
                "dtype": "bfloat16",
                "views": ["front_rgb"],
                "num_episodes": 1,
            }
        )
    )

    # Embedding file: T_TOTAL rows, each row filled with its original index value for easy verification
    full_emb = np.zeros((T_TOTAL, NUM_TOKENS, EMB_DIM), dtype=ml_dtypes.bfloat16)
    for i in range(T_TOTAL):
        full_emb[i] = ml_dtypes.bfloat16(float(i))
    np.save(ds_root / "embeddings" / "episode_0_front_rgb.npy", full_emb)

    from unittest.mock import MagicMock
    from unittest.mock import patch

    from mme_vla_suite.training.dataset_lerobot import RoboMMEDatasetLerobot

    mock_data_config = MagicMock()
    mock_data_config.norm_stats = {"state": MagicMock(mean=np.zeros(8), std=np.ones(8))}
    mock_data_config.use_quantile_norm = False

    with patch("mme_vla_suite.training.dataset_lerobot.MultiLeRobotDataset"):
        ds = RoboMMEDatasetLerobot(
            root=tmp_path,
            data_config=mock_data_config,
            history_config=None,
            action_horizon=4,
        )

    # Current design: the demo offset is applied UPSTREAM (in __getitem__ via
    # abs_step_idx = step_idx + _frame_offset), and _load_emb_frames indexes the
    # mmap directly. Verify both pieces compose correctly.
    offset = ds._frame_offset(0, 0)
    assert offset == N_DEMO

    # _load_emb_frames indexes directly (no internal offset shift)
    direct = ds._load_emb_frames(0, 0, "front_rgb", [0, 3])
    assert float(direct[0, 0, 0]) == 0.0
    assert float(direct[1, 0, 0]) == 3.0

    # Applying the offset upstream (as __getitem__ does) yields the demo-shifted rows.
    # LeRobot frame 0 → embedding index N_DEMO (value = 5.0)
    # LeRobot frame 3 → embedding index N_DEMO+3 (value = 8.0)
    result = ds._load_emb_frames(0, 0, "front_rgb", [0 + offset, 3 + offset])
    assert result.shape == (2, NUM_TOKENS, EMB_DIM)
    assert float(result[0, 0, 0]) == float(N_DEMO)
    assert float(result[1, 0, 0]) == float(N_DEMO + 3)


def test_lookup_summary_returns_correct_segment(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)

    assert ds._lookup_summary(0, 0, 0) == "first half"
    assert ds._lookup_summary(0, 0, 9) == "first half"
    assert ds._lookup_summary(0, 0, 10) == "second half"
    assert ds._lookup_summary(0, 0, 19) == "second half"
    assert ds._lookup_summary(0, 1, 15) == "whole ep1"


def test_lookup_summary_returns_none_for_missing(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)
    # episode 99 does not exist
    assert ds._lookup_summary(0, 99, 0) is None


# ---- helpers for history config mocks ----


def _make_history_config(
    representation_type="perceptual",
    memory_type="frame_sampling",
    token_per_image=16,
    num_views=1,
    state_emb_dim=0,
    aux_lang_task=None,
):
    hc = MagicMock()
    hc.representation_type = representation_type
    hc.memory_feature.img.input_dim = 2048
    hc.memory_feature.pos.input_dim = 768
    hc.memory_feature.state.input_dim = state_emb_dim
    hc.num_views = num_views
    hc.streaming_obs_horizon = 16
    hc.token_per_image = token_per_image
    hc.dataset_token_budget = 64
    hc.perceptual_memory.type = memory_type
    hc.selective_memory.type = memory_type
    hc.aux_lang_task = aux_lang_task
    return hc


def _build_ds(fake_ds_root, history_config):
    from mme_vla_suite.training.dataset_lerobot import RoboMMEDatasetLerobot

    mock_data_config = MagicMock()
    mock_data_config.norm_stats = {"state": MagicMock(mean=np.zeros(STATE_DIM), std=np.ones(STATE_DIM))}
    mock_data_config.use_quantile_norm = False

    with patch("mme_vla_suite.training.dataset_lerobot.MultiLeRobotDataset") as MockMulti:
        MockMulti.return_value.__len__ = MagicMock(return_value=NUM_FRAMES * 2)
        with patch("mme_vla_suite.training.dataset_lerobot.RoboMMEDatasetLerobot._init_pos_emb_dict") as mock_pos:
            mock_pos.return_value = {"4x4": np.zeros((4096, NUM_TOKENS, 768), dtype=np.float32)}
            return RoboMMEDatasetLerobot(
                root=fake_ds_root,
                data_config=mock_data_config,
                history_config=history_config,
                action_horizon=4,
            )


def test_assert_state_emb_dim_raises(fake_ds_root):
    hc = _make_history_config(state_emb_dim=8)
    with pytest.raises(AssertionError, match="state_emb_dim"):
        _build_ds(fake_ds_root, hc)


def test_assert_unsupported_aux_lang_task_raises(fake_ds_root):
    hc = _make_history_config(aux_lang_task="unsupported_mode")
    with pytest.raises(AssertionError, match="aux_lang_task"):
        _build_ds(fake_ds_root, hc)


def test_assert_token_mismatch_raises(fake_ds_root):
    # dataset has 16 tokens (4x4) but config says 64
    hc = _make_history_config(token_per_image=64)
    with pytest.raises(AssertionError, match="num_tokens"):
        _build_ds(fake_ds_root, hc)


def test_assert_token_dropping_wrong_num_tokens_raises(fake_ds_root):
    # dataset has 16 tokens (4x4) != 64 required by token_dropping
    hc = _make_history_config(memory_type="token_dropping", token_per_image=16)
    with pytest.raises(AssertionError, match="num_tokens"):
        _build_ds(fake_ds_root, hc)


def test_assert_num_views_exceeds_available_raises(fake_ds_root):
    # dataset has 1 view, config says 2
    hc = _make_history_config(num_views=2)
    with pytest.raises(AssertionError, match="num_views"):
        _build_ds(fake_ds_root, hc)


def _make_mock_item(ep_idx=0, frame_idx=5, dataset_idx=0, subtask="grasp block", task="pick up block"):
    import torch

    rng = np.random.default_rng(0)
    return {
        "observation.state": torch.tensor(rng.standard_normal(STATE_DIM).astype(np.float32)),
        "action": torch.tensor(rng.standard_normal((8, ACTION_DIM)).astype(np.float32)),
        "episode_index": torch.tensor(ep_idx),
        "frame_index": torch.tensor(frame_idx),
        "dataset_index": torch.tensor(dataset_idx),
        "subtask": subtask,
        "task": task,
        "video_description_index": torch.tensor(0),
    }


def test_len(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)
    assert len(ds) == NUM_FRAMES * 2


def test_getitem_no_history_keys(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)
    ds.multi_dataset.__getitem__ = MagicMock(return_value=_make_mock_item())
    ds._video_description_lookup[0] = {0: "episode overview"}

    item = ds[0]

    # All expected keys present
    for key in (
        "actions",
        "state",
        "epis_idx",
        "step_idx",
        "grounded_subgoal",
        "simple_subgoal",
        "prompt",
        "video_description",
        "summary",
        "static_image_emb",
        "static_pos_emb",
        "static_state_emb",
        "static_mask",
    ):
        assert key in item, f"missing key: {key}"

    assert item["grounded_subgoal"] == "grasp block"
    assert item["prompt"] == "pick up block"
    assert item["video_description"] == "episode overview"
    assert item["simple_subgoal"] is None
    # frame_index=5 falls in "first half" segment (0-9) of episode 0
    assert item["summary"] == "first half"
    # memory keys are None without history
    assert item["static_image_emb"] is None


def test_getitem_video_description_sentinel_returns_none(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)
    item = _make_mock_item()
    item["video_description_index"] = np.array([-1], dtype=np.int32)
    ds.multi_dataset.__getitem__ = MagicMock(return_value=item)
    ds._video_description_lookup[0] = {0: "episode overview"}

    output = ds[0]

    assert output["video_description"] is None


def test_getitem_actions_truncated(fake_ds_root):
    ds = _make_dataset_no_history(fake_ds_root, None)
    ds.multi_dataset.__getitem__ = MagicMock(return_value=_make_mock_item())
    item = ds[0]
    assert item["actions"].shape == (4, ACTION_DIM)  # action_horizon=4


def test_gather_history_feat_shapes(fake_ds_root):
    hc = _make_history_config()
    ds = _build_ds(fake_ds_root, hc)

    indices = [0, 3, 7]
    feats = ds._gather_history_feat(0, indices, epis_idx=0)

    assert set(feats.keys()) == {0, 3, 7}
    for idx in indices:
        feat = feats[idx]
        assert "image_emb_4x4" in feat
        assert "pos_emb_4x4" in feat
        assert "state_emb" in feat
        assert feat["image_emb_4x4"].shape == (1, NUM_TOKENS, EMB_DIM)  # num_views=1
        assert feat["pos_emb_4x4"].shape == (1, NUM_TOKENS, 768)
        assert feat["state_emb"].shape == (0,)  # state_emb_dim=0
        assert feat["image_emb_4x4"].dtype == ml_dtypes.bfloat16


def test_getitem_with_frame_sampling_history(fake_ds_root):
    """static_image_emb and static_mask should have correct sizes."""
    hc = _make_history_config(
        representation_type="perceptual",
        memory_type="frame_sampling",
        token_per_image=NUM_TOKENS,
    )
    # token_budget=64, token_per_image=16, num_views=1 → max 4 frames
    hc.dataset_token_budget = 64

    ds = _build_ds(fake_ds_root, hc)
    ds.multi_dataset.__getitem__ = MagicMock(return_value=_make_mock_item(ep_idx=0, frame_idx=10))

    item = ds[0]

    assert item["static_image_emb"] is not None
    assert item["static_pos_emb"] is not None
    assert item["static_mask"] is not None
    assert isinstance(item["static_state_emb"], np.ndarray)
    assert item["static_state_emb"].shape[-1] == 0

    # shape: (token_budget, emb_dim)
    assert item["static_image_emb"].shape[1] == EMB_DIM
    assert item["static_image_emb"].shape[0] == item["static_pos_emb"].shape[0]
    assert item["static_image_emb"].shape[0] == item["static_mask"].shape[0]
