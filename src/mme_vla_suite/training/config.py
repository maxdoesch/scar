import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
import sentencepiece
from typing_extensions import override
import tyro

from mme_vla_suite.models.config.history_config_utils import get_history_config
from mme_vla_suite.models.config.history_config_utils import resolve_max_token_len
from mme_vla_suite.models.integration import history_pi0
from mme_vla_suite.policies.robomme_policy import RoboMMEInputs
from mme_vla_suite.policies.robomme_policy import RoboMMEOutputs
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets dir to load from (e.g. a base-model checkpoint); falls back to the config assets_dirs.
    assets_dir: str | None = None

    # Asset id to reference (e.g. a different robot platform); falls back to the repo id.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats; skipped if None.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Adapts inputs from a dataset-specific format to the common format expected by data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Robot-specific transforms, applied before normalization. See `model.Observation`/`model.Actions`.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model-specific transforms, applied after normalization.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, use quantile normalization instead of z-score normalization.
    use_quantile_norm: bool = False

    # Keys used to build the action sequence (length = `action_horizon` in the model config).
    # Adjust if your LeRobot dataset uses different action keys.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        subgoal: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            if subgoal is None:
                # Pi05 format: state is part of the discrete language input.
                state = np.clip(state, -1, 1)
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}; State: {state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            else:
                # Real-robot subgoal-conditioned variant (no proprioceptive state used).
                subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
                full_prompt = f"Task: {cleaned_text}\nCurrent Subgoal: {subgoal}.\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)

        elif subgoal is not None:
            # Subgoal format: subgoal is part of the language input (simple or grounded).
            subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
            full_prompt = f"Task: {cleaned_text};\nCurrent Subgoal: {subgoal};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format: state is part of the continuous action-expert input.
            # Tokenize "\n" separately as the "start of answer" token.
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)

        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


_AUX_LANG_TASK_FORMAT: dict[str, dict[str, Any]] = {
    "subtask": {
        "target_prefixes": ["Subtask: "],
        "target_fields": ["grounded_subgoal"],
    },
    "summary": {
        "target_prefixes": ["Demonstration: ", "Summary: "],
        "target_fields": ["video_description", "summary"],
    },
    "subtask_summary": {
        "target_prefixes": ["Demonstration: ", "Summary: ", "Subtask: "],
        "target_fields": ["video_description", "summary", "grounded_subgoal"],
    },
}


class PaligemmaTokenizerAuxLangTask:
    # Segment ids for aux-lang-task attention are defined centrally in
    # history_pi0.py (SEG_PROMPT / SEG_AUX_TARGET) and drive the
    # segment-allow attention matrix.

    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        mode: str | None = None,
        target_segments: list[tuple[str, str]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt, optionally with an auxiliary language-task segment.

        Args:
            prompt: Task description.
            state: Optional robot state for discretized state conditioning.
            mode: One of ``"subtask"``, ``"summary"``, ``"subtask_summary"`` or ``None``.
                Determines the prefix(es) used for each target segment.
            target_segments: Optional list of ``(prefix, text)`` target segments.
                Each tuple is tokenized as one supervised target chunk in order.
                When ``None`` or empty, the auxiliary-task segment is omitted.
        """
        cleaned_task = prompt.strip().replace("_", " ").replace("\n", " ")

        if mode is None or not target_segments:
            # No auxiliary task: behave like the plain prompt (block attention, no token loss).
            if state is not None:
                state = np.clip(state, -1, 1)
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_task}; State: {state_str};\nAction: "
            else:
                full_prompt = f"Task: {cleaned_task};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            # Prompt shares the image block: no ar-block boundary inside the prompt.
            # (Segment-ids drive the attention mask when aux_lang_task is enabled, so this is cosmetic.)
            ar_mask = [False] * len(tokens)
            loss_mask = [False] * len(tokens)
            segment_ids = [history_pi0.SEG_PROMPT] * len(tokens)

        else:
            # Base prompt: "Task: ...; State: ...;\n" or "Task: ...;\n"
            if state is not None:
                state = np.clip(state, -1, 1)
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                sub_prompt_1 = f"Task: {cleaned_task}; State: {state_str};\n"
            else:
                sub_prompt_1 = f"Task: {cleaned_task};\n"
            sub_prompt_1_tokens = self._tokenizer.encode(sub_prompt_1, add_bos=True)

            target_all_tokens: list[int] = []
            target_all_loss_mask: list[bool] = []
            target_all_segment_ids: list[int] = []
            n_targets = len(target_segments)
            for i, (prefix, target_text) in enumerate(target_segments):
                target_text = target_text.strip().replace("_", " ").replace("\n", " ")
                separator = "" if i == 0 else ";\n"
                new_target_tokens = self._tokenizer.encode(
                    f"{separator}{prefix}{target_text}", add_eos=True if i == n_targets - 1 else False
                )
                target_all_tokens += new_target_tokens
                target_all_loss_mask += [True] * len(new_target_tokens)
                target_all_segment_ids += [history_pi0.SEG_AUX_TARGET] * len(new_target_tokens)

            action_tail_tokens = self._tokenizer.encode(";\nAction: ")

            # ar_mask isn't used when aux_lang_task is enabled (segment-ids drive a 5x5 allow
            # matrix in history_pi0 instead); use a flat False placeholder.
            ar_mask = [False] * (len(sub_prompt_1_tokens) + len(target_all_tokens) + len(action_tail_tokens))

            loss_mask = [False] * len(sub_prompt_1_tokens) + target_all_loss_mask + [False] * len(action_tail_tokens)

            segment_ids = (
                [history_pi0.SEG_PROMPT] * len(sub_prompt_1_tokens)
                + target_all_segment_ids
                + [history_pi0.SEG_PROMPT] * len(action_tail_tokens)
            )

            tokens = sub_prompt_1_tokens + target_all_tokens + action_tail_tokens

        tokens_len = len(tokens)

        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            padding_false = [False] * pad_len
            mask = [True] * tokens_len + padding_false
            tokens = tokens + padding_false
            ar_mask = ar_mask + padding_false
            loss_mask = loss_mask + padding_false
            # Segment-id for padding doesn't matter (masked out by input_mask), use SEG_PROMPT.
            segment_ids = segment_ids + [history_pi0.SEG_PROMPT] * pad_len
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
            segment_ids = segment_ids[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
            np.asarray(segment_ids, dtype=np.int32),
        )


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithAuxLangTask(DataTransformFn):
    tokenizer: PaligemmaTokenizer | PaligemmaTokenizerAuxLangTask
    discrete_state_input: bool = True
    aux_lang_task: str | None = None

    @staticmethod
    def _prepare_target_parts(fmt: dict[str, Any], fields: dict[str, Any]) -> list[tuple[str, str]]:
        """Build filtered (prefix, text) target segments."""
        return [
            (prefix, fields[field_name])
            for prefix, field_name in zip(fmt["target_prefixes"], fmt["target_fields"])
            if fields[field_name] is not None
        ]

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        state = data.get("state", None) if self.discrete_state_input else None
        if self.discrete_state_input and state is None:
            raise ValueError("State is required.")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        fields = {
            k: data.pop(k, None)
            for k in (
                "summary",
                "simple_subgoal",
                "grounded_subgoal",
                "video_description",
            )
        }

        extra: DataDict = {}

        if self.aux_lang_task is not None:
            fmt = _AUX_LANG_TASK_FORMAT.get(self.aux_lang_task)
            if fmt is None:
                raise ValueError(f"Unsupported aux_lang_task mode: {self.aux_lang_task!r}")

            target_segments = self._prepare_target_parts(fmt, fields)

            (
                tokenized_prompt,
                tokenized_prompt_mask,
                token_ar_mask,
                token_loss_mask,
                token_segment_ids,
            ) = self.tokenizer.tokenize(
                prompt=prompt,
                state=state,
                mode=self.aux_lang_task,
                target_segments=target_segments,
            )
            extra = {
                "token_ar_mask": token_ar_mask,
                "token_loss_mask": token_loss_mask,
                "token_segment_ids": token_segment_ids,
            }
        else:
            tokenized_prompt, tokenized_prompt_mask = self.tokenizer.tokenize(prompt, state)

        return {
            **data,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            **extra,
        }


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    train: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # This is MME-VLA set up
                aux_lang_task: str | None = None
                max_token_len = resolve_max_token_len(
                    model_config.history_config, base=model_config.max_token_len, train=self.train
                )

                if model_config.use_history and model_config.history_config is not None:
                    loaded_config = get_history_config(model_config.history_config)

                    aux_lang_task = getattr(loaded_config, "aux_lang_task", None)

                    if aux_lang_task == "subtask":
                        aux_lang_task = str(loaded_config.aux_lang_task)

                print(f"max_token_len: {max_token_len}")

                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        TokenizePromptWithAuxLangTask(
                            PaligemmaTokenizerAuxLangTask(max_token_len)
                            if aux_lang_task is not None
                            else PaligemmaTokenizer(max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            aux_lang_task=aux_lang_task,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *, train: bool = True
    ) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *, train: bool = True
    ) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfig(DataConfigFactory):
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *, train: bool = True
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb",  # (b, l, d1)
                        "static_pos_emb": "static_pos_emb",  # (b, l, d2)
                        "static_state_emb": "static_state_emb",  # (b, l, d3)
                        "static_mask": "static_mask",  # (b, l)
                        # subgoal fields
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                        "summary": "summary",
                        "video_description": "video_description",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(train=train)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMMEVLARealRobotDataConfig(DataConfigFactory):
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, *, train: bool = True
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/left_shoulder_image": "left_shoulder_image",
                        "observation/right_shoulder_image": "right_shoulder_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb",  # (b, l, d1)
                        "static_pos_emb": "static_pos_emb",  # (b, l, d2)
                        "static_state_emb": "static_state_emb",  # (b, l, d3)
                        "static_mask": "static_mask",  # (b, l)
                        # subgoal fields
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                        "summary": "summary",
                        "video_description": "video_description",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[RoboMMERealRobotInputs(model_type=model_config.model_type)],
            outputs=[RoboMMERealRobotOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(train=train)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


import openpi.shared.array_typing as at
from openpi.training.weight_loaders import WeightLoader
from openpi.training.weight_loaders import _merge_params
from openpi.training.weight_loaders import download


@dataclasses.dataclass(frozen=True)
class MMEVLAWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Loaded as np.ndarray; the training code handles conversion and sharding.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Merge in any missing weights (e.g. LoRA, our new modules).
        return _merge_params(loaded_params, params, missing_regex=".*")


####################### RoboMME #######################


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Model config. action_dim/action_horizon/max_token_len are shared (see BaseModelConfig);
    # specific implementations (e.g. Pi0Config) may add more attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # Optionally loads (possibly partial) weights from disk after model init.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # If > 0, zero out gradients for params not matching `mem_unfreeze_regex` for the first
    # `mem_only_warmup_steps` steps (Soft Memory Alignment stage 1, paper Sec. 4.3 / Talign).
    # opt_state stays sized for the full trainable set; non-mem AdamW moments decay toward
    # zero during warmup and recover after the boundary.
    mem_only_warmup_steps: int = 0
    mem_only_warmup_lr_multiplier: float = 0.0
    mem_unfreeze_regex: str = ".*mem.*"
    # After `mem_only_warmup_steps`, ramp the base-param LR multiplier 0->1 over this many
    # steps (0 = instant switch).
    warmup_base_steps: int = 0

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "runs/assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "runs/ckpts"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # More workers speed up data loading but increase memory/CPU usage.
    num_workers: int = 2
    # More prefetched batches speed up data loading but increase memory usage.
    prefetch_factor: int = 4
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 500
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If > 1, shards the model across this many devices via FSDP (data-parallel across the
    # remaining device groups), trading memory for speed.
    fsdp_devices: int = 1

    resum_ckpt_id: int | None = None

    dataset_path: str = "data/robomme"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")

_CONFIGS = [
    TrainConfig(
        name="pi05_baseline",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=False,
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=80_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000,
        save_interval=20_000,
        keep_period=20_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="mme_vla_suite",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=True,
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=80_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000,
        save_interval=20_000,
        keep_period=20_000,
        num_workers=4,
        prefetch_factor=4,
        ema_decay=0.999,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="scar",
        model=history_pi0.HistoryPi0Config(
            pi05=True,
            action_horizon=20,
            use_history=True,
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id="robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=80_000,
            decay_lr=1e-5,
        ),
        mem_only_warmup_steps=20_000,
        warmup_base_steps=500,
        mem_only_warmup_lr_multiplier=0.1,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000,
        save_interval=20_000,
        keep_period=20_000,
        num_workers=4,
        prefetch_factor=4,
        ema_decay=0.999,
        fsdp_devices=1,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
