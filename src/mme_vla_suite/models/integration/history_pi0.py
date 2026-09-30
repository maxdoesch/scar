import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from mme_vla_suite.models.config.history_config_utils import get_history_config
from mme_vla_suite.models.config.history_config_utils import resolve_max_token_len
from mme_vla_suite.models.integration import history_gemma as _gemma
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from openpi.models.model import Actions
from openpi.models.model import BaseModel
from openpi.models.pi0 import posemb_sincos
from openpi.models.pi0_config import Pi0Config
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("history-pi0")


SEG_MEMORY = 0
SEG_IMAGE = 1
SEG_PROMPT = 2
SEG_AUX_TARGET = 3
SEG_ACTION = 4

_SEGMENT_ATTN_PATTERN: np.ndarray = np.array(
    [
        [1, 0, 1, 0, 0],  # memory
        [0, 1, 1, 0, 0],  # image
        [1, 1, 1, 0, 0],  # prompt
        [1, 0, 1, 1, 0],  # target
        [1, 1, 1, 0, 1],  # action
    ],
    dtype=np.bool_,
)


def make_aux_lang_task_attn_mask(
    input_mask: at.Bool[at.Array, "b n"],
    segment_ids: at.Int[at.Array, "b n"],
    mem_causal_block_size: int | None = None,
) -> at.Bool[at.Array, "b n n"]:
    """Attention mask for the auxiliary-language-task pattern.

    Allows attention iff _SEGMENT_ATTN_PATTERN[seg_q, seg_k] AND both tokens are valid AND
    (for two SEG_AUX_TARGET tokens) key-idx <= query-idx for causality.

    Args:
        mem_causal_block_size: if not None, memory tokens (assumed to start at position 0)
            are block-causal in blocks of this size: tokens within a block are bidirectional,
            a block may attend to earlier blocks but not later ones.
    """
    # Convert lazily to avoid initializing JAX/CUDA at import time.
    attn_pattern = jnp.asarray(_SEGMENT_ATTN_PATTERN)

    allow = attn_pattern[segment_ids[:, :, None], segment_ids[:, None, :]]  # [B, N, N]

    # Within-target causal mask: only applies when both q and k are SEG_AUX_TARGET.
    n = segment_ids.shape[-1]
    idx = jnp.arange(n)
    causal = idx[None, :] <= idx[:, None]  # [N, N] lower-triangular, broadcast over batch.
    both_target = (segment_ids[:, :, None] == SEG_AUX_TARGET) & (segment_ids[:, None, :] == SEG_AUX_TARGET)
    answer_causal = jnp.where(both_target, causal[None, :, :], True)

    # Memory tokens are block-causal: query block must be >= key block, so older memory
    # frames can't attend to newer ones.
    if mem_causal_block_size is not None:
        mem_block_idx = idx // mem_causal_block_size  # [N]
        both_mem = (segment_ids[:, :, None] == SEG_MEMORY) & (segment_ids[:, None, :] == SEG_MEMORY)
        mem_causal = mem_block_idx[None, :] <= mem_block_idx[:, None]  # [N_q, N_k], key block <= query block
        answer_causal = jnp.where(both_mem, mem_causal[None, :, :], answer_causal)

    valid = input_mask[:, None, :] & input_mask[:, :, None]
    return allow & answer_causal & valid


# Offset placing aux-task token positions outside the range used by other tokens (bounded by
# max sequence length, well under this value).
_AUX_POSITION_OFFSET = 1 << 15


def compute_aux_lang_task_positions(
    input_mask: at.Bool[at.Array, "b n"],
    segment_ids: at.Int[at.Array, "b n"],
) -> at.Int[at.Array, "b n"]:
    """Positions that are invariant under auxiliary-task removal.

    Non-auxiliary-task tokens get positions `cumsum(input_mask AND not aux) - 1`, i.e. the same
    positions they would have had if auxiliary-task tokens were never in the sequence.
    Auxiliary-task tokens are assigned their own strictly monotonic positions within their
    segment, shifted by `_AUX_POSITION_OFFSET` so they don't clash with the non-auxiliary-task
    range.

    This preserves train/inference parity for every non-auxiliary-task token (memory, image,
    prompt base, actions), which is what the downstream action head conditions on.
    """
    is_aux_target = segment_ids == SEG_AUX_TARGET
    non_aux_valid = input_mask & ~is_aux_target
    non_aux_positions = jnp.cumsum(non_aux_valid.astype(jnp.int32), axis=1) - 1

    aux_valid = input_mask & is_aux_target
    aux_positions = jnp.cumsum(aux_valid.astype(jnp.int32), axis=1) - 1 + _AUX_POSITION_OFFSET

    return jnp.where(is_aux_target, aux_positions, non_aux_positions)


def make_attn_mask(input_mask, mask_ar, mask_na=None):
    """Adapted from pi0.py

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    mask_na: bool[B, N]
       [[0 0 0 1 1 1 0 0 0 ...]]: The 1s tokens can not attend to the first three 0s.
       this is used for vision tokens not attend to memory tokens, but action tokens can

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    if mask_na is not None:
        mask_na = jnp.broadcast_to(mask_na, input_mask.shape)
        mask_not_attend = jnp.logical_and(
            jnp.logical_or(mask_na[:, None, :], mask_na[:, :, None]),
            einops.repeat(jnp.cumsum(mask_na, axis=1) <= 0, "b k -> b s k", s=mask_na.shape[1]),
        )
        return jnp.where(mask_not_attend, False, jnp.logical_and(attn_mask, valid_mask))
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class HistoryPi0Config(Pi0Config):
    memory_expert_variant: _gemma.Variant = "gemma_150m"

    use_history: bool = False  # Use history or not
    history_config: str | None = None  # history config
    max_token_len: int = 64

    @override
    def create(self, rng: at.KeyArrayLike) -> "HistoryPi0":
        # Load the history config if it's specified
        if self.history_config is not None:
            loaded_config = get_history_config(self.history_config)
            # Create a new config with the loaded history config and the resolved token length.
            max_token_len = resolve_max_token_len(loaded_config, base=self.max_token_len)
            config_with_loaded_history = dataclasses.replace(
                self, history_config=loaded_config, max_token_len=max_token_len
            )

            logger.info("max_token_len: %s", config_with_loaded_history.max_token_len)

            return HistoryPi0(config_with_loaded_history, rngs=nnx.Rngs(rng))
        return HistoryPi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        base_obs_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            if not self.use_history:
                observation_spec = base_obs_spec  # basic pi0
            elif self.history_config.representation_type in ["perceptual", "selective"]:
                observation_spec = HistAugObservation.from_base_obs(
                    base_obs_spec,
                    static_image_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.dataset_token_budget,
                            self.history_config.memory_feature.img.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_mask=jax.ShapeDtypeStruct(
                        [batch_size, self.history_config.dataset_token_budget],
                        jnp.bool_,
                    ),
                    static_pos_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.dataset_token_budget,
                            self.history_config.memory_feature.pos.input_dim,
                        ],
                        jnp.float32,
                    ),
                    static_state_emb=jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.history_config.dataset_token_budget,
                            self.history_config.memory_feature.state.input_dim,
                        ],
                        jnp.float32,
                    ),
                )
            else:
                raise ValueError(f"Not supported representation type: {self.history_config.representation_type}")

            if self.use_history and getattr(self.history_config, "aux_lang_task", None):
                observation_spec = dataclasses.replace(
                    observation_spec,
                    token_segment_ids=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                )

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params and mem params
            filters.extend(
                [
                    nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                    nnx.Not(nnx_utils.PathRegex(".*mem.*")),
                ]
            )
            return nnx.Any(nnx.All(*filters), nnx_utils.PathRegex(".*img.*"))
        return nnx_utils.PathRegex(".*img.*")


class HistoryPi0(BaseModel):
    def __init__(self, config: HistoryPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.config = config
        self.use_history = config.use_history
        self.aux_lang_task = None

        if self.use_history:
            self.history_config = config.history_config
            self.integration_type = config.history_config.integration_type
            self.representation_type = config.history_config.representation_type
            self.aux_lang_task = getattr(self.history_config, "aux_lang_task", None)
            assert self.integration_type in ["context", "modulation", "expert"]
            assert self.representation_type in ["perceptual", "selective"]

            if self.representation_type == "perceptual":
                from mme_vla_suite.models.representation.gated_percep_mem import GatedPerceptualMemory
                from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
                from mme_vla_suite.models.representation.spatial_conv import SpatialConv
                from mme_vla_suite.models.representation.temporal_conv import TemporalConv
                from mme_vla_suite.models.representation.temporal_conv_identity import TemporalConvIdentity

                encoder_type = getattr(self.history_config.perceptual_memory, "encoder_type", None)
                assert encoder_type in [
                    None,
                    "gated_perceptual_memory",
                    "spatial_conv",
                    "temporal_conv",
                    "temporal_conv_identity",
                ], f"Unsupported encoder type: {encoder_type}"

                if encoder_type == "gated_perceptual_memory":
                    self.mem_encoder = GatedPerceptualMemory(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
                elif encoder_type == "spatial_conv":
                    self.mem_encoder = SpatialConv(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
                elif encoder_type == "temporal_conv":
                    self.mem_encoder = TemporalConv(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
                elif encoder_type == "temporal_conv_identity":
                    self.mem_encoder = TemporalConvIdentity(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
                else:
                    self.mem_encoder = PerceptualMemory(
                        config=self.history_config,
                        rngs=rngs,
                        dtype=config.dtype,
                    )
            elif self.representation_type == "selective":
                from mme_vla_suite.models.representation.select_mem import SelectiveMemory

                self.mem_encoder = SelectiveMemory(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
            else:
                raise ValueError(f"Not supported representation type: {self.representation_type}")

            logger.info(
                "====== Using History, Representation Type: %s , Integration Type: %s ======",
                self.representation_type,
                self.integration_type,
            )
            if self.representation_type == "perceptual":
                encoder_type = getattr(self.history_config.perceptual_memory, "encoder_type", None)
                if encoder_type == "gated_perceptual_memory":
                    logger.info("Gated Perceptual Memory using %s type", self.history_config.perceptual_memory.type)
                elif encoder_type == "spatial_conv":
                    logger.info("Spatial Conv Memory using %s type", self.history_config.perceptual_memory.type)
                elif encoder_type == "temporal_conv":
                    logger.info("Temporal Conv Memory using %s type", self.history_config.perceptual_memory.type)
                elif encoder_type == "temporal_conv_identity":
                    logger.info(
                        "Temporal Conv Identity (Ablation) Memory using %s type",
                        self.history_config.perceptual_memory.type,
                    )
                else:
                    logger.info("Perceptual Memory using %s type", self.history_config.perceptual_memory.type)
            elif self.representation_type == "selective":
                logger.info("Selective Memory using %s type", self.history_config.selective_memory.type)

            if self.integration_type == "expert":
                memory_expert_config = _gemma.get_config(config.memory_expert_variant)
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[
                            memory_expert_config,
                            paligemma_config,
                            action_expert_config,
                        ],
                        embed_dtype=config.dtype,
                        adarms=config.pi05,
                        integration_type=self.integration_type,
                        use_memory_gate=getattr(self.history_config, "use_memory_gate", False),
                        memory_gate_bias=getattr(self.history_config, "memory_gate_bias", 5.0),
                        memory_gate_type=getattr(self.history_config, "memory_gate_type", "scalar"),
                    )
                )
                llm.lazy_init(
                    rngs=rngs,
                    method="init",
                    use_adarms=([False, False, True] if config.pi05 else [False, False, False]),
                    mem_mods=[False, False, False],
                )
            else:
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[paligemma_config, action_expert_config],
                        embed_dtype=config.dtype,
                        adarms=config.pi05,
                        integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs,
                    method="init",
                    use_adarms=[False, True] if config.pi05 else [False, False],
                    mem_mods=[False, True] if self.integration_type == "modulation" else [False, False],
                )

        else:
            # safe setting
            self.history_config = self.integration_type = self.representation_type = None

            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    integration_type=self.integration_type,
                )
            )
            llm.lazy_init(
                rngs=rngs,
                method="init",
                use_adarms=[False, True] if config.pi05 else [False, False],
                mem_mods=[False, False],
            )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @property
    def _mem_causal_block_size(self) -> int | None:
        """Tokens per temporal block for block-causal memory attention; None if not applicable.

        Block-causal memory attention is controlled by the ``causal_memory_attention``
        flag in the history config (defaults to ``False``).
        When disabled, memory tokens attend fully bidirectionally and this returns None.
        """
        if not self.use_history or self.representation_type != "perceptual":
            return None
        hc = self.history_config
        if not getattr(hc, "causal_memory_attention", False):
            return None
        encoder_type = getattr(getattr(hc, "perceptual_memory", None), "encoder_type", None)
        if encoder_type in ["temporal_conv", "temporal_conv_identity"]:
            return hc.token_per_image
        num_frames = hc.dataset_token_budget // (hc.num_views * hc.token_per_image)
        return hc.model_token_budget // num_frames

    @at.typecheck
    def embed_memory(self, obs: HistAugObservation):
        if self.representation_type == "perceptual":
            tokens, input_mask, stats = self.mem_encoder(
                obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb, obs.static_mask
            )
        elif self.representation_type == "selective":
            tokens, input_mask, stats = self.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_mask)
        else:
            tokens = None
            input_mask = None
            stats = None
        if tokens is None:
            return None, None, None, None, None, None
        block_size = self._mem_causal_block_size
        if block_size is not None:
            # Block-causal ar_mask: the first token of each temporal block marks the start
            # of a new causal group. Tokens within the same block are fully bidirectional;
            # later blocks can attend to all earlier blocks but not vice-versa.
            ar_mask = [(i % block_size == 0) for i in range(tokens.shape[1])]
        else:
            ar_mask = [False] * tokens.shape[1]
        na_mask = [False] * tokens.shape[1]
        seg_ids = [SEG_MEMORY] * tokens.shape[1]
        return tokens, input_mask, ar_mask, na_mask, seg_ids, stats

    @at.typecheck
    def embed_prefix(
        self, obs: HistAugObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        at.Int[at.Array, "b s"] | None,
        Any | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []
        # Per-token segment ids, built as static ints and later broadcast to [B, N]. Only the
        # tokenized-prompt region varies per-example (aux-task context/target spans), spliced
        # in from obs.token_segment_ids below.
        static_seg_ids: list[int] = []
        prompt_seg_ids_dynamic = None  # [B, L] from obs when aux_lang_task is active.

        if self.integration_type == "context":
            (
                mem_tokens,
                mem_input_mask,
                mem_ar_mask,
                mem_na_mask,
                mem_seg_ids,
                stats,
            ) = self.embed_memory(obs)
            if mem_tokens is not None:
                tokens.append(mem_tokens)
                input_mask.append(mem_input_mask)
                ar_mask += mem_ar_mask
                na_mask += mem_na_mask
                static_seg_ids += mem_seg_ids
        else:
            stats = None

        # embed images
        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]
            static_seg_ids += [SEG_IMAGE] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Under aux_lang_task the attention is driven by segment-ids (see
            # make_aux_lang_task_attn_mask), so ar_mask is ignored for this slice and we
            # pass all-False to keep cumsum flat. Non-auxiliary-task path keeps the original
            # fully-bidirectional-with-images behavior (also all-False).
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]
            static_seg_ids += [SEG_PROMPT] * tokenized_inputs.shape[1]
            if self.aux_lang_task and obs.token_segment_ids is not None:
                prompt_seg_ids_dynamic = obs.token_segment_ids

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)

        batch_size = tokens.shape[0]
        seg_ids = jnp.broadcast_to(jnp.asarray(static_seg_ids, dtype=jnp.int32), (batch_size, tokens.shape[1]))
        if prompt_seg_ids_dynamic is not None:
            prompt_len = prompt_seg_ids_dynamic.shape[1]
            seg_ids = seg_ids.at[:, -prompt_len:].set(prompt_seg_ids_dynamic)

        return tokens, input_mask, ar_mask, na_mask, seg_ids, stats

    @at.typecheck
    def embed_suffix(
        self,
        obs: HistAugObservation,
        noisy_actions: Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        at.Int[at.Array, "b s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []
        static_seg_ids: list[int] = []

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
            na_mask += [False]
            # The optional pi0 state token lives in the "prompt/base-context" bucket for the
            # purposes of segment attention (it's part of the observation, not the action).
            static_seg_ids += [SEG_PROMPT]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        na_mask += [False] * self.action_horizon
        static_seg_ids += [SEG_ACTION] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)
        batch_size = tokens.shape[0]
        seg_ids = jnp.broadcast_to(jnp.asarray(static_seg_ids, dtype=jnp.int32), (batch_size, tokens.shape[1]))
        return tokens, input_mask, ar_mask, na_mask, seg_ids, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, prefix_seg_ids, stats = self.embed_prefix(
            observation
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, suffix_seg_ids, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, mem_seg_ids, stats = self.embed_memory(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            input_mask = jnp.concatenate([mem_input_mask, prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([mem_ar_mask, prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([mem_na_mask, prefix_na_mask, suffix_na_mask], axis=0)
            mem_seg_ids_arr = jnp.broadcast_to(
                jnp.asarray(mem_seg_ids, dtype=jnp.int32),
                (mem_tokens.shape[0], mem_tokens.shape[1]),
            )
            seg_ids = jnp.concatenate([mem_seg_ids_arr, prefix_seg_ids, suffix_seg_ids], axis=1)
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)
            seg_ids = jnp.concatenate([prefix_seg_ids, suffix_seg_ids], axis=1)

        if self.aux_lang_task:
            # Fully-bidirectional base context, causal within the target, actions isolated
            # from the auxiliary task. Drops na_mask entirely.
            attn_mask = make_aux_lang_task_attn_mask(input_mask, seg_ids, self._mem_causal_block_size)
            # Non-aux-task tokens keep the positions they'd have without the aux task.
            positions = compute_aux_lang_task_positions(input_mask, seg_ids)
        elif self.use_history:
            attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
        else:
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

        if self.integration_type == "expert":
            (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [mem_tokens, prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, None, adarms_cond],
            )
        elif self.integration_type == "modulation":
            mem_seq, mem_mask, _, _, _, stats = self.embed_memory(observation)
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
            )
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )

        total_loss = 0.0
        stats_out = dict(stats) if isinstance(stats, dict) else {}

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_mse = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [*b, ah]

        total_loss = action_mse

        if self.aux_lang_task:
            targets = jax.nn.one_hot(
                observation.tokenized_prompt[:, 1:],
                self.PaliGemma.llm.module.vocab_size,
            )
            # Pick the hiddens that predict txt[1..L-1]: h_txt[0..L-2] == prefix_out[:, -L:-1].
            logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] - 1 : -1], method="deembed")
            logp = jax.nn.log_softmax(logits, axis=-1)

            assert observation.token_loss_mask is not None, "Token loss mask is required"
            loss_mask = observation.token_loss_mask[:, 1:]
            token_pplx = jnp.sum(targets * logp, axis=-1)
            aux_lang_task_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(
                jnp.sum(loss_mask, -1), 1
            )

            total_loss += self.history_config.aux_lang_task_loss_weight * aux_lang_task_generation_loss[..., None]

            stats_out["action_mse_loss"] = jnp.mean(action_mse)
            stats_out["aux_lang_task_ce_loss"] = jnp.mean(aux_lang_task_generation_loss)

        return total_loss, stats_out

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
        observation = preprocess_observation(None, observation, train=False)
        # Convention: t=1 is noise, t=0 is the target distribution (opposite of the pi0 paper).
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # For "expert" integration, memory tokens sit in a separate expert and are prepended
        # here; for other integration_types they're already inside embed_prefix or unused.
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, prefix_seg_ids, _ = self.embed_prefix(observation)
        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, mem_seg_ids, _ = self.embed_memory(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            mem_seg_ids_arr = jnp.broadcast_to(
                jnp.asarray(mem_seg_ids, dtype=jnp.int32),
                (mem_tokens.shape[0], mem_tokens.shape[1]),
            )
            prefix_mask = jnp.concatenate([mem_input_mask, prefix_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([mem_ar_mask, prefix_ar_mask], axis=0)
            prefix_na_mask = jnp.concatenate([mem_na_mask, prefix_na_mask], axis=0)
            prefix_seg_ids = jnp.concatenate([mem_seg_ids_arr, prefix_seg_ids], axis=1)

        if self.aux_lang_task:
            prefix_attn_mask = make_aux_lang_task_attn_mask(prefix_mask, prefix_seg_ids, self._mem_causal_block_size)
            positions = compute_aux_lang_task_positions(prefix_mask, prefix_seg_ids)
        else:
            # Safe for all paths: with no memory tokens, cumsum never <=0 at a valid key, so
            # the na-constraint is inert.
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1

        if self.integration_type == "expert":
            _, kv_cache = self.PaliGemma.llm(
                [mem_tokens, prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
        else:
            if self.integration_type == "modulation":
                mem_seq, mem_mask, _, _, _, _ = self.embed_memory(observation)
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, _, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # suffix_attn_mask: (b, suffix_len, suffix_len), how suffix tokens attend to each other.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # prefix_attn_mask_step: (b, suffix_len, prefix_len), how suffix tokens attend to the prefix.
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # full_attn_mask: (b, suffix_len, prefix_len + suffix_len), suffix queries attending
            # to the full prefix+suffix keys/values.
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            # positions: (b, suffix_len), absolute positions of the suffix tokens.
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.integration_type == "expert":
                (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, None, adarms_cond],
                )
                assert mem_out is None
            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_aux_lang_task_tokens(
        self,
        observation: HistAugObservation,
        prompt_len: at.Int[at.Array, " b"] | int,
        *,
        max_new_tokens: int = 32,
        eos_token_id: int = 1,
    ) -> at.Int[at.Array, "b l"]:
        """Greedy-decode the aux-lang-task target tokens autoregressively. Debug helper — no KV cache.

        Args:
            observation: HistAugObservation whose `tokenized_prompt` already contains the
                prompt up to and including the target prefix (e.g. "Subtask: "). Positions
                at and after `prompt_len` are treated as writable slots for generated tokens.
                `token_segment_ids` must be provided (aux_lang_task must be enabled).
            prompt_len: per-example prompt length (scalar int or shape [B]). Tokens at
                positions < prompt_len are taken as given; the target is written into
                positions [prompt_len, prompt_len + max_new_tokens).
            max_new_tokens: max number of target tokens to generate.
            eos_token_id: stop condition (defaults to PaliGemma EOS = 1).

        Returns:
            Token ids of shape [B, max_new_tokens]. Slots past EOS (and slots for
            already-finished examples) are filled with 0. Decode with a SentencePiece
            processor on the caller side.
        """
        assert self.aux_lang_task, (
            "sample_aux_lang_task_tokens is only meaningful when aux_lang_task is enabled "
            "(e.g. aux_lang_task='subtask')"
        )
        assert observation.tokenized_prompt is not None, "tokenized_prompt required"
        assert observation.token_segment_ids is not None, "token_segment_ids required"

        observation = preprocess_observation(None, observation, train=False)

        prompt = observation.tokenized_prompt  # [B, L]
        seg_ids = observation.token_segment_ids  # [B, L]
        batch_size, max_len = prompt.shape

        prompt_len_arr = jnp.broadcast_to(jnp.asarray(prompt_len, jnp.int32), (batch_size,))
        idx_range = jnp.arange(max_len)[None, :]  # [1, L]

        # Canonicalize the observation: keep the user-supplied prompt before prompt_len as-is;
        # mark positions >= prompt_len as SEG_AUX_TARGET writable slots with tokens zeroed.
        # tokenized_prompt_mask flags only the populated prefix and advances one step at a
        # time as answer tokens are appended.
        before_prompt_end = idx_range < prompt_len_arr[:, None]
        prompt = jnp.where(before_prompt_end, prompt, 0)
        seg_ids = jnp.where(before_prompt_end, seg_ids, SEG_AUX_TARGET)

        obs_state = dataclasses.replace(
            observation,
            tokenized_prompt=prompt,
            tokenized_prompt_mask=jnp.zeros_like(before_prompt_end),  # overwritten per step
            token_segment_ids=seg_ids,
        )

        def _forward_prefix_out(obs_step):
            """Run the full prefix (+ memory, for expert) forward and return prefix_out."""
            p_tokens, p_mask, _, _, p_seg, _ = self.embed_prefix(obs_step)
            if self.integration_type == "expert":
                m_tokens, m_mask, _, _, m_seg, _ = self.embed_memory(obs_step)
                m_seg_arr = jnp.broadcast_to(
                    jnp.asarray(m_seg, dtype=jnp.int32), (m_tokens.shape[0], m_tokens.shape[1])
                )
                combined_mask = jnp.concatenate([m_mask, p_mask], axis=1)
                combined_seg = jnp.concatenate([m_seg_arr, p_seg], axis=1)
                attn = make_aux_lang_task_attn_mask(combined_mask, combined_seg, self._mem_causal_block_size)
                pos = compute_aux_lang_task_positions(combined_mask, combined_seg)
                (_, out, _), _ = self.PaliGemma.llm(
                    [m_tokens, p_tokens, None],
                    mask=attn,
                    positions=pos,
                    adarms_cond=[None, None, None],
                )
                return out
            attn = make_aux_lang_task_attn_mask(p_mask, p_seg, self._mem_causal_block_size)
            pos = compute_aux_lang_task_positions(p_mask, p_seg)
            if self.integration_type == "modulation":
                mem_seq, mem_mask, _, _, _, _ = self.embed_memory(obs_step)
                (out, _), _ = self.PaliGemma.llm(
                    [p_tokens, None],
                    mask=attn,
                    positions=pos,
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
            else:
                (out, _), _ = self.PaliGemma.llm([p_tokens, None], mask=attn, positions=pos)
            return out

        # JIT-friendly decoding loop. Carry = (k, done, tokenized_prompt, answer_ids); runs up
        # to `max_new_tokens` iterations and stops once every example has emitted EOS. Once an
        # example is done, its next_tok is forced to 0 and nothing is scattered into
        # tokenized_prompt, so the loop stays well-defined.
        def cond_fn(carry):
            k, done, _, _ = carry
            return jnp.logical_and(k < max_new_tokens, jnp.logical_not(jnp.all(done)))

        def body_fn(carry):
            k, done, tokenized_prompt, answer_ids = carry
            write_pos = prompt_len_arr + k  # [B]
            step_mask = idx_range < write_pos[:, None]
            obs_step = dataclasses.replace(
                obs_state,
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=step_mask,
            )

            prefix_out = _forward_prefix_out(obs_step)
            txt_out = prefix_out[:, -max_len:]
            gather = jnp.clip(write_pos - 1, 0, max_len - 1)[:, None, None]
            gather = jnp.broadcast_to(gather, (batch_size, 1, txt_out.shape[-1]))
            h_last = jnp.take_along_axis(txt_out, gather, axis=1)[:, 0, :]

            logits = self.PaliGemma.llm(h_last[:, None, :], method="deembed")[:, 0, :]
            next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            next_tok = jnp.where(done, jnp.zeros_like(next_tok), next_tok)

            answer_ids = answer_ids.at[:, k].set(next_tok)
            done = done | (next_tok == eos_token_id)

            one_hot_pos = idx_range == write_pos[:, None]
            tokenized_prompt = jnp.where(one_hot_pos, next_tok[:, None], tokenized_prompt)
            return k + jnp.int32(1), done, tokenized_prompt, answer_ids

        init_carry = (
            jnp.int32(0),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            obs_state.tokenized_prompt,
            jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32),
        )
        _, _, _, answer_ids = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return answer_ids

    def vision_encode(self, images: at.Float[at.Array, "k v 224 224 3"]):
        shape = images.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"

        x = einops.rearrange(images, "k v h w c -> (k v) h w c")
        out, _ = self.PaliGemma.img(x, train=False)
        out = einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])

        return out
