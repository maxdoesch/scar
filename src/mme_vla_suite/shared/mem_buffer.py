from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
import heapq
import math

import cv2
import einops
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.shared.data_utils import even_sampling_indices
from mme_vla_suite.shared.data_utils import even_sampling_indices_every_n
from mme_vla_suite.shared.data_utils import pool_tokens_to_size
from mme_vla_suite.shared.data_utils import right_padding_token_emb
from openpi.shared import image_tools


def create_dict(indices):
    result = defaultdict(lambda: defaultdict(list))  # {step_idx: {view_idx: [patch_idx]}}
    for step_idx, view_idx, patch_idx in indices:
        result[step_idx][view_idx].append(patch_idx)
    return result


class MemoryBuffer:
    # We use 8x8 tokens each image for token dropping and 4x4 tokens for frame sampling.
    def __init__(
        self,
        num_views: int = 1,  # camera views used as memory; prefer fixed views
        img_emb_dim: int = 2048,  # initial image token embedding dim (SigLIP: 2048)
        pos_emb_dim: int = 768,
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        compute_token_drop_score: bool = False,  # set true when using token dropping
        token_drop_keptsize: int = 2048,  # max tokens kept in the similarity-score heap
        token_drop_stride: int = 8,  # timestep stride for similarity scoring (too small -> negligible diff)
        prepare_buffer: bool = False,  # true for online eval, false for offline training
        pool_type: str = "mean",  # pooling type for image tokens: mean or max
        vision_enc_fn: Callable | None = None,  # vision encoding function
    ):
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim

        self.pool_type = pool_type
        self.token_drop_keptsize = token_drop_keptsize
        self.token_drop_stride = token_drop_stride
        self.token_drop_last_frame = -1
        self.scored_token_heap = []
        self.compute_token_drop_score = compute_token_drop_score

        self._history_feats = {}

        if prepare_buffer:
            # load models in GPUs to provide embeedings
            # used for dataset buliding and evaluation
            from mme_vla_suite.shared.posemb_3d import PosEmb3D

            if vision_enc_fn is None:
                from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer

                siglip_tokenizer = SigLipTokenizer()
                self.vision_enc = jax.jit(siglip_tokenizer.__call__)
            else:
                self.vision_enc = vision_enc_fn
            pos_embedder = PosEmb3D(dim=pos_emb_dim)
            ranges = jnp.arange(max_steps)
            self.pos_emb_dict = {
                "8x8": np.array(pos_embedder(ranges, 8)),
                "4x4": np.array(pos_embedder(ranges, 4)),
                "2x2": np.array(pos_embedder(ranges, 2)),
            }
        else:
            # used in dataset loading mode
            self.vision_enc = None
            self.pos_emb_dict = None

    def add_buffer(
        self,
        images,  #: (t v h w 3), np.int8
        states,  #: (t d), np.float32
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        assert v == self.num_views

        image_jnp = jnp.array(images.astype(np.float32) / 255.0 * 2.0 - 1.0)
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)

        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)  # (t, v, 64, 2048)
        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)  # (t, v, 16, 2048)
        pooled_emb_2x2 = pool_tokens_to_size(output_emb, 4)  # (t, v, 4, 2048)

        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]  # (v, 64, 2048)
            image_emb_4x4 = jax.device_get(pooled_emb_4x4)[i]  # (v, 16, 2048)
            image_emb_2x2 = jax.device_get(pooled_emb_2x2)[i]  # (v, 4, 2048)

            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]  # (v, 64, 768)
            pos_emb_4x4 = self.pos_emb_dict["4x4"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]  # (v, 16, 768)
            pos_emb_2x2 = self.pos_emb_dict["2x2"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]  # (v, 4, 768)

            token_emb_to_save = {
                "image_pixels": images[i].copy(),  # fp32, this is for visualization
                "image_emb_8x8": image_emb_8x8,  # bf16
                "image_emb_4x4": image_emb_4x4,  # bf16
                "image_emb_2x2": image_emb_2x2,  # bf16
                "pos_emb_8x8": pos_emb_8x8,  # fp32
                "pos_emb_4x4": pos_emb_4x4,  # fp32
                "pos_emb_2x2": pos_emb_2x2,  # fp32
                "state_emb": states[i],  # fp32
            }
            assert step_idx not in self._history_feats, f"step_idx {step_idx} already in buffer"
            self._history_feats[step_idx] = token_emb_to_save

            if self.compute_token_drop_score:
                self._process_token_drop_score(step_idx)

    def get_history_feats(self, step_idx: int, remove_image_pixels: bool = True):
        if remove_image_pixels:
            return {k: v for k, v in self._history_feats[step_idx].items() if k != "image_pixels"}
        return self._history_feats[step_idx]

    def _process_token_drop_score(self, step_idx):
        if step_idx == 0:  # keep all first-frame tokens; only later frames are dropped
            for patch_idx in range(64):  # 8x8 grid, fixed for now
                for view_idx in range(self.num_views):
                    heapq.heappush(self.scored_token_heap, (1000.0, step_idx, view_idx, patch_idx))

        if step_idx == self.token_drop_last_frame + self.token_drop_stride:
            prev_img = self._history_feats[max(0, self.token_drop_last_frame)]["image_pixels"]
            curr_img = self._history_feats[step_idx]["image_pixels"]

            prev_img = prev_img.astype(np.float32) / 255.0 * 2.0 - 1.0
            curr_img = curr_img.astype(np.float32) / 255.0 * 2.0 - 1.0

            prev_img_tokens = einops.rearrange(prev_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            curr_img_tokens = einops.rearrange(curr_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)

            for view_idx in range(self.num_views):
                # pixel works better than siglip embedding
                difference = np.abs(prev_img_tokens[view_idx] - curr_img_tokens[view_idx]).mean(axis=-1)
                for patch_idx in range(64):
                    if difference[patch_idx] < 1e-4:
                        # too similar, skip this patch
                        continue
                    heapq.heappush(
                        self.scored_token_heap,
                        (
                            difference[patch_idx],
                            step_idx,  # timestep index
                            view_idx,  # view index
                            patch_idx,  # patch index
                        ),
                    )
                    if len(self.scored_token_heap) > self.token_drop_keptsize:
                        heapq.heappop(self.scored_token_heap)

            self.token_drop_last_frame += self.token_drop_stride

    def clear(self):
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self._history_feats.clear()

    def get_token_dropping_indices(self):
        selected_tokens_heap = deepcopy(self.scored_token_heap)
        kept_indices = []
        while selected_tokens_heap:
            _, buffer_idx, view_idx, patch_idx = heapq.heappop(selected_tokens_heap)
            kept_indices.append((buffer_idx, view_idx, patch_idx))

        return kept_indices

    @staticmethod
    def filter_token_dropping_indices(kept_indices, step_idx, token_budget, is_sorted=True):
        kept_indices_filtered = [item for item in kept_indices if item[0] <= step_idx]
        kept_indices = kept_indices_filtered[-token_budget:]
        if is_sorted:
            kept_indices = sorted(kept_indices)
        return kept_indices

    def _prepare_token_dropping(self, history_feats, sorted_kept_indices, token_budget):
        img_emb = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        pos_emb = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state_emb = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        mask = np.zeros((token_budget), dtype=np.bool_)

        for idx, (buffer_idx, view_idx, patch_idx) in enumerate(sorted_kept_indices[:token_budget]):
            img_emb[idx] = history_feats[buffer_idx]["image_emb_8x8"][view_idx][patch_idx]
            pos_emb[idx] = history_feats[buffer_idx]["pos_emb_8x8"][view_idx][patch_idx]
            state_emb[idx] = history_feats[buffer_idx]["state_emb"]
            mask[idx] = True

        return img_emb, pos_emb, state_emb, mask

    def prepare_token_dropping(
        self, step_idx, token_budget, history_feats_gather_fn, kept_indices=None, *args, **kwargs
    ):
        if kept_indices is None:
            indices_to_load = self.get_token_dropping_indices()
        else:
            indices_to_load = kept_indices

        sorted_kept_indices = self.filter_token_dropping_indices(
            indices_to_load, step_idx, token_budget, is_sorted=True
        )

        indices_to_load = sorted(set([item[0] for item in sorted_kept_indices]))

        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)

        return self._prepare_token_dropping(history_feats, sorted_kept_indices, token_budget)

    def _visualize_token_dropping(self, kept_indices, step_idx):
        dic = create_dict(sorted(kept_indices))
        images = [self._history_feats[idx]["image_pixels"][0] for idx in range(len(self._history_feats))]

        images_anno = []
        for step_idx in dic:
            img_anno = images[step_idx].copy()

            img_anno = cv2.putText(
                img_anno,
                f"{step_idx}",
                (img_anno.shape[1] // 2, img_anno.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            mask = np.zeros(img_anno.shape[:2], dtype=np.uint8)

            for patch_idx in range(64):
                if patch_idx in dic[step_idx][0]:
                    continue
                h_center = patch_idx // 8 * 32 + 16
                w_center = patch_idx % 8 * 32 + 16
                mask = cv2.rectangle(
                    mask,
                    (w_center - 16, h_center - 16),
                    (w_center + 16, h_center + 16),
                    255,
                    -1,
                )
            img_anno[mask == 255] = img_anno[mask == 255] * 0.3 + np.array([255, 255, 255]) * 0.7

            images_anno.append(img_anno)

        import os

        import imageio

        imageio.mimsave(
            os.path.join(
                "debug",
                f"token_drop_anno_step_{step_idx}.mp4",
            ),
            images_anno,
            fps=10,
        )

    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        """RoboMME's evenly-spaced sampling scheme (paper Sec. 4.1)."""
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)

    def get_frame_sampling_indices_every_n(self, step_idx, token_budget, step_size, token_per_image):
        """SCAR's strided sampling strategy with stride `step_size` (paper Eq. 6, ∆TF)."""
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices_every_n(step_idx, step_size, max_size)

    def _prepare_frame_sampling(self, history_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)

        sampled_img_emb = self._load_emb(history_feats, indices_to_load, f"image_emb_{spatial_key}").astype(np.float32)
        sampled_pos_emb = self._load_emb(history_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(history_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)

        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )

        img_emb = np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)

        return img_emb, pos_emb, state_emb, mask

    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)

        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)

    def prepare_frame_sampling_every_n(
        self,
        step_idx,
        token_budget,
        step_size,
        token_per_image,
        history_feats_gather_fn,
        *args,
        **kwargs,
    ):
        indices_to_load = self.get_frame_sampling_indices_every_n(step_idx, token_budget, step_size, token_per_image)

        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)

    def _visualize_frame_sampling(self, indices_to_load, step_idx):
        images = [self._history_feats[idx]["image_pixels"][0] for idx in indices_to_load]

        import os

        import imageio

        imageio.mimsave(
            os.path.join(
                "debug",
                f"frame_sampling_step_{step_idx}.mp4",
            ),
            images,
            fps=2,
        )

    @staticmethod
    def _load_emb(history_feats: dict, indices_to_load: list[int], key: str):
        return np.stack(
            [history_feats[idx][key] for idx in indices_to_load],
            axis=0,
        )

    def default_history_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._history_feats[idx] for idx in indices_to_load}
