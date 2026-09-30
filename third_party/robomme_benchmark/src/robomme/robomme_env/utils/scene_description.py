from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robomme.robomme_env.utils.streak_counter import _ordinal_word

from robomme.robomme_env.utils.choice_action_mapping import (
    extract_actor_position_xyz,
    project_world_to_pixel,
)
from robomme.robomme_env.utils.task_goal import num2words_2

# Fixed base_camera (eye=(0.3,0,0.4), target=(0,0,-0.2), fov=pi/2, 256x256),
# derived analytically so it is available before a live sensor_param exists.
_SQRT5 = 5.0**0.5
_BASE_CAMERA_INTRINSIC_CV = (
    # fx = fy = 128 / tan(pi/4) = 128
    (128.0, 0.0, 128.0),
    (0.0, 128.0, 128.0),
    (0.0, 0.0, 1.0),
)
_BASE_CAMERA_EXTRINSIC_CV = (
    # World-to-camera (OpenCV) from sapien_utils.look_at(eye=(0.3,0,0.4), target=(0,0,-0.2)).
    (0.0, 1.0, 0.0, 0.0),
    (2.0 / _SQRT5, 0.0, -1.0 / _SQRT5, -0.2 / _SQRT5),
    (-1.0 / _SQRT5, 0.0, -2.0 / _SQRT5, 1.1 / _SQRT5),
)
_BASE_CAMERA_IMAGE_SHAPE = (256, 256)  # (height, width)


def _actor_xy(actor) -> tuple[float, float]:
    pose = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
    if hasattr(pose, "detach"):
        # GPU-backed envs return a CUDA tensor; np.asarray() on it would raise.
        pose = pose.detach().cpu().numpy()
    xy = np.asarray(pose, dtype=np.float32).reshape(-1)
    return float(xy[0]), float(xy[1])


def _pixel_yx(actor) -> tuple[int, int]:
    """Project an actor's world position onto the base_camera image plane."""
    world_xyz = extract_actor_position_xyz(actor)
    pixel = project_world_to_pixel(
        world_xyz,
        _BASE_CAMERA_INTRINSIC_CV,
        _BASE_CAMERA_EXTRINSIC_CV,
        _BASE_CAMERA_IMAGE_SHAPE,
    )
    if pixel is None:
        raise ValueError(f"actor {actor!r} does not project into base_camera frame")
    px, py = pixel
    return py, px


def _format_pixel_yx(actor) -> str:
    """Pixel-coordinate string in the "<y, x>" format used elsewhere, e.g. "<188, 151>"."""
    y, x = _pixel_yx(actor)
    return f"<{y}, {x}>"


def _coords_suffix(actor) -> str:
    """Pixel-coordinate annotation appended after a noun phrase, e.g. " <188, 151>"."""
    return f" {_format_pixel_yx(actor)}"


def _tertile_bucket_labels(
    ordered_actors: Sequence[object], low_word: str, mid_word: str, high_word: str
) -> dict[int, str]:
    """Split a sorted actor list into 3 near-equal-size index buckets."""
    n = len(ordered_actors)
    labels: dict[int, str] = {}
    for index, actor in enumerate(ordered_actors):
        bucket = min(index * 3 // n, 2)
        labels[id(actor)] = (low_word, mid_word, high_word)[bucket]
    return labels


def _left_to_right_position_labels(actors: Sequence[object]) -> dict[int, str]:
    """Bare left-to-right positional words (no noun), e.g. "leftmost", "middle"."""
    ordered = sorted(
        actors,
        key=lambda actor: (_actor_xy(actor)[1], _actor_xy(actor)[0]),
    )
    count = len(ordered)
    labels: dict[int, str] = {}

    if count == 1:
        labels[id(ordered[0])] = "only"
        return labels
    if count == 2:
        labels[id(ordered[0])] = "left"
        labels[id(ordered[1])] = "right"
        return labels
    if count == 3:
        labels[id(ordered[0])] = "leftmost"
        labels[id(ordered[1])] = "middle"
        labels[id(ordered[2])] = "rightmost"
        return labels

    return _tertile_bucket_labels(ordered, "left", "middle", "right")


def _top_to_bottom_position_labels(actors: Sequence[object]) -> dict[int, str]:
    """Bare near/far positional words (no noun); table x increases toward the camera."""
    count = len(actors)
    labels: dict[int, str] = {}

    if count == 1:
        labels[id(actors[0])] = "only"
        return labels

    xs = [_actor_xy(actor)[0] for actor in actors]
    if max(xs) - min(xs) < 1e-9:
        # All actors are at the same table depth; no real top/bottom split to report.
        for actor in actors:
            labels[id(actor)] = "middle"
        return labels

    ordered = sorted(
        actors,
        key=lambda actor: (_actor_xy(actor)[0], _actor_xy(actor)[1]),
    )

    if count == 2:
        labels[id(ordered[0])] = "top"
        labels[id(ordered[1])] = "bottom"
        return labels
    if count == 3:
        labels[id(ordered[0])] = "top"
        labels[id(ordered[1])] = "middle"
        labels[id(ordered[2])] = "bottom"
        return labels

    return _tertile_bucket_labels(ordered, "top", "middle", "bottom")


_SIDE_WORDS = {
    "left": "left",
    "leftmost": "left",
    "right": "right",
    "rightmost": "right",
}


def _combine_axis_words(lr_word: str, tb_word: str) -> str:
    """Merge left-right and near/far words into one descriptor, e.g. "top-left corner"; drops "middle"/"only" axes."""
    if lr_word == "only" or tb_word == "only":
        return "only"

    side = _SIDE_WORDS.get(lr_word)
    depth = tb_word if tb_word in ("top", "bottom") else None

    if side and depth:
        return f"{depth}-{side} corner"
    if side:
        return lr_word
    if depth:
        return depth
    return "middle"


def _disambiguate_mentioned_ties(
    actors: Sequence[object],
    combined_labels: dict[int, str],
    mentioned: Sequence[object] | None,
) -> dict[int, str]:
    """Append a lightweight tie-break only for mentioned actors that collide."""
    if mentioned is None:
        return dict(combined_labels)

    mentioned_ids = {id(actor) for actor in mentioned}
    actors_by_id = {id(actor): actor for actor in actors}
    groups: dict[str, list[int]] = {}
    for actor_id in mentioned_ids:
        if actor_id not in combined_labels:
            continue
        groups.setdefault(combined_labels[actor_id], []).append(actor_id)

    result = dict(combined_labels)
    for label, tied_ids in groups.items():
        if len(tied_ids) < 2:
            continue
        tied_ids_ordered = sorted(
            tied_ids, key=lambda actor_id: _actor_xy(actors_by_id[actor_id])[0]
        )
        if len(tied_ids_ordered) == 2:
            result[tied_ids_ordered[0]] = f"farther {label}"
            result[tied_ids_ordered[1]] = f"nearer {label}"
            continue
        for rank, actor_id in enumerate(tied_ids_ordered, start=1):
            ordinal = num2words_2.get(rank, f"{rank}th")
            result[actor_id] = f"{ordinal} {label}"
    return result


def _two_axis_position_labels(
    actors: Sequence[object], mentioned: Sequence[object] | None = None
) -> dict[int, str]:
    """Bare two-axis position words (no noun), e.g. "leftmost", "top-right corner".

    `actors` defines the full reference set used to rank position buckets.
    `mentioned`, if given, restricts tie-break disambiguation to the actors that
    will actually be named in the rendered sentence.
    """
    lr_labels = _left_to_right_position_labels(actors)
    tb_labels = _top_to_bottom_position_labels(actors)
    combined = {
        actor_id: _combine_axis_words(lr_labels[actor_id], tb_labels[actor_id])
        for actor_id in lr_labels
    }
    return _disambiguate_mentioned_ties(actors, combined, mentioned)


def _two_axis_labels(
    actors: Sequence[object], noun: str, mentioned: Sequence[object] | None = None
) -> dict[int, str]:
    position_labels = _two_axis_position_labels(actors, mentioned=mentioned)
    return {
        actor_id: f"{position} {noun}" for actor_id, position in position_labels.items()
    }


def _oxford_join_clauses(clauses: Sequence[str]) -> str:
    """Oxford-comma join: "A", "A and B", "A, B, and C"."""
    clauses = list(clauses)
    if len(clauses) <= 1:
        return "".join(clauses)
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _render_covering_clauses_sentence(
    pairs: Sequence[tuple[object, str]],
    labels: dict[int, str],
    *,
    verb: str,
    container_of=lambda container: container,
) -> dict[str, str]:
    """Shared clause/sentence builder for covering-summary sentences of the form
    "The {color} cube {verb} the {label}, the {color} cube by the {label}, ...".
    `container_of` resolves the container whose label/coordinates apply to a given
    pair's container (identity by default; lets callers swap in a post-swap owner)."""

    def clause(container, color, is_first, grounded) -> str:
        owner = container_of(container)
        label = labels[id(owner)]
        suffix = _coords_suffix(owner) if grounded else ""
        if is_first:
            return f"The {color} cube {verb} the {label}{suffix}"
        return f"the {color} cube by the {label}{suffix}"

    def sentence(grounded) -> str:
        clauses = [
            clause(container, color, i == 0, grounded)
            for i, (container, color) in enumerate(pairs)
        ]
        return f"{_oxford_join_clauses(clauses)}."

    return {"simple": sentence(False), "grounded": sentence(True)}


def render_covering_events_sentence(
    bin_color_pairs: Sequence[tuple[object, str]],
    all_containers: Sequence[object] | None = None,
) -> dict[str, str]:
    """Build a combined covering-event sentence for all cube/container pairings, e.g.
    "The red cube was covered by the leftmost container, the green cube by the middle
    container, and the blue cube by the rightmost container." Pass `all_containers`
    (bins + decoys) so position labels are ranked across the full scene."""
    bins = [bin_actor for bin_actor, _ in bin_color_pairs]
    position_reference = list(all_containers) if all_containers is not None else bins
    labels = _two_axis_labels(position_reference, "container", mentioned=bins)
    return _render_covering_clauses_sentence(
        bin_color_pairs, labels, verb="was covered by"
    )


def render_current_covering_sentence(
    cube_container_pairs: Sequence[tuple[object, str]],
    all_containers: Sequence[object] | None = None,
) -> dict[str, str]:
    """Present-tense description of where each cube is *currently* hidden, e.g.
    "The blue cube is covered by the leftmost container, the green cube by the
    middle container, and the red cube by the rightmost container." Pairs are
    ordered left-to-right by the container's current position. Pass
    `all_containers` (cube containers + decoys) so position labels are ranked
    across the full scene. The descriptive `hybrid` variant mirrors `grounded`
    (a clean sentence that keeps the raw pixel coordinates)."""
    containers = [container for container, _ in cube_container_pairs]
    position_reference = (
        list(all_containers) if all_containers is not None else containers
    )
    labels = _two_axis_labels(position_reference, "container", mentioned=containers)
    ordered = sorted(
        cube_container_pairs,
        key=lambda pair: (_actor_xy(pair[0])[1], _actor_xy(pair[0])[0]),
    )
    rendered = _render_covering_clauses_sentence(ordered, labels, verb="is covered by")
    grounded = rendered["grounded"]
    return {"simple": rendered["simple"], "grounded": grounded, "hybrid": grounded}


def _simulate_nearest_neighbor_swaps(
    actors: Sequence[object],
    swap_anchors: Sequence[object],
) -> list[tuple[object, object]]:
    """Compute swap events by nearest-neighbor matching on a local position
    snapshot, without mutating live actor/env state."""
    current_xy = {
        id(actor): np.asarray(_actor_xy(actor), dtype=np.float32) for actor in actors
    }
    events: list[tuple[object, object]] = []

    for anchor in swap_anchors:
        if anchor is None:
            continue
        candidates = [actor for actor in actors if actor is not anchor]
        if not candidates:
            continue

        partner = min(
            candidates,
            key=lambda actor: np.linalg.norm(
                current_xy[id(anchor)] - current_xy[id(actor)]
            ),
        )
        events.append((anchor, partner))
        current_xy[id(anchor)], current_xy[id(partner)] = (
            current_xy[id(partner)],
            current_xy[id(anchor)],
        )

    return events


def _simulate_nearest_neighbor_swaps_with_labels(
    actors: Sequence[object],
    swap_anchors: Sequence[object],
    position_labels: dict[int, str],
) -> list[tuple[object, object, str, str]]:
    """Like `_simulate_nearest_neighbor_swaps`, but also tracks each participant's
    position label before the swap so multi-step chains reference current positions."""
    current_xy = {
        id(actor): np.asarray(_actor_xy(actor), dtype=np.float32) for actor in actors
    }
    current_label = dict(position_labels)
    events: list[tuple[object, object, str, str]] = []

    for anchor in swap_anchors:
        if anchor is None:
            continue
        candidates = [actor for actor in actors if actor is not anchor]
        if not candidates:
            continue

        partner = min(
            candidates,
            key=lambda actor: np.linalg.norm(
                current_xy[id(anchor)] - current_xy[id(actor)]
            ),
        )
        events.append(
            (anchor, partner, current_label[id(anchor)], current_label[id(partner)])
        )
        current_xy[id(anchor)], current_xy[id(partner)] = (
            current_xy[id(partner)],
            current_xy[id(anchor)],
        )
        current_label[id(anchor)], current_label[id(partner)] = (
            current_label[id(partner)],
            current_label[id(anchor)],
        )

    return events


def _mark_swap_backs(events: Sequence[tuple]) -> list[tuple]:
    """Append `is_swap_back` and `is_immediate_swap_back` flags to each event tuple.
    A swap-back repeats an earlier pair; an immediate swap-back directly follows its
    matching swap with no other pair in between (enables pronoun shorthand)."""
    seen_pairs: set[frozenset[int]] = set()
    marked: list[tuple] = []
    prev_pair_key: frozenset[int] | None = None
    for event in events:
        anchor, partner, *extra = event
        pair_key = frozenset((id(anchor), id(partner)))
        is_swap_back = pair_key in seen_pairs
        is_immediate_swap_back = is_swap_back and pair_key == prev_pair_key
        marked.append((anchor, partner, *extra, is_swap_back, is_immediate_swap_back))
        if is_swap_back:
            seen_pairs.discard(pair_key)
        else:
            seen_pairs.add(pair_key)
        prev_pair_key = pair_key
    return marked


def _swap_participant_phrase(label: str, color: str | None, capitalize: bool) -> str:
    """Render one swap participant with its pre-swap position label, e.g.
    "The red cube's container (bottom-left corner)" or "the empty container (middle)"."""
    lead = "The" if capitalize else "the"
    if color is None:
        return f"{lead} empty container ({label})"
    return f"{lead} {color} cube's container ({label})"


def _swap_participant_bare_phrase(color: str | None, capitalize: bool) -> str:
    """Like `_swap_participant_phrase` but without the position label, for swap-back sentences."""
    lead = "The" if capitalize else "the"
    if color is None:
        return f"{lead} empty container"
    return f"{lead} {color} cube's container"


def render_swap_event_sentence(
    label_a: str,
    color_a: str | None,
    label_b: str | None,
    color_b: str | None,
    is_swap_back: bool,
    is_immediate_swap_back: bool = False,
) -> str:
    """Build one swap-narration sentence. Labels are pre-swap positions; colors are
    None for empty decoy containers. Immediate swap-backs collapse to a pronoun form."""
    if is_swap_back:
        if is_immediate_swap_back:
            return "They then swapped back."
        return (
            f"{_swap_participant_bare_phrase(color_a, capitalize=True)} and "
            f"{_swap_participant_bare_phrase(color_b, capitalize=False)} then swapped back."
        )
    return (
        f"{_swap_participant_phrase(label_a, color_a, capitalize=True)} "
        "swapped with "
        f"{_swap_participant_phrase(label_b, color_b, capitalize=False)}."
    )


def _build_unmask_sentences(
    bin_color_pairs: Sequence[tuple[object, str]],
    swap_anchors: Sequence[object],
    grounded: bool,
    all_containers: Sequence[object] | None = None,
) -> list[str]:
    bins = [bin_actor for bin_actor, _ in bin_color_pairs]
    ordered_pairs = sorted(
        bin_color_pairs,
        key=lambda pair: (_actor_xy(pair[0])[1], _actor_xy(pair[0])[0]),
    )
    # Rank positions across all containers (including decoys at higher difficulties).
    position_reference = list(all_containers) if all_containers is not None else bins
    position_labels = _two_axis_position_labels(position_reference, mentioned=bins)
    color_by_actor = {id(bin_actor): color for bin_actor, color in bin_color_pairs}

    def suffix(actor) -> str:
        return _coords_suffix(actor) if grounded else ""

    sentences = [
        f"The {position_labels[id(bin_actor)]} container{suffix(bin_actor)} "
        f"hides the {color} cube."
        for bin_actor, color in ordered_pairs
    ]

    events = _simulate_nearest_neighbor_swaps(position_reference, swap_anchors)
    for anchor, partner, is_swap_back, is_immediate_swap_back in _mark_swap_backs(
        events
    ):
        label_a = position_labels[id(anchor)] + (
            f" {_format_pixel_yx(anchor)}" if grounded else ""
        )
        label_b = position_labels[id(partner)] + (
            f" {_format_pixel_yx(partner)}" if grounded else ""
        )
        sentences.append(
            render_swap_event_sentence(
                label_a=label_a,
                color_a=color_by_actor.get(id(anchor)),
                label_b=label_b,
                color_b=color_by_actor.get(id(partner)),
                is_swap_back=is_swap_back,
                is_immediate_swap_back=is_immediate_swap_back,
            )
        )

    return sentences


def build_unmask_video_description(
    bin_color_pairs: Sequence[tuple[object, str]],
    swap_anchors: Sequence[object] = (),
    all_containers: Sequence[object] | None = None,
) -> dict[str, str]:
    """
    `all_containers` should list every container spawned in the scene
    (cube-hiding bins plus any empty decoys). When omitted, position words
    are ranked within `bin_color_pairs` only. The descriptive `hybrid` variant
    mirrors `grounded` (keeps the raw pixel coordinates).
    """
    grounded = " ".join(
        _build_unmask_sentences(
            bin_color_pairs,
            swap_anchors,
            grounded=True,
            all_containers=all_containers,
        )
    )
    return {
        "simple": " ".join(
            _build_unmask_sentences(
                bin_color_pairs,
                swap_anchors,
                grounded=False,
                all_containers=all_containers,
            )
        ),
        "grounded": grounded,
        "hybrid": grounded,
    }


def build_unmask_final_locations_description(
    bin_color_pairs: Sequence[tuple[object, str]],
    swap_anchors: Sequence[object] = (),
    all_containers: Sequence[object] | None = None,
) -> dict[str, str]:
    """Present-tense description of where each cube is hidden at the *end* of the
    video, after all container swaps have played out, e.g. "The green cube is
    covered by the leftmost container, the blue cube by the middle container, and
    the red cube by the rightmost container."

    Swaps only permute which container occupies which original position slot, so we
    replay the nearest-neighbour swap sequence as a slot permutation: after the
    replay, `final_slot[id(container)]` is the container whose original position
    (label and pixel coordinates) the cube's container ends up occupying. Sentence
    structure matches `render_current_covering_sentence`; the descriptive `hybrid`
    variant mirrors `grounded` (keeps the raw pixel coordinates)."""
    bins = [bin_actor for bin_actor, _ in bin_color_pairs]
    position_reference = list(all_containers) if all_containers is not None else bins

    final_slot = {id(actor): actor for actor in position_reference}
    for anchor, partner in _simulate_nearest_neighbor_swaps(
        position_reference, swap_anchors
    ):
        final_slot[id(anchor)], final_slot[id(partner)] = (
            final_slot[id(partner)],
            final_slot[id(anchor)],
        )

    def slot(container):
        return final_slot.get(id(container), container)

    final_slots = [slot(bin_actor) for bin_actor in bins]
    labels = _two_axis_labels(position_reference, "container", mentioned=final_slots)
    ordered = sorted(
        bin_color_pairs,
        key=lambda pair: (
            _actor_xy(slot(pair[0]))[1],
            _actor_xy(slot(pair[0]))[0],
        ),
    )

    rendered = _render_covering_clauses_sentence(
        ordered, labels, verb="is covered by", container_of=slot
    )

    grounded = rendered["grounded"]
    return {"simple": rendered["simple"], "grounded": grounded, "hybrid": grounded}


def _repick_cube_phrase(
    position_label: str, color: str | None, capitalize: bool
) -> str:
    """Render one repick participant, e.g. "the rightmost blue cube" or,
    when no color mapping is available, "the rightmost cube"."""
    lead = "The" if capitalize else "the"
    if color is None:
        return f"{lead} {position_label} cube"
    return f"{lead} {position_label} {color} cube"


def _build_repick_sentences(
    *,
    target_cube,
    all_cubes: Sequence[object],
    swap_anchors: Sequence[object],
    grounded: bool,
    cube_colors: dict[int, str] | None = None,
) -> list[str]:
    # num_repeats governs eval repetitions only; the video always shows one pick-and-place.
    position_labels = _two_axis_position_labels(all_cubes, mentioned=all_cubes)
    colors = cube_colors or {}

    def color_of(actor) -> str | None:
        return colors.get(id(actor))

    def suffix(actor) -> str:
        return _coords_suffix(actor) if grounded else ""

    sentences = [
        f"The manipulated cube was initially "
        f"{_repick_cube_phrase(position_labels[id(target_cube)], color_of(target_cube), capitalize=False)}"
        f"{suffix(target_cube)}.",
    ]

    events = _simulate_nearest_neighbor_swaps_with_labels(
        all_cubes, swap_anchors, position_labels
    )
    for (
        anchor,
        partner,
        label_a,
        label_b,
        is_swap_back,
        is_immediate_swap_back,
    ) in _mark_swap_backs(events):
        if is_swap_back:
            if is_immediate_swap_back:
                sentences.append("They swapped back.")
                continue
            sentences.append(
                f"{_repick_cube_phrase(label_a, color_of(anchor), capitalize=True)} "
                f"and {_repick_cube_phrase(label_b, color_of(partner), capitalize=False)} "
                "then swapped back."
            )
            continue
        sentences.append(
            f"{_repick_cube_phrase(label_a, color_of(anchor), capitalize=True)}{suffix(anchor)} "
            "swapped with "
            f"{_repick_cube_phrase(label_b, color_of(partner), capitalize=False)}{suffix(partner)}."
        )

    return sentences


def _repick_final_position(
    target_cube,
    all_cubes: Sequence[object],
    swap_anchors: Sequence[object],
    position_labels: dict[int, str],
):
    """Replay the swap schedule (via `_simulate_nearest_neighbor_swaps`) to find
    where `target_cube`'s identity ends up.

    Also tracks which original actor's (untouched) pose now represents the final
    position — the physical swap has not executed yet when descriptions are built,
    so that actor's real pose gives correct final pixel coordinates.
    """
    current_label = dict(position_labels)
    current_actor = {id(actor): actor for actor in all_cubes}

    for anchor, partner in _simulate_nearest_neighbor_swaps(all_cubes, swap_anchors):
        current_label[id(anchor)], current_label[id(partner)] = (
            current_label[id(partner)],
            current_label[id(anchor)],
        )
        current_actor[id(anchor)], current_actor[id(partner)] = (
            current_actor[id(partner)],
            current_actor[id(anchor)],
        )

    return current_label[id(target_cube)], current_actor[id(target_cube)]


def _build_repick_hybrid_sentence(
    *,
    target_cube,
    all_cubes: Sequence[object],
    swap_anchors: Sequence[object],
    cube_colors: dict[int, str] | None = None,
) -> str:
    """Describe only the manipulated cube's final (post-swap) position, e.g.
    "The manipulated cube is the middle green cube at <105, 70>."."""
    position_labels = _two_axis_position_labels(all_cubes, mentioned=all_cubes)
    colors = cube_colors or {}
    final_label, final_actor = _repick_final_position(
        target_cube, all_cubes, swap_anchors, position_labels
    )
    return (
        f"The manipulated cube is "
        f"{_repick_cube_phrase(final_label, colors.get(id(target_cube)), capitalize=False)}"
        f"{_coords_suffix(final_actor)}."
    )


def build_repick_video_description(
    *,
    target_cube,
    all_cubes: Sequence[object],
    swap_anchors: Sequence[object] = (),
    cube_colors: dict[int, str] | None = None,
) -> dict[str, str]:
    return {
        "simple": " ".join(
            _build_repick_sentences(
                target_cube=target_cube,
                all_cubes=all_cubes,
                swap_anchors=swap_anchors,
                grounded=False,
                cube_colors=cube_colors,
            )
        ),
        "grounded": " ".join(
            _build_repick_sentences(
                target_cube=target_cube,
                all_cubes=all_cubes,
                swap_anchors=swap_anchors,
                grounded=True,
                cube_colors=cube_colors,
            )
        ),
        "hybrid": _build_repick_hybrid_sentence(
            target_cube=target_cube,
            all_cubes=all_cubes,
            swap_anchors=swap_anchors,
            cube_colors=cube_colors,
        ),
    }


def _build_place_sentences(
    *,
    demonstration_targets: Sequence[object],
    all_targets: Sequence[object],
    cube_color: str | None,
    button_after_n_placements: int | None,
    swap_pair: tuple[object, object] | None,
    grounded: bool,
) -> list[str]:
    mentioned_targets = list(demonstration_targets)
    if swap_pair is not None:
        for target in swap_pair:
            if all(id(existing) != id(target) for existing in mentioned_targets):
                mentioned_targets.append(target)

    labels = _two_axis_labels(all_targets, "target", mentioned=mentioned_targets)

    def labeled(target) -> str:
        base = f"the {labels[id(target)]}"
        return f"{base}{_coords_suffix(target)}" if grounded else base

    cube_phrase = f"The {cube_color} cube" if cube_color else "The cube"

    # Button event only inserted when button_after_n_placements is set (some tasks omit it).
    events: list[tuple[str, object | None]] = []
    for index, target in enumerate(demonstration_targets):
        events.append(("place", target))
        if (
            button_after_n_placements is not None
            and (index + 1) == button_after_n_placements
        ):
            events.append(("button", None))

    clauses: list[str] = []
    prev_kind: str | None = None
    for index, (kind, payload) in enumerate(events):
        if kind == "button":
            clauses.append(
                "The button was pressed" if index == 0 else "the button was pressed"
            )
        else:
            if index == 0:
                clauses.append(f"{cube_phrase} was placed on {labeled(payload)}")
            elif prev_kind == "button":
                clauses.append(f"the cube was placed on {labeled(payload)}")
            else:
                clauses.append(f"on {labeled(payload)}")
        prev_kind = kind

    sentences: list[str] = []
    if clauses:
        if len(clauses) == 1:
            sentences.append(clauses[0] + ".")
        else:
            parts = [clauses[0]]
            for clause in clauses[1:-1]:
                parts.append(f", {clause}")
            parts.append(f", and {clauses[-1]}")
            sentences.append("".join(parts) + ".")

    if swap_pair is not None:
        target_a, target_b = swap_pair
        sentences.append(f"{_cap(labeled(target_a))} swapped with {labeled(target_b)}.")

    return sentences


def build_place_video_description(
    *,
    demonstration_targets: Sequence[object],
    all_targets: Sequence[object],
    cube_color: str | None = None,
    button_after_n_placements: int | None = None,
    swap_pair: tuple[object, object] | None = None,
) -> dict[str, str]:
    shared_kwargs = dict(
        demonstration_targets=demonstration_targets,
        all_targets=all_targets,
        cube_color=cube_color,
        button_after_n_placements=button_after_n_placements,
        swap_pair=swap_pair,
    )
    return {
        "simple": " ".join(_build_place_sentences(**shared_kwargs, grounded=False)),
        "grounded": " ".join(_build_place_sentences(**shared_kwargs, grounded=True)),
    }


def build_placebutton_hybrid_video_description(
    *, target_target, all_targets: Sequence[object], timing: str
) -> str:
    """Resolve the correct target directly: "The cube was placed on the {label}
    {coords} {timing} the button was pressed." (`timing` is "before" or "after")."""
    label = _two_axis_labels(all_targets, "target", mentioned=[target_target])[
        id(target_target)
    ]
    return (
        f"The cube was placed on the {label}{_coords_suffix(target_target)} "
        f"{timing} the button was pressed."
    )


def build_placeorder_hybrid_video_description(
    *, target_target, all_targets: Sequence[object], ordinal_rank: int
) -> str:
    """Resolve the correct target directly: "The cube was placed on the {label}
    {coords} for the {ordinal} time."."""
    label = _two_axis_labels(all_targets, "target", mentioned=[target_target])[
        id(target_target)
    ]
    return (
        f"The cube was placed on the {label}{_coords_suffix(target_target)} "
        f"for the {_ordinal_word(ordinal_rank)} time."
    )


_MOVECUBE_STRATEGY_ACTIONS = {
    "peg_push": "pushed with the peg",
    "gripper_push": "pushed with the gripper",
    "grasp_putdown": "picked up and placed",
}


def build_movecube_video_description(*, strategy: str) -> dict[str, str]:
    """Describe the demonstrated manipulation strategy (peg push / gripper push / grasp-place)."""
    action = _MOVECUBE_STRATEGY_ACTIONS.get(strategy, "moved")
    text = f"The cube was {action} onto the target."
    return {"simple": text, "grounded": text}


def build_insertpeg_video_description(
    *, peg_position: str, grasp_end: str, insertion_side: str
) -> dict[str, str]:
    """Describe which peg (by spatial position), which end was grasped, and which hole it was inserted into."""
    text = (
        f"Picked the {peg_position} peg at the {grasp_end} end "
        f"and inserted it into the {insertion_side} hole."
    )
    return {"simple": text, "grounded": text}


def _times_phrase(count: int) -> str:
    """Return "once", "twice", or "{cardinal} times"."""
    if count == 1:
        return "once"
    if count == 2:
        return "twice"
    from robomme.robomme_env.utils.task_goal import num2words

    word = num2words.get(count, str(count))
    return f"{word} times"


def _group_consecutive_moves(move_sequence: Sequence[str]) -> list[tuple[str, int]]:
    """Run-length encode consecutive identical moves: ["right","right","left"] -> [("right",2),("left",1)]."""
    groups: list[tuple[str, int]] = []
    for move in move_sequence:
        if groups and groups[-1][0] == move:
            label, count = groups[-1]
            groups[-1] = (label, count + 1)
        else:
            groups.append((move, 1))
    return groups


def _patternlock_move_clauses(move_sequence: Sequence[str]) -> list[str]:
    groups = _group_consecutive_moves(move_sequence)
    return [
        f"{label} {_times_phrase(count)}" if count > 1 else label
        for label, count in groups
    ]


def _cap(s: str) -> str:
    """Capitalise the first character; empty string is returned unchanged."""
    return s[:1].upper() + s[1:]


def _join_then(clauses: Sequence[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return clauses[0] + "".join(f", {clause}" for clause in clauses[1:])


def build_patternlock_video_description(
    *, move_sequence: Sequence[str]
) -> dict[str, str]:
    """Describe the demonstration move sequence, collapsing repeats (e.g. "moved right twice")."""
    clauses = _patternlock_move_clauses(move_sequence)
    text = f"Moved {_join_then(clauses)}." if clauses else ""
    return {"simple": text, "grounded": text}


@dataclass
class SummaryEvent:
    """One ordered execution event for the summary log.

    A matching `merge_key` upgrades the earlier event in place rather than appending.
    Coordinates are "<y, x>" strings captured at emission time.
    `hybrid=True` opts the event into the hybrid summary channel.
    """

    verb: str
    object_group: str | None = None
    merge_key: object | None = None
    source_coord: str | None = None
    source_prep: str = "from"
    target_phrase: str | None = None
    target_coord: str | None = None
    target_prep: str = "into"
    hybrid: bool = False


def _join_events(clauses: Sequence[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]}, and {clauses[1]}"
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def render_event_log(events: Sequence[SummaryEvent]) -> dict[str, str]:
    """Render an ordered event log into short simple/grounded/hybrid summary strings.

    `hybrid` renders only events flagged `hybrid=True`, never grounded (no coordinates),
    with per-group ordinals recomputed within the flagged subset.
    """

    def _render_subset(subset: Sequence[SummaryEvent], grounded: bool) -> str:
        group_keys: dict[str, list] = {}
        for event in subset:
            if event.object_group is None or event.merge_key is None:
                continue
            keys = group_keys.setdefault(event.object_group, [])
            if event.merge_key not in keys:
                keys.append(event.merge_key)

        def _object_phrase(event: SummaryEvent) -> str | None:
            if event.object_group is None:
                return None
            keys = group_keys.get(event.object_group, [])
            if event.merge_key is not None and len(keys) > 1:
                rank = keys.index(event.merge_key) + 1
                return f"the {_ordinal_word(rank)} {event.object_group}"
            return f"the {event.object_group}"

        def _clause(event: SummaryEvent, seen_targets: set) -> str:
            parts = [event.verb]
            object_phrase = _object_phrase(event)
            if object_phrase:
                parts.append(object_phrase)
            if grounded and event.source_coord:
                parts.append(f"{event.source_prep} {event.source_coord}")
            if event.target_phrase:
                target = f"{event.target_prep} {event.target_phrase}"
                if (
                    grounded
                    and event.target_coord
                    and event.target_coord not in seen_targets
                ):
                    target = f"{target} at {event.target_coord}"
                    seen_targets.add(event.target_coord)
                parts.append(target)
            return " ".join(parts)

        seen_targets: set = set()
        clauses = [_clause(event, seen_targets) for event in subset]
        if not clauses:
            return ""
        return _cap(_join_events(clauses)) + "."

    hybrid_subset = [e for e in events if e.hybrid]
    return {
        "simple": _render_subset(events, grounded=False),
        "grounded": _render_subset(events, grounded=True),
        "hybrid": _render_subset(hybrid_subset, grounded=False),
    }


def build_patternlock_summary_text(*, completed_moves: Sequence[str]) -> dict[str, str]:
    """Describe completed moves so far, collapsing repeats (same format as the video description)."""
    clauses = _patternlock_move_clauses(completed_moves)
    text = f"Moved {_join_then(clauses)}." if clauses else ""
    return {"simple": text, "grounded": text}


def build_routestick_video_description(
    *, move_sequence: Sequence[str]
) -> dict[str, str]:
    """Describe the demonstration move sequence (each label encodes stick side + circling direction)."""
    clauses = _patternlock_move_clauses(move_sequence)
    text = f"Moved {_join_then(clauses)}." if clauses else ""
    return {"simple": text, "grounded": text}


def build_routestick_summary_text(*, completed_moves: Sequence[str]) -> dict[str, str]:
    """Describe completed RouteStick moves (side + circling direction) so far, collapsing repeats."""
    clauses = _patternlock_move_clauses(completed_moves)
    text = f"Moved {_join_then(clauses)}." if clauses else ""
    return {"simple": text, "grounded": text}
