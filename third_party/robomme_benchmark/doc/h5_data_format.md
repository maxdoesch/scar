# HDF5 Training Data Format

Structure inside each `record_dataset_<EnvID>.h5` file:

```text
episode_1/
  setup/
  timestep_1/
    obs/
    action/
    info/
  timestep_2/
    obs/
    action/
    info/
  ...
...
```

Each episode contains:
- `setup/`: episode-level configuration.
- `timestep_<K>/`: per-timestep data.

## Field semantics: `subgoal` vs. `video_description` vs. `summary`

Three related-but-distinct natural-language fields appear across `info/` and `setup/`.
Each has its own `_simple`/`_grounded` variant pair, and their content must never
overlap:

- **`subgoal`** (`info/simple_subgoal`, `info/grounded_subgoal`, and their `_online`
  counterparts): describes **only what the robot is currently doing** — no completed
  progress, no past events, no dynamic environment changes.
- **`video_description`** (`setup/video_description_simple`,
  `setup/video_description_grounded`): a **static, episode-level** description of the
  memory-relevant facts and ordered events shown in the conditioning video
  demonstration, present only for video-based tasks (see the field description below).
  Written once per episode; does not change across timesteps.
- **`summary`** (`info/summary_simple`, `info/summary_grounded`): a **per-timestep**
  description of memory-relevant events that have happened during robot execution so
  far. It combines relevant robot progress with relevant dynamic environment changes,
  excludes the current subgoal and any purely observable state that is not a past
  event worth remembering, and stays empty (`""`) until the first memory-relevant
  event occurs. The field is maintained as one up-to-date sentence per
  memory-relevant category (for example, a cube-count category or a button-progress
  category): each category's sentence is replaced with its current cumulative state
  whenever that category changes, while multiple independent categories coexist by
  concatenation in first-appearance order.

## `setup/` fields (episode configuration)

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Environment seed (fixed for benchmarking) |
| `difficulty` | `str` | Difficulty level (fixed for benchmarking) |
| `task_goal` | `list[str]` | Possible language goals for the task |
| `video_description_simple` | `str` | optional episode-level natural-language description of the memory-relevant facts and ordered events shown in the conditioning video demonstration (qualitative left-right and near-far position words only, e.g. "leftmost" or "top-right corner", no numeric locations); describes historical facts/events without resolving the task-specific answer. Present only for the 9 video-based tasks: VideoUnmask, VideoUnmaskSwap, VideoRepick, VideoPlaceButton, VideoPlaceOrder, MoveCube, InsertPeg, PatternLock, and RouteStick. Field is omitted (not written) for all other tasks, including ButtonUnmask and ButtonUnmaskSwap, which have no conditioning video demonstration |
| `video_description_grounded` | `str` | same content and task scope as `video_description_simple`, with numeric base_camera pixel coordinates (`<y, x>`, same format as grounded subgoal placeholders) annotated after each mentioned object; the qualitative left-right/near-far wording itself is still derived from table coordinates |
| `front_camera_intrinsic` | `float32 (3, 3)` | Front camera intrinsic matrix |
| `wrist_camera_intrinsic` | `float32 (3, 3)` | Wrist camera intrinsic matrix |
| `available_multi_choices` | `str` | Available options for the multi-choice Video-QA problem |

## `obs/` fields (observations)

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `front_rgb` | `uint8 (256, 256, 3)` | Front camera RGB |
| `wrist_rgb` | `uint8 (256, 256, 3)` | Wrist camera RGB |
| `front_depth` | `int16 (256, 256, 1)` | Front camera depth (mm) |
| `wrist_depth` | `int16 (256, 256, 1)` | Wrist camera depth (mm) |
| `joint_state` | `float32 (7,)` | Absolute joint positions (7 joints) |
| `eef_state` | `float32 (6,)` | Absolute end-effector pose `[x, y, z, roll, pitch, yaw]` |
| `gripper_state` | `float32 (2,)` | Gripper opening width in [0, 0.04] |
| `is_gripper_close` | `bool` | Whether gripper is closed |
| `front_camera_extrinsic` | `float32 (3, 4)` | Front camera extrinsic matrix |
| `wrist_camera_extrinsic` | `float32 (3, 4)` | Wrist camera extrinsic matrix |

## `action/` fields

| Field | Type / shape | Description |
|-------|---------------|-------------|
| `joint_action` | `float32 (8,)` | Absolute joint-space action: 7 joint angles + gripper |
| `eef_action` | `float32 (7,)` | Absolute end-effector action `[x, y, z, roll, pitch, yaw, gripper]` |
| `waypoint_action` | `float32 (7,)` | Absolute end-effector action at discrete time steps; a subtask may contain multiple waypoint actions. Used for data generation. |
| `choice_action` | `str` | JSON string for multi-choice selection with an optional grounded pixel location on the front image, e.g., `{"choice": "A", "point": [y, x]}` |

In RoboMME, a gripper action of -1 means close and 1 means open.

## `info/` fields (metadata)

| Field | Type | Description |
|-------|------|-------------|
| `simple_subgoal` | `bytes (UTF-8)` | Simple subgoal text (built-in planner view) |
| `simple_subgoal_online` | `bytes (UTF-8)` | Simple subgoal text (online view; may advance to the next subgoal earlier than planner view) |
| `grounded_subgoal` | `bytes (UTF-8)` | Grounded subgoal text (built-in planner view) |
| `grounded_subgoal_online` | `bytes (UTF-8)` | Grounded subgoal text (online view; may advance to the next subgoal earlier than planner view) |
| `summary_simple` | `bytes (UTF-8)` | Running natural-language summary of memory-relevant events that have happened during robot execution so far (robot progress plus relevant dynamic environment changes). Always present; empty string `""` until the first memory-relevant event has occurred. Never restates the current subgoal. Maintained as one current sentence per memory-relevant category: a category's sentence is replaced whenever that category's cumulative state changes, while multiple categories coexist by concatenation in first-appearance order |
| `summary_grounded` | `bytes (UTF-8)` | Same content semantics as `summary_simple`, with numeric base_camera pixel coordinates (`<y, x>`) annotated after each mentioned object/target where grounding is applicable |
| `is_video_demo` | `bool` | Whether this frame is from the conditioning video shown before execution |
| `is_subgoal_boundary` | `bool` | Whether this is a keyframe (i.e., a boundary between subtasks) |
| `is_completed` | `bool` | Whether the task is finished |
