# SCAR: Supervised, Compressed, and Aligned Memory-Augmented VLA Policy Learning

This repository implements **SCAR** (Supervised, Compressed, and Aligned memory Representations), a
memory-augmented vision-language-action (VLA) method for long-horizon robotic manipulation. SCAR
combines (1) an auxiliary textual-supervision objective for memory-relevant events, (2) a learned
temporal-compression module for the perceptual memory, and (3) a staged training curriculum that
first aligns the memory pathway before jointly fine-tuning the full model. SCAR is evaluated on the
16 memory-dependent manipulation tasks of the [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark),
building on a $\pi_{0.5}$-based training/serving stack (see [Acknowledgement](#acknowledgement) for provenance).

## Outline

- [Installation](#installation)
  - [Install this repository](#install-this-repository)
  - [Install the RoboMME benchmark subrepository](#install-the-robomme-benchmark-subrepository)
  - [Install the RoboMME evaluation harness](#install-the-robomme-evaluation-harness)
- [Repository Structure](#repository-structure)
- [Download Pre-trained Backbones](#download-pre-trained-backbones)
- [Data Preparation](#data-preparation)
  - [1. Generate the 16-task dataset](#1-generate-the-16-task-dataset)
  - [2. Convert HDF5 to LeRobot format](#2-convert-hdf5-to-lerobot-format)
  - [3. Compute normalization statistics](#3-compute-normalization-statistics)
- [Model Training](#model-training)
  - [Train SCAR (main method)](#train-scar-main-method)
  - [Train other configs / baselines](#train-other-configs--baselines)
- [Evaluation](#evaluation)
  - [Manual evaluation (single checkpoint)](#manual-evaluation-single-checkpoint)
  - [Batch evaluation across checkpoints](#batch-evaluation-across-checkpoints)
- [Acknowledgement](#acknowledgement)


## Installation

### Install this repository
This repository (VLA training/serving) uses [uv](https://docs.astral.sh/uv/getting-started/installation/) and targets Python 3.12+:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set the `OPENPI_DATA_HOME` path in your `~/.bashrc`, e.g. `export OPENPI_DATA_HOME=<your_openpi_homedir>`. For more details, please refer to [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file#fine-tuned-models).

This installs an isolated `.venv` used for all `uv run scripts/*.py` commands below (data conversion, norm-stats, training, serving).

### Install the RoboMME benchmark subrepository
[`third_party/robomme_benchmark`](third_party/robomme_benchmark) is the RoboMME simulator/benchmark
codebase (vendored as a subdirectory of this repo, not a git submodule). It defines the 16 tasks and
is used to **generate training/eval data** and to run the simulator during evaluation. It is a
standalone `uv` project with its own `pyproject.toml`/`uv.lock` (Python 3.11+), kept separate from
the main `.venv` because it depends on ManiSkill/Vulkan:
```bash
cd third_party/robomme_benchmark
uv sync
uv pip install -e .
cd ../..
```
This creates `third_party/robomme_benchmark/.venv`. See [`third_party/robomme_benchmark/readme.md`](third_party/robomme_benchmark/readme.md)
for simulator prerequisites (Vulkan/display setup, Docker option) and its own quickstart.

### Install the RoboMME evaluation harness
`robomme_eval` (root-level `robomme_eval/`) drives policy evaluation against the simulator. It is
also a standalone `uv` project (pins `torch`/`transformers` and depends on ManiSkill, which conflict
with the main JAX-based `.venv`), installed into its own `.venv-robomme`:
```bash
uv venv .venv-robomme --python 3.11
uv pip install -e robomme_eval --python .venv-robomme
```
Full details, including the optional `flash-attn` install and a smoke test, are in [`robomme_eval/readme.md`](robomme_eval/readme.md).
We use a WebSocket connection between the VLA policy server (`.venv`) and the simulator (`.venv-robomme`), following [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main).


## Repository Structure
```
.
├── data
│   └── robomme_lerobot                 # per-task LeRobot datasets (output of convert_h5_to_lerobot.py)
├── packages
│   └── openpi-client                   # VLA client & server interface
├── robomme_eval                        # RoboMME simulator evaluation harness (standalone uv project)
├── runs
│   ├── assets                          # norm_stats.json files (per training config)
│   ├── ckpts                           # fine-tuned checkpoints
│   └── evaluation                      # evaluation results
├── scripts                             # data conversion/train/eval scripts
├── src
│   ├── mme_vla_suite                   # SCAR / MME-VLA model, training & config code, follows openpi structure
│   └── openpi                          # original OpenPi code with minor changes
└── third_party
    └── robomme_benchmark                # RoboMME simulator + 16-task dataset generation (standalone uv project)
```

This repository is built on top of [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main). We highly recommend becoming familiar with OpenPi first before working with this repo.

## Download Pre-trained Backbones
Download the $\pi_{0.5}$-base backbone (used to initialize all training configs below):
```bash
uv run scripts/download_pi05_base.py
```
Download the [pi05_vision_encoder](https://huggingface.co/Yinpei/pi05_vision_encoder), which is a subset of the $\pi_{0.5}$ parameters used for dataset feature construction without loading the full model. Visual token embeddings are computed and cached during LeRobot conversion, and the vision encoder remains frozen during training:
```bash
cd $OPENPI_DATA_HOME
git clone git@hf.co:Yinpei/pi05_vision_encoder
```

`runs/ckpts`, `runs/evaluation`, and `runs/assets` are created automatically as needed by the
scripts above/below; create them upfront with `mkdir -p runs/{ckpts,evaluation,assets}` if you
prefer.


## Data Preparation

The training pipeline is: **generate a raw HDF5 dataset for the 16 RoboMME tasks with the benchmark
subrepo → convert it to LeRobot format (this repo) → compute normalization statistics (this repo)**.
`train.py`'s data loader (`RoboMMEDatasetLerobot`) only reads the LeRobot format, so the conversion
step is required even if you already have raw HDF5 data.

### 1. Generate the 16-task dataset
Using the `third_party/robomme_benchmark` environment installed above, generate one HDF5 file per
task (`record_dataset_<Task>.h5`) for the train split, covering all 16 tasks (`BinFill`,
`PickXtimes`, `SwingXtimes`, `StopCube`, `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`,
`ButtonUnmaskSwap`, `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`,
`MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick`):
```bash
cd third_party/robomme_benchmark
export CUDA_VISIBLE_DEVICES=0
uv run scripts/generate_dataset.py \
    --dataset train \
    --task-id All \
    --difficulty all \
    --output-dir runs/generated_dataset/train
cd ../..
```
Pass `--task-id BinFill RouteStick` (space-separated) instead of `All` to generate only specific
tasks. Repeat with `--dataset val`/`--dataset test` (and a matching `--output-dir`) if you also want
held-out splits. The script is idempotent/resumable, and `--dataset debug` is useful for a quick
smoke test with far fewer episodes.

### 2. Convert HDF5 to LeRobot format
Back in the main repo's `.venv`, convert the generated HDF5 files into per-task LeRobot datasets with
cached SigLIP visual-token embeddings:
```bash
export CUDA_VISIBLE_DEVICES=0
uv run scripts/convert_h5_to_lerobot.py \
    --raw_data_path third_party/robomme_benchmark/runs/generated_dataset/train \
  --output_path data/robomme_lerobot \
  --fps 20 \
  --resolution 4x4
```
This produces `data/robomme_lerobot/record_dataset_<Task>/` (one LeRobot dataset directory per
task), which `train.py` and `compute_norm_stats.py` auto-discover from the parent `data/robomme_lerobot`
directory. See the script's `--help` for embedding-resolution/camera-view options.

### 3. Compute normalization statistics
Compute normalization statistics for each training config you plan to use (this takes a few
minutes):
```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name scar --repo-id robomme --dataset-path="data/robomme_lerobot"
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name mme_vla_suite --repo-id robomme --dataset-path="data/robomme_lerobot"
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_baseline --repo-id robomme --dataset-path="data/robomme_lerobot"
```
This writes `norm_stats.json` under `runs/assets/<config_name>/robomme/`:
```
runs
└── assets
    ├── scar
    │   └── robomme
    │       └── norm_stats.json
    ├── mme_vla_suite
    │   └── robomme
    │       └── norm_stats.json
    └── pi05_baseline
        └── robomme
            └── norm_stats.json
```

## Model Training

Training uses `scripts/train.py`, built on OpenPi's `tyro`-based config CLI. Configs are named
(`scar`, `mme_vla_suite`, `pi05_baseline`, in `src/mme_vla_suite/training/config.py`); the memory
*architecture* variant (e.g. which perceptual-memory encoder, integration type, and auxiliary
losses) is selected via `--model.history_config=<name>.yaml`, with all available variants defined
under [`src/mme_vla_suite/models/config/robomme`](src/mme_vla_suite/models/config/robomme).

### Train SCAR (main method)
SCAR uses the `scar` config with the history config
`perceptual-temporal-conv-framesamp-every-n-expert-subtask-summary.yaml`, which combines the three
SCAR components:
- **Supervised** — `aux_lang_task: subtask_summary` adds an auxiliary textual-summary prediction loss over memory-relevant events.
- **Compressed** — `perceptual_memory.encoder_type: temporal_conv` applies a learned temporal-compression module over the sampled observation history.
- **Aligned** — the two-stage Soft Memory Alignment curriculum (memory-only warm-up before jointly fine-tuning the full model) is a property of the `scar` `TrainConfig` itself (`mem_only_warmup_steps=20_000`, `mem_only_warmup_lr_multiplier=0.1`), so it applies automatically whenever `--model.use_history` is set; no extra flag is needed.

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py scar \
    --exp-name=<model_name> \
    --batch-size=64 \
    --num-workers=4 \
    --fsdp-devices=4 \
    --dataset-path=data/robomme_lerobot \
    --model.use_history \
    --model.history_config=perceptual-temporal-conv-framesamp-every-n-expert-subtask-summary.yaml
```
Checkpoints are written to `runs/ckpts/scar/<model_name>/<step>`. Pass `--resume` to resume
from the latest checkpoint of the same `--exp-name`. Set `--wandb-enabled=False` to disable W&B
logging.

### Train other configs / baselines
The $\pi_{0.5}$ baseline does not use history and fine-tunes $\pi_{0.5}$ with the vision encoder
frozen (for comparison with SCAR). It uses the original RoboMME/$\pi_{0.5}$ constant-learning-rate
recipe (no warmup, no memory-alignment curriculum):
```bash
uv run scripts/train.py pi05_baseline \
    --exp-name=<model_name> \
    --batch-size=64 \
    --dataset-path=data/robomme_lerobot
```

The `mme_vla_suite` config uses the same memory architecture as `scar` but trains it with the
original RoboMME constant-learning-rate recipe instead of Soft Memory Alignment, matching the
RoboMME `FrameSamp+Expert` baseline. Other perceptual-memory variants/ablations (e.g.
`perceptual-framesamp-expert.yaml`, `perceptual-framesamp-modul.yaml`,
`perceptual-framesamp-context.yaml`) can be trained the same way by swapping
`--model.history_config` on either the `scar` or `mme_vla_suite` config, depending on which
training recipe you want:
```bash
export MME_VLA_TYPE="perceptual-framesamp-expert"
uv run scripts/train.py mme_vla_suite \
    --exp-name=<model_name> \
    --batch-size=64 \
    --dataset-path=data/robomme_lerobot \
    --model.use_history \
    --model.history_config="${MME_VLA_TYPE}.yaml"
```

## Evaluation

Evaluation runs a policy server (`.venv`, JAX/serving) and the RoboMME simulator harness
(`.venv-robomme`) side by side, connected over a WebSocket, following OpenPi.

### Manual evaluation (single checkpoint)
```bash
# terminal 0: serve the policy
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --seed=7 --port=8000 \
    policy:checkpoint \
    --policy.dir=runs/ckpts/scar/<model_name>/<step> \
    --policy.config=scar

# terminal 1: run the RoboMME evaluation harness
source .venv-robomme/bin/activate
CUDA_VISIBLE_DEVICES=1 robomme-eval --args.model_seed=7 --args.port=8000 \
    --args.policy_name=<model_name> --args.model_ckpt_id=<step>
```
Results are stored in `runs/evaluation/<model_name>/ckpt<step>/seed7/log.json`.
> Manually set `CUDA_VISIBLE_DEVICES` to a single card for `serve_policy.py`, as JAX otherwise uses all visible GPUs by default.

### Batch evaluation across checkpoints
`scripts/eval_all_checkpoints.py` scans `runs/ckpts/<suite>/` and runs the manual flow above for
every checkpoint that doesn't already have a `log.json`, resolving free ports automatically. It
supports two execution backends via `--backend`:

- **`--backend slurm`** (default): submits one `sbatch` job per batch of runs, requesting `--n-gpus`
  GPU(s) per job — unchanged from the original behavior, for Slurm clusters.
- **`--backend local`**: runs each batch as a foreground subprocess directly in the current shell,
  one batch at a time, so it also works on a single workstation/server without a job scheduler. Set
  `CUDA_VISIBLE_DEVICES` before invoking the script to control which GPU(s) are used (`--n-gpus` is
  ignored for this backend).

```bash
# Slurm cluster (default)
uv run scripts/eval_all_checkpoints.py --suite scar --seed 7

# Single machine without Slurm
CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_all_checkpoints.py --suite scar --seed 7 --backend local
```
Use `--ckpt <step> [<step> ...]` to restrict to specific checkpoints, `--tasks <Task1,Task2,...>` to
restrict to specific RoboMME tasks, and `--dry-run` to preview the generated job script(s) without
running/submitting them. `uv run scripts/eval_all_checkpoints.py --help` lists all options.


## Acknowledgement
This repository is adapted from [`robomme_policy_learning`](https://github.com/RoboMME/robomme_policy_learning),
the official MME-VLA policy-learning codebase for the RoboMME benchmark:
```
@article{dai2026robomme,
  title={RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies},
  author={Dai, Yinpei and Fu, Hongze and Lee, Jayjun and and Liu, Yuejiang and Zhang, Haoran and Yang, Jianing and Finn, Chelsea and Fazeli, Nima and Chai, Joyce},
  journal={arXiv preprint arXiv:2603.04639},
  year={2026}
}
```
