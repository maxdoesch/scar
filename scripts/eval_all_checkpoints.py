#!/usr/bin/env python3
"""Run one evaluation job per checkpoint in runs/ckpts/mme_vla_suite/.

Checkpoints that already have a log.json in runs/evaluation/ are skipped.
Each job finds a free port in 8000-8008 at runtime.
serve_policy.py runs under .venv (via uv); robomme-eval runs under
.venv-robomme (see robomme_eval/readme.md).

Two backends are supported via --backend:
  slurm (default): submit one job per batch with `sbatch` (original behavior).
  local: run each batch as a foreground subprocess, one batch at a time, so
    this also works on workstations/servers without a Slurm scheduler. Use
    CUDA_VISIBLE_DEVICES before invoking this script to restrict which
    GPU(s) are visible to the server/eval processes, since JAX otherwise
    defaults to using all visible GPUs.
"""

import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import subprocess
import textwrap

CKPT_ROOT = Path("runs/ckpts")
EVAL_BASE = Path("runs/evaluation")
SLURM_DIR = Path("slurm")

# Max characters of an experiment name kept in a script/job label before it is
# truncated (a short hash is appended to preserve uniqueness).
EXP_LABEL_MAXLEN = 24


def abbrev_exp(exp_name: str) -> str:
    """Return a short but human-readable token for an experiment name.

    Truncates long names on a ``-`` boundary and appends a 6-char hash of the
    full name so distinct experiments never collide after truncation.
    """
    if len(exp_name) <= EXP_LABEL_MAXLEN:
        return exp_name
    short = exp_name[:EXP_LABEL_MAXLEN].rsplit("-", 1)[0].rstrip("-_")
    digest = hashlib.md5(exp_name.encode()).hexdigest()[:6]
    return f"{short}_{digest}"


def runs_label(runs: list) -> str:
    """Build a compact label for a batch of (suite, exp_name, ckpt_id, seed) runs.

    Runs sharing a suite and experiment are grouped so checkpoint ids are
    listed once per experiment, keeping the label short and readable.
    """
    groups: OrderedDict[tuple[str, str], list] = OrderedDict()
    for suite, exp_name, ckpt_id, _ in runs:
        groups.setdefault((suite, exp_name), []).append(ckpt_id)
    return "+".join(
        f"{abbrev_exp(suite)}-{abbrev_exp(exp)}_ck{'-'.join(ckpts)}" for (suite, exp), ckpts in groups.items()
    )


def find_checkpoints(base: Path):
    """Yield (exp_name, ckpt_id) for every numeric checkpoint directory."""
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        for ckpt_dir in sorted(exp_dir.iterdir()):
            if ckpt_dir.is_dir() and ckpt_dir.name.isdigit():
                yield exp_dir.name, ckpt_dir.name


def is_evaluated(exp_name: str, ckpt_id: str, seed: int) -> bool:
    log_path = EVAL_BASE / exp_name / f"ckpt{ckpt_id}" / f"seed{seed}" / "log.json"
    return log_path.exists()


def resolve_repo_id(ckpt_path: Path) -> str | None:
    """Return the asset/norm-stats subdirectory stored inside a checkpoint.

    Checkpoints are saved with their norm stats under ``<ckpt>/assets/<repo_id>/``.
    The ``repo_id`` may differ from the serving config's default (e.g. a dataset
    variant like ``robomme-bf-bu-vpb-rs``), so detect it here and pass it explicitly
    to serve_policy.py. Returns None if it cannot be determined unambiguously.
    """
    assets_dir = ckpt_path / "assets"
    if not assets_dir.is_dir():
        return None
    subdirs = [p.name for p in sorted(assets_dir.iterdir()) if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return None


def make_job_script(
    workdir: Path,
    runs: list,  # list of (suite, exp_name, ckpt_id, seed) tuples
    only_tasks: str,
    discrete_state_input: bool = False,
) -> str:
    """Generate a Slurm job script that runs up to len(runs) evals in parallel on one GPU."""
    n = len(runs)
    tasks_flag = f"--args.only-tasks {only_tasks}" if only_tasks else ""
    discrete_state_input_flag = "--policy.model_discrete_state_input True" if discrete_state_input else ""

    # Build the port-finding block: find n free ports
    port_block = textwrap.dedent(f"""\
        # Find {n} free port(s) in the range 8000-8099 (randomised order to avoid collisions)
        PORTS=()
        for p in $(shuf -i 8000-8099); do
            if ! ss -tlnp 2>/dev/null | grep -q ":$p "; then
                PORTS+=($p)
                if [[ ${{#PORTS[@]}} -ge {n} ]]; then
                    break
                fi
            fi
        done
        if [[ ${{#PORTS[@]}} -lt {n} ]]; then
            echo "Not enough free ports available in range 8000-8099" >&2
            exit 1
        fi
        echo "Using port(s): ${{PORTS[*]}}"
    """)

    # Build server launch lines
    server_lines = [
        "source .venv/bin/activate",
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45",
        "SERVER_PIDS=()",
    ]
    for i, (suite, exp_name, ckpt_id, seed) in enumerate(runs):
        ckpt_path = f"runs/ckpts/{suite}/{exp_name}/{ckpt_id}"
        repo_id = resolve_repo_id(Path(ckpt_path))
        server_cmd = (
            f"uv run scripts/serve_policy.py \\\n"
            f"    --seed={seed} \\\n"
            f"    --port=${{PORTS[{i}]}} \\\n"
            f"    policy:checkpoint \\\n"
            f"    --policy.dir={ckpt_path} \\\n"
            f"    --policy.repo_id={repo_id} \\\n"
            f"    --policy.config={suite}"
        )
        if discrete_state_input_flag:
            server_cmd += f" \\\n    {discrete_state_input_flag}"
        server_cmd += " &"
        server_lines.append(server_cmd)
        server_lines.append("SERVER_PIDS+=($!)")
    server_block = "\n".join(server_lines)

    # Build eval launch lines
    eval_lines = ["source .venv-robomme/bin/activate", "EVAL_PIDS=()"]
    for i, (suite, exp_name, ckpt_id, seed) in enumerate(runs):
        eval_cmd = (
            f"robomme-eval \\\n"
            f"    --args.model_seed={seed} \\\n"
            f"    --args.host=127.0.0.1 \\\n"
            f"    --args.port=${{PORTS[{i}]}} \\\n"
            f"    --args.policy_name={exp_name} \\\n"
            f"    --args.model_ckpt_id={ckpt_id}"
        )
        if tasks_flag:
            eval_cmd += f" \\\n    {tasks_flag}"
        eval_cmd += " &"
        eval_lines.append(eval_cmd)
        eval_lines.append("EVAL_PIDS+=($!)")
    eval_lines.append("# Wait for all evals to finish")
    eval_lines.append('for pid in "${EVAL_PIDS[@]}"; do wait "$pid"; done')
    eval_block = "\n".join(eval_lines)

    return f"""\
#!/usr/bin/env bash
set -euo pipefail

cd {workdir}

# Stagger startup to reduce port collisions when many jobs launch together
sleep $((RANDOM % 60))

{port_block}
# Start policy server(s) under .venv (via uv)
{server_block}
trap 'for _pid in "${{SERVER_PIDS[@]}}"; do kill "$_pid" 2>/dev/null || true; done' EXIT

sleep 20

# Run evaluation(s) under .venv-robomme
{eval_block}
"""


def submit_job(
    workdir: Path,
    runs: list,  # list of (suite, exp_name, ckpt_id, seed) tuples
    only_tasks: str,
    n_gpus: int,
    dry_run: bool,
    discrete_state_input: bool = False,
    backend: str = "slurm",
):
    # Name script and job after all runs in the batch
    _, _, _, seed = runs[0]
    label = runs_label(runs)
    script_path = SLURM_DIR / f"eval_{label}_seed{seed}.sh"
    script_path.write_text(make_job_script(workdir, runs, only_tasks, discrete_state_input))
    script_path.chmod(0o755)

    if dry_run:
        print(f"  [dry-run] script written to {script_path}")
        return

    if backend == "local":
        # Run the batch script directly, in the foreground, one batch at a time.
        # Set CUDA_VISIBLE_DEVICES before invoking this script to restrict GPU
        # visibility (n_gpus/--cpus-per-gpu only apply to the Slurm backend).
        print(f"  [local] running {script_path} (waiting for completion)...")
        result = subprocess.run(["bash", str(script_path)], check=False)
        if result.returncode != 0:
            print(f"  ERROR: {script_path} exited with code {result.returncode}")
        else:
            print(f"  [local] finished {script_path}")
        return

    job_name = f"eval_{label}"
    cpus_per_gpu = 8 * len(runs)
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--gres=gpu:{n_gpus}",
        f"--cpus-per-gpu={cpus_per_gpu}",
        "--output=slurm/slurm-%j.out",
        str(script_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    else:
        print(f"  {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Run one eval job per checkpoint under runs/ckpts/<suite>/.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["slurm", "local"],
        default="slurm",
        help=(
            "Execution backend (default: slurm). 'slurm' submits one sbatch job per "
            "batch (original behavior). 'local' runs each batch as a foreground "
            "subprocess, one batch at a time, in this shell -- useful on machines "
            "without a Slurm scheduler. --n-gpus/--cpus-per-gpu are ignored for "
            "'local'; set CUDA_VISIBLE_DEVICES before invoking this script instead "
            "to control which GPU(s) are visible."
        ),
    )
    parser.add_argument(
        "--suite",
        type=str,
        nargs="+",
        default=["mme_vla_suite"],
        help=(
            "Checkpoint suite directories under runs/ckpts/ to evaluate "
            "(default: mme_vla_suite). Multiple values are accepted, "
            "e.g. --suite mme_vla_suite mme_vla_suite_lerobot."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[7],
        help="Model seed(s) (default: 7). Multiple values are accepted, e.g. --seed 0 1 2.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated task names to evaluate. Omit to evaluate all tasks.",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        nargs="+",
        default=None,
        help="Optional checkpoint ID(s) to evaluate (e.g. --ckpt 79999 or --ckpt 20000 40000).",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="GPUs per job (default: 1)")
    parser.add_argument(
        "--runs-per-gpu",
        type=int,
        default=1,
        help="Number of eval runs to co-locate on a single GPU (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write job scripts and print what would be run/submitted, without executing them.",
    )
    parser.add_argument(
        "--discrete-state-input",
        action="store_true",
        help="Set discrete_state_input to true in the model config during evaluation.",
    )
    args = parser.parse_args()

    workdir = Path.cwd().resolve()
    SLURM_DIR.mkdir(exist_ok=True)

    selected_ckpts = {str(ckpt_id) for ckpt_id in args.ckpt} if args.ckpt is not None else None
    pending = []
    skipped = 0
    for suite in args.suite:
        ckpt_base = CKPT_ROOT / suite
        checkpoints = list(find_checkpoints(ckpt_base))
        if selected_ckpts is not None:
            checkpoints = [(exp_name, ckpt_id) for exp_name, ckpt_id in checkpoints if ckpt_id in selected_ckpts]
        print(f"[{suite}] Found {len(checkpoints)} checkpoint(s).\n")

        # Collect unevaluated checkpoints
        for exp_name, ckpt_id in checkpoints:
            for seed in args.seed:
                if is_evaluated(exp_name, ckpt_id, seed):
                    print(f"[skip]   {suite}/{exp_name}/{ckpt_id}  seed={seed}  (log.json already exists)")
                    skipped += 1
                else:
                    pending.append((suite, exp_name, ckpt_id, seed))

    # Batch pending runs across all selected suites into groups of --runs-per-gpu
    submitted = 0
    for i in range(0, len(pending), args.runs_per_gpu):
        batch = pending[i : i + args.runs_per_gpu]
        labels = ", ".join(f"{s}/{e}/{c}" for s, e, c, _ in batch)
        print(f"[submit] {labels}")
        submit_job(
            workdir,
            batch,
            args.tasks,
            args.n_gpus,
            args.dry_run,
            args.discrete_state_input,
            backend=args.backend,
        )
        submitted += len(batch)

    print(f"\nSubmitted: {submitted}  |  Skipped (already evaluated): {skipped}")


if __name__ == "__main__":
    main()
