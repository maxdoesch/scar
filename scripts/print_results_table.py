"""
Print a mean ± std success-rate table for every experiment found under
runs/evaluation/, separated with | for easy reading and as LaTeX code.

Usage:
    uv run scripts/print_results_table.py
    uv run scripts/print_results_table.py --eval_dir runs/evaluation
    uv run scripts/print_results_table.py --eval_dir runs/evaluation --ckpt ckpt29999
"""

import argparse
from decimal import ROUND_HALF_UP
from decimal import Decimal
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval_dir", type=str, default="runs/evaluation", help="Root directory containing per-experiment subdirectories."
)
parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="If given, only include this checkpoint (e.g. ckpt29999). Otherwise all checkpoints are averaged together.",
)
args = parser.parse_args()

EVAL_DIR = Path(args.eval_dir)

# ── discover experiments ──────────────────────────────────────────────────────


# Trailing keyword in name.txt marking a checkpoint for spatial separation at the table bottom.
SEPARATE_KEYWORD = "separate"


def collect_rows(
    eval_dir: Path, ckpt_filter: str | None
) -> tuple[dict[str, dict[str, list[float]]], set[str], set[str]]:
    """
    Returns {experiment_label: {task_name: [seed_values, ...]}}
    Values are in [0, 100] (percentage).
    """
    rows: dict[str, dict[str, list[float]]] = {}
    raw_latex_labels: set[str] = set()
    separate_labels: set[str] = set()

    for exp_dir in sorted(eval_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        name_path = exp_dir / "name.txt"
        run_name = name_path.read_text().strip() if name_path.exists() else exp_dir.name
        run_name = run_name or exp_dir.name

        is_separate = False
        name_tokens = run_name.split()
        if name_tokens and name_tokens[-1].lower() == SEPARATE_KEYWORD:
            is_separate = True
            run_name = " ".join(name_tokens[:-1]).strip() or exp_dir.name

        for ckpt_dir in sorted(exp_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if ckpt_filter and ckpt_dir.name != ckpt_filter:
                continue

            label = run_name if ckpt_filter else f"{run_name}/{ckpt_dir.name}"
            task_values: dict[str, list[float]] = {}

            for seed_dir in sorted(ckpt_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                log_path = seed_dir / "log.json"
                if not log_path.exists():
                    continue
                with open(log_path) as f:
                    data = json.load(f)
                for task, val in data.get("success_rate", {}).items():
                    task_values.setdefault(task, []).append(float(val) * 100)

            if task_values:
                rows[label] = task_values
                if name_path.exists() and run_name:
                    raw_latex_labels.add(label)
                if is_separate:
                    separate_labels.add(label)

    return rows, raw_latex_labels, separate_labels


rows, raw_latex_labels, separate_labels = collect_rows(EVAL_DIR, args.ckpt)

if not rows:
    print(f"No log.json files found under {EVAL_DIR}.")
    raise SystemExit(1)

# ── collect ordered task list (union across all experiments) ──────────────────
task_set: set[str] = set()
for task_values in rows.values():
    task_set.update(task_values.keys())
ALL_TASKS = sorted(task_set)

# Preferred grouped ordering for table output.
TASK_GROUP_SPECS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Temporal",
        [
            ("BinFill", "BinFill", r"Bin\\Fill"),
            ("PickXtimes", "PickXtimes", r"Pick\\Xtimes"),
            ("SwingXtimes", "SwingXtimes", r"Swing\\Xtimes"),
            ("StopCube", "StopCube", r"Stop\\Cube"),
        ],
    ),
    (
        "Spatial",
        [
            ("VideoUnmask", "VideoUnmask", r"Video\\Unmask"),
            ("ButtonUnmask", "ButtonUnmask", r"Button\\Unmask"),
            ("VideoUnmaskSwap", "VideoUnmaskSwap", r"Video\\Unmask\\Swap"),
            ("ButtonUnmaskSwap", "ButtonUnmaskSwap", r"Button\\Unmask\\Swap"),
        ],
    ),
    (
        "Object",
        [
            ("PickHighlight", "PickHighlight", r"Pick\\Highlight"),
            ("VideoRepick", "VideoRepick", r"Video\\Repick"),
            ("VideoPlaceButton", "VideoPlaceButton", r"Video\\Place\\Button"),
            ("VideoPlaceOrder", "VideoPlaceOrder", r"Video\\Place\\Order"),
        ],
    ),
    (
        "Procedural",
        [
            ("MoveCube", "MoveCube", r"Move\\Cube"),
            ("InsertPeg", "InsertPeg", r"Insert\\Peg"),
            ("PatternLock", "PatternLock", r"Pattern\\Lock"),
            ("RouteStick", "RouteStick", r"Route\\Stick"),
        ],
    ),
]

task_display_plain: dict[str, str] = {}
task_display_latex: dict[str, str] = {}
for _, specs in TASK_GROUP_SPECS:
    for task, plain_name, latex_name in specs:
        task_display_plain[task] = plain_name
        task_display_latex[task] = latex_name

active_groups: list[tuple[str, list[str]]] = []
for group_name, specs in TASK_GROUP_SPECS:
    present_tasks = [task for task, _, _ in specs if task in ALL_TASKS]
    if present_tasks:
        active_groups.append((group_name, present_tasks))

grouped_tasks = [task for _, tasks in active_groups for task in tasks]
ungrouped_tasks = [task for task in ALL_TASKS if task not in grouped_tasks]

# Functional-requirement grouping (see docs figure): category -> tasks.
FUNCTIONAL_GROUP_SPECS: list[tuple[str, list[str]]] = [
    ("Motion-Centric", ["PatternLock", "RouteStick", "SwingXtimes", "InsertPeg"]),
    ("Time-Sensitive", ["StopCube"]),
    ("Short-Horizon Video Reasoning", ["VideoUnmask", "VideoRepick", "VideoUnmaskSwap"]),
    ("Long-Horizon Video Reasoning", ["VideoPlaceButton", "VideoPlaceOrder"]),
    ("Dynamic Online Scene-Change", ["ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight"]),
    ("Event-Salient", ["PickXtimes", "BinFill", "MoveCube"]),
]

active_functional_groups: list[tuple[str, list[str]]] = []
for group_name, tasks in FUNCTIONAL_GROUP_SPECS:
    present_tasks = [task for task in tasks if task in ALL_TASKS]
    if present_tasks:
        active_functional_groups.append((group_name, present_tasks))


# ── compute stats per (experiment, task) ─────────────────────────────────────
def stats(values: list[float]) -> tuple[float, float]:
    a = np.array(values)
    return float(np.mean(a)), float(np.std(a, ddof=0))


# Build a structured result: {label: {task: (mean, std)}}
results: dict[str, dict[str, tuple[float, float]]] = {}
for label, task_values in rows.items():
    results[label] = {}

    # per-seed overall: average across tasks that this seed reported
    seed_task_lists: list[list[float]] = []
    n_seeds = max(len(v) for v in task_values.values())
    for s in range(n_seeds):
        seed_vals = [task_values[t][s] for t in ALL_TASKS if t in task_values and s < len(task_values[t])]
        seed_task_lists.append(seed_vals)

    for task in ALL_TASKS:
        if task in task_values:
            results[label][task] = stats(task_values[task])
        else:
            results[label][task] = (float("nan"), float("nan"))

    # overall = mean over tasks, then mean/std over seeds
    overall_per_seed = [np.mean(sv) for sv in seed_task_lists if sv]
    results[label]["Overall"] = stats(overall_per_seed)

    # per task-group = mean over the group's tasks per seed, then mean/std over seeds
    for group_name, tasks in active_groups:
        group_per_seed = []
        for s in range(n_seeds):
            vals = [task_values[t][s] for t in tasks if t in task_values and s < len(task_values[t])]
            if vals:
                group_per_seed.append(np.mean(vals))
        results[label][group_name] = stats(group_per_seed) if group_per_seed else (float("nan"), float("nan"))

    # per functional-group = mean over the group's tasks per seed, then mean/std over seeds
    for group_name, tasks in active_functional_groups:
        group_per_seed = []
        for s in range(n_seeds):
            vals = [task_values[t][s] for t in tasks if t in task_values and s < len(task_values[t])]
            if vals:
                group_per_seed.append(np.mean(vals))
        results[label][group_name] = stats(group_per_seed) if group_per_seed else (float("nan"), float("nan"))

    # Display runs from lowest to highest overall average performance; "separate" runs always go last.
    results = dict(sorted(results.items(), key=lambda item: (item[0] in separate_labels, item[1]["Overall"][0])))

COLUMNS = grouped_tasks + ungrouped_tasks + ["Overall"]
GROUP_COLUMNS = [group_name for group_name, _ in active_groups] + ["Overall"]
FUNC_GROUP_COLUMNS = [group_name for group_name, _ in active_functional_groups] + ["Overall"]

# ── formatting helpers ────────────────────────────────────────────────────────


def fmt(mean: float, std: float, sep: str = " ± ") -> str:
    if np.isnan(mean):
        return "—"
    return f"{format_number(mean)}{sep}{format_number(std)}"


def format_number(value: float, decimals: int = 1) -> str:
    """Round only for final display, using conventional half-up rounding."""
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


# ── plain table with | separators ─────────────────────────────────────────────

col_header = "Method"


def plain_col_name(col: str) -> str:
    if col == "Overall":
        return "AVG"
    return task_display_plain.get(col, col)


plain_task_headers = [plain_col_name(c) for c in COLUMNS]
col_widths = [max(len(col_header), max(len(lbl) for lbl in results))]
col_widths += [
    max(len(h), max(len(fmt(*results[lbl][c])) for lbl in results)) for h, c in zip(plain_task_headers, COLUMNS)
]


def row_str(cells: list[str], widths: list[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def centered_group_row(widths: list[int]) -> str:
    cells: list[str] = ["".center(widths[0])]

    col_offset = 1
    for group_name, tasks in active_groups:
        span_width = sum(widths[col_offset : col_offset + len(tasks)]) + 3 * (len(tasks) - 1)
        cells.append(group_name.center(span_width))
        col_offset += len(tasks)

    if ungrouped_tasks:
        span_width = sum(widths[col_offset : col_offset + len(ungrouped_tasks)]) + 3 * (len(ungrouped_tasks) - 1)
        cells.append("Other".center(span_width))
        col_offset += len(ungrouped_tasks)

    cells.append("".center(widths[-1]))
    return " | ".join(cells)


header_cells = [col_header] + plain_task_headers
separator = "-+-".join("-" * w for w in col_widths)

print()
print("=" * len(separator))
print("SUCCESS RATE TABLE  (mean ± std across seeds, in %)")
print("=" * len(separator))
print(centered_group_row(col_widths))
print(row_str(header_cells, col_widths))
print(separator)
prev_was_separate = False
for label, task_stats in results.items():
    is_separate_row = label in separate_labels
    if is_separate_row and not prev_was_separate:
        print(separator)
    cells = [label] + [fmt(*task_stats[c]) for c in COLUMNS]
    print(row_str(cells, col_widths))
    prev_was_separate = is_separate_row
print("=" * len(separator))

# ── LaTeX table ───────────────────────────────────────────────────────────────

# Find per-column best mean for bolding
best_mean_per_col: dict[str, float] = {}
for col in COLUMNS:
    vals = [
        results[lbl][col][0] for lbl in results if lbl not in separate_labels and not np.isnan(results[lbl][col][0])
    ]
    best_mean_per_col[col] = max(vals) if vals else float("nan")


def latex_fmt(mean: float, std: float, bold: bool = False) -> str:
    if np.isnan(mean):
        return "---"
    if bold:
        return rf"$\mathbf{{{format_number(mean)}}}^{{\pm{format_number(std)}}}$"
    return rf"${format_number(mean)}^{{\pm{format_number(std)}}}$"


def latex_label(label: str) -> str:
    if label in raw_latex_labels:
        return label
    return label.replace("_", r"\_")


labels = list(results.keys())
is_split_table = len(COLUMNS) > 8
if is_split_table:
    task_columns = [column for column in COLUMNS if column != "Overall"]
    split_at = len(task_columns) // 2
    column_chunks = [task_columns[:split_at], task_columns[split_at:]]
    render_chunks = [
        columns + (["Overall"] if index == len(column_chunks) - 1 else [""])
        for index, columns in enumerate(column_chunks)
    ]
else:
    column_chunks = [COLUMNS]
    render_chunks = [COLUMNS]

data_column_count = max(len(columns) for columns in render_chunks)
col_spec = "l" + "c" * data_column_count

lines: list[str] = [
    "",
    "% LaTeX table — paste into your document",
    r"\begin{table}[h]",
    r"  \centering",
    r"  \small",
    r"  \setlength{\tabcolsep}{2.5pt}",
    rf"  \begin{{tabular}}{{{col_spec}}}",
]

for chunk_index, columns in enumerate(column_chunks):
    render_columns = render_chunks[chunk_index] + [""] * (data_column_count - len(render_chunks[chunk_index]))
    chunk_grouped_tasks = [col for col in grouped_tasks if col in columns]
    chunk_ungrouped_tasks = [col for col in ungrouped_tasks if col in columns]

    top_header_cells = ["" if is_split_table else r"\textbf{Method}"]
    for group_name, tasks in active_groups:
        group_tasks = [task for task in tasks if task in columns]
        if group_tasks:
            top_header_cells.append(rf"\multicolumn{{{len(group_tasks)}}}{{c}}{{\textbf{{{group_name}}}}}")
    if chunk_ungrouped_tasks:
        top_header_cells.append(rf"\multicolumn{{{len(chunk_ungrouped_tasks)}}}{{c}}{{\textbf{{Other}}}}")

    if "Overall" in render_columns and not is_split_table:
        top_header_cells.append(r"\textbf{AVG}")
    elif is_split_table:
        top_header_cells.append(r"\multicolumn{1}{c}{}")
    latex_top_header = " & ".join(top_header_cells) + r" \\"
    cmidrules: list[str] = []
    start_col = 2
    for _, tasks in active_groups:
        group_tasks = [task for task in tasks if task in columns]
        if group_tasks:
            end_col = start_col + len(group_tasks) - 1
            cmidrules.append(rf"    \cmidrule(lr){{{start_col}-{end_col}}}")
            start_col = end_col + 1
    if chunk_ungrouped_tasks:
        end_col = start_col + len(chunk_ungrouped_tasks) - 1
        cmidrules.append(rf"    \cmidrule(lr){{{start_col}-{end_col}}}")

    second_header_cells = [r"\textbf{Method}" if is_split_table else ""]
    for col in render_columns:
        if col == "Overall":
            second_header_cells.append(r"\textbf{AVG}" if is_split_table else "")
        elif col in task_display_latex:
            second_header_cells.append(rf"\textbf{{\makecell{{{task_display_latex[col]}}}}}")
        else:
            second_header_cells.append(rf"\textbf{{{col.replace('_', r'\_')}}}")

    latex_second_header = " & ".join(second_header_cells) + r" \\"
    section_rules = [r"    \toprule"] if chunk_index == 0 else [r"    \bottomrule", r"    \toprule"]
    lines += section_rules + [
        "    " + latex_top_header,
        *cmidrules,
        "    " + latex_second_header,
        r"    \midrule",
    ]

    for i, label in enumerate(labels):
        task_stats = results[label]
        escaped_label = latex_label(label)
        model_cell = rf"\textbf{{{escaped_label}}}"
        data_cells = [
            ""
            if not col
            else latex_fmt(
                task_stats[col][0],
                task_stats[col][1],
                bold=not np.isnan(task_stats[col][0]) and np.isclose(task_stats[col][0], best_mean_per_col[col]),
            )
            for col in render_columns
        ]
        lines.append("    " + " & ".join([model_cell] + data_cells) + r" \\")
        if i < len(labels) - 1:
            if label not in separate_labels and labels[i + 1] in separate_labels:
                lines.append(r"    \specialrule{1pt}{1pt}{2pt}")
            else:
                lines.append(r"    \midrule")

lines += [
    r"    \bottomrule",
    r"  \end{tabular}",
    r"  \caption{Evaluation results. Success rate (\%) and standard deviation across seeds.}",
    r"  \label{tab:results}",
    r"\end{table}",
    "",
]

print()
print("=" * len(separator))
print("LATEX CODE")
print("=" * len(separator))
print("\n".join(lines))

# ── generic group-average table printers (plain + LaTeX) ─────────────────────


def group_col_name(col: str) -> str:
    return "AVG" if col == "Overall" else col


def print_plain_group_table(columns: list[str], title: str) -> None:
    headers = [group_col_name(c) for c in columns]
    widths = [max(len(col_header), max(len(lbl) for lbl in results))]
    widths += [max(len(h), max(len(fmt(*results[lbl][c])) for lbl in results)) for h, c in zip(headers, columns)]

    header_row = [col_header] + headers
    sep = "-+-".join("-" * w for w in widths)

    print()
    print("=" * len(sep))
    print(title)
    print("=" * len(sep))
    print(row_str(header_row, widths))
    print(sep)
    prev_was_separate = False
    for label, task_stats in results.items():
        is_separate_row = label in separate_labels
        if is_separate_row and not prev_was_separate:
            print(sep)
        cells = [label] + [fmt(*task_stats[c]) for c in columns]
        print(row_str(cells, widths))
        prev_was_separate = is_separate_row
    print("=" * len(sep))


def print_latex_group_table(columns: list[str], title: str, comment: str, caption: str, label_id: str) -> None:
    best_mean_per_col: dict[str, float] = {}
    for col in columns:
        vals = [
            results[lbl][col][0] for lbl in results if lbl not in separate_labels and not np.isnan(results[lbl][col][0])
        ]
        best_mean_per_col[col] = max(vals) if vals else float("nan")

    col_spec = "l" + "c" * len(columns)
    header_cells = [r"\textbf{Method}"] + [rf"\textbf{{{group_col_name(c)}}}" for c in columns]
    header_row = " & ".join(header_cells) + r" \\"

    out_lines: list[str] = [
        "",
        f"% LaTeX table — {comment}, paste into your document",
        r"\begin{table}[h]",
        r"  \centering",
        r"  \small",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    " + header_row,
        r"    \midrule",
    ]

    for i, label in enumerate(labels):
        task_stats = results[label]
        escaped_label = latex_label(label)
        model_cell = rf"\textbf{{{escaped_label}}}"
        data_cells = [
            latex_fmt(
                task_stats[col][0],
                task_stats[col][1],
                bold=not np.isnan(task_stats[col][0]) and np.isclose(task_stats[col][0], best_mean_per_col[col]),
            )
            for col in columns
        ]
        out_lines.append("    " + " & ".join([model_cell] + data_cells) + r" \\")
        if i < len(labels) - 1:
            if label not in separate_labels and labels[i + 1] in separate_labels:
                out_lines.append(r"    \specialrule{1pt}{1pt}{4pt}")
            else:
                out_lines.append(r"    \midrule")

    out_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label_id}}}",
        r"\end{table}",
        "",
    ]

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("\n".join(out_lines))


print_plain_group_table(GROUP_COLUMNS, "TASK-GROUP AVERAGE TABLE  (mean ± std across seeds, in %)")
print_latex_group_table(
    GROUP_COLUMNS,
    "LATEX CODE (TASK-GROUP AVERAGES)",
    "task-group averages",
    r"Evaluation results averaged per task group. Success rate (\%) and standard deviation across seeds.",
    "tab:results_group_avg",
)

print_plain_group_table(
    FUNC_GROUP_COLUMNS, "FUNCTIONAL-REQUIREMENT GROUP AVERAGE TABLE  (mean ± std across seeds, in %)"
)
print_latex_group_table(
    FUNC_GROUP_COLUMNS,
    "LATEX CODE (FUNCTIONAL-REQUIREMENT GROUP AVERAGES)",
    "functional-requirement group averages",
    r"Evaluation results averaged per functional-requirement category. Success rate (\%) and standard deviation across seeds.",
    "tab:results_functional_group_avg",
)
