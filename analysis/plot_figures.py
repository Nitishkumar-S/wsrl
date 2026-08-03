"""
Builds paper-compatible Figure 17 / Figure 9 style plots and a final
return table for Hopper-medium (WSRL vs IQL vs CQL, no-retention
and retention variants) from per-run wandb history exports.

Expected input (one file per run) in DATA_DIR, named
    {experiment}_seed{N}.parquet    (".csv" also accepted)
e.g. wsrl_seed0.parquet, iql_retain_seed2.csv, wsrl_halfcheetah_lr3e4_seed1.parquet.
Everything before the trailing "_seed{N}" is the experiment name; all seeds
sharing an experiment name are averaged together into one curve.

Which experiments exist and which ones share a plot is configured in the
EXPERIMENTS and PLOTS blocks below -- see the comments there.

Each file must have a "_step" column and the two eval metric columns
"evaluation/average_return" and "evaluation/average_normalized_return"
(a plain wandb run-history export, e.g. via run.history(pandas=True)
or the same format as wandb's parquet artifact download). ".csv"
files with the same columns are also accepted.

Usage:
    python analysis/plot_figures.py
"""

import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Folder holding the run files, and where figures/tables are written.
# Point these at a subfolder (e.g. "data" / "mine") to plot a different batch.
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

# Step at which offline pretraining ends / online fine-tuning begins
# (matches --num_offline_steps 250_000 used by all three launch scripts).
# The paper plots step 0 as the start of online fine-tuning, so we
# shift and drop the offline portion to match.
OFFLINE_STEPS = 250_000

# How many trailing eval points to average for the "final" return
# reported in the summary table (reduces single-point eval noise).
FINAL_WINDOW = 5

RETURN_COL = "evaluation/average_return"
NORM_RETURN_COL = "evaluation/average_normalized_return"

# Step column names seen across export formats: wandb's parquet/API history
# uses "_step", the chart "Download data" CSV uses "Step".
STEP_COLS = ["_step", "Step", "step", "global_step"]

# (random, expert) reference returns, mirroring get_locomotion_normalized_score()
# in finetune.py. Used to fill in whichever metric an export is missing: a chart
# CSV downloaded from wandb contains only the one metric that chart plots.
#
# Prefer downloading the RAW "evaluation/average_return" chart -- raw return is
# invariant, so normalizing here keeps every run on a figure consistent even if
# the reference pair in finetune.py is later corrected. (The reverse direction
# works too, but a normalized export is frozen against the constants that were
# live when that run executed.)
REF_SCORES = {
    "halfcheetah": (-281.05892, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker": (1.629008, 4592.3),
    "humanoid": (105.712692, 8602.9),
}

# Fallback environment when a file name doesn't contain any REF_SCORES key,
# e.g. "wsrl_seed0.csv". Set to None to make that case an error instead.
DEFAULT_ENV = "humanoid"

# When an export has both metrics, recompute the normalized score from the raw
# return using REF_SCORES rather than trusting the logged column. Keeps parquet
# runs on the same footing as CSV-only runs (whose normalized score is derived
# here anyway) and immune to REF_SCORES changing after a run finished. Set False
# to plot exactly what was logged.
RENORMALIZE_FROM_RAW = True

FILE_RE = re.compile(r"^(?P<experiment>.+)_seed(?P<seed>\d+)\.(?:parquet|csv)$")

# Okabe-Ito palette: distinguishable under all common forms of colour blindness
# and, unlike matplotlib's defaults, still separable through a projector's
# washed-out contrast. Every figure draws from these.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"

# Bigger type, heavier lines, light grid -- readable from the back of a room.
# Tweak FIGSIZE/DPI here rather than per figure.
FIGSIZE = (8, 5.5)
DPI = 200
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.8,
})

# ---------------------------------------------------------------------------
# CONFIG 1: the experiments (= file-name prefixes before "_seed{N}").
#
# Add one entry per experiment you want available. Anything found in DATA_DIR
# but missing here still loads -- it just gets an auto-assigned colour and its
# raw name as the legend label. Keys must match the file names exactly.
#
# The KEY is the file name with "_seed{N}.parquet" chopped off; you never list
# the seeds, every file sharing that prefix is picked up and averaged. So if
# DATA_DIR contains
#
#     wsrl_halfcheetah_lr3e-4_seed0.parquet
#     wsrl_halfcheetah_lr3e-4_seed1.parquet
#     wsrl_halfcheetah_lr3e-4_seed2.parquet
#
# the key is "wsrl_halfcheetah_lr3e-4" and those three files become one mean
# curve with a +/-1 std band. Mixed extensions and non-contiguous seed numbers
# are fine (seed0.csv + seed7.parquet works). Then reference that same key in
# PLOTS below to put it on a figure.
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # The baseline WSRL run doubles as the 5k-warmup arm of Figure 2, the
    # CQL-pretrain arm of Figure 3 and the 0%-retention arm of Figure 4; each
    # renames and recolours it via its own "styles" block.
    "wsrl": dict(label="WSRL", color=BLUE, linestyle="-"),
    "iql": dict(label="IQL", color=ORANGE, linestyle="-"),
    "cql": dict(label="CQL", color=PURPLE, linestyle="-"),

    # Figure 2: --warmup_steps sweep, all resuming the same CQL pretrain.
    "1kWarmup": dict(label="1000", color=BLUE, linestyle="-"),
    "20kWarmup": dict(label="20000", color=GREEN, linestyle="-"),

    # Figure 4: --offline_data_ratio sweep. 0% is the baseline wsrl run (WSRL
    # retains nothing by construction); colours ramp green -> red with the
    # retained fraction, which is also the order of the outcome.
    "wsrl-ablation-retain25-finetune": dict(
        label="25%", color=BLUE, linestyle="-",
    ),
    "wsrl-ablation-retain50-finetune": dict(
        label="50%", color=ORANGE, linestyle="-",
    ),
    "wsrl-ablation-retain75-finetune": dict(
        label="75%", color=VERMILLION, linestyle="-",
    ),

    # Figure 5: reward scale/bias sweep, pretraining + fine-tuning on one axis.
    "reward_default_full": dict(
        label="scale 1.0, bias 0.0 (default)", color=BLUE, linestyle="-",
    ),
    "reward_bias5_full": dict(
        label="scale 1.0, bias -5.0", color=ORANGE, linestyle="-",
    ),
    "reward_scale10_full": dict(
        label="scale 10.0, bias 0.0", color=GREEN, linestyle="-",
    ),

    # Figure 1: what the warmup buffer is filled with before online RL starts.
    "randomWarmup": dict(
        label="Random action init", color=BLUE, linestyle="-",
    ),
    # WSRL fine-tuned off an IQL pretrain (Figure 3 renames this to "IQL").
    "wsrl_iql_finetuning": dict(
        label="WSRL (IQL pretrain)", color=ORANGE, linestyle="-",
    ),

    # Example -- a specific named experiment, reading
    # wsrl_halfcheetah_lr3e-4_seed{0,1,2,...}.parquet:
    # "wsrl_halfcheetah_lr3e-4": dict(
    #     label="WSRL (lr 3e-4)", color="#e7298a", linestyle="-",
    # ),
}

# ---------------------------------------------------------------------------
# CONFIG 1b: composite experiments -- one curve spanning BOTH phases.
#
# Pretraining and fine-tuning are logged as two separate wandb runs (fine-tuning
# resumes from the pretrain checkpoint and restarts its own logging), so a full
# 0..750k curve needs both files. Each entry names two file-name prefixes; seeds
# are matched by number, and a seed present in only one half is skipped.
#
# Style these like any other experiment via EXPERIMENTS / a figure's "styles",
# and give the figure phase_line=OFFLINE_STEPS to mark the handover.
# ---------------------------------------------------------------------------
COMPOSITES = {
    "reward_default_full": dict(pretrain="wsrl_pretrain", finetune="wsrl"),
    "reward_bias5_full": dict(
        pretrain="reward-scale1-bias5_cql", finetune="reward-scale1-bias5_sac",
    ),
    "reward_scale10_full": dict(
        pretrain="reward-scale10-bias0_cql", finetune="reward-scale10-bias0_sac",
    ),
}

# Colours handed out to experiments that aren't listed in EXPERIMENTS.
_FALLBACK_COLORS = itertools.cycle(
    ["#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#1f78b4"]
)

# ---------------------------------------------------------------------------
# CONFIG 2: which experiments go together in which figure.
#
# One dict per figure:
#   experiments : list of experiment names to draw on the same axes
#   title       : figure title
#   outfile     : file name written into OUT_DIR
#   metric      : "normalized" (default) or "raw"
#   required    : if True, warn+skip the figure when an experiment has no data;
#                 if False (default), just drop the missing ones and plot the rest
# ---------------------------------------------------------------------------
PLOTS = [
    # Figure 1
    dict(
        experiments=["randomWarmup", "wsrl"],
        title="Policy Warmup vs Random Warmup",
        outfile="fig1_warmup_policy_vs_random.png",
        metric="normalized",
        required=True,
        styles={"wsrl": dict(label="Policy warmup", color=ORANGE)},
    ),
    # Figure 2
    dict(
        experiments=["1kWarmup", "wsrl", "20kWarmup"],
        title="Different Warmup Steps",
        outfile="fig2_warmup_steps.png",
        metric="normalized",
        required=True,
        styles={"wsrl": dict(label="5000", color=ORANGE)},
    ),
    # Figure 3 -- WSRL fine-tuning off an IQL vs a CQL pretrain; the CQL arm is
    # the baseline wsrl run.
    dict(
        experiments=["wsrl_iql_finetuning", "wsrl"],
        title="Different Types of Value Initialization",
        outfile="fig3_value_init.png",
        metric="normalized",
        required=True,
        styles={
            "wsrl_iql_finetuning": dict(label="IQL", color=BLUE),
            "wsrl": dict(label="CQL", color=GREEN),
        },
    ),
    # Figure 4
    dict(
        experiments=[
            "wsrl",
            "wsrl-ablation-retain25-finetune",
            "wsrl-ablation-retain50-finetune",
            "wsrl-ablation-retain75-finetune",
        ],
        title="Humanoid Specific - Different Offline Data Retained",
        outfile="fig4_offline_data_retained.png",
        metric="normalized",
        required=True,
        styles={"wsrl": dict(label="0%", color=GREEN)},
    ),
    # Figure 5
    dict(
        experiments=["reward_default_full", "reward_bias5_full", "reward_scale10_full"],
        title="Humanoid specific - Reward Configuration",
        outfile="fig5_reward_scale_bias.png",
        metric="normalized",
        required=True,
        xlabel="Steps",
        phase_line=OFFLINE_STEPS,
    ),
]

# Column index into a loaded run tuple (steps, returns, norm_returns).
METRICS = {
    "raw": dict(index=1, ylabel="Average Return"),
    "normalized": dict(index=2, ylabel="Normalized Score"),
}


def style_for(experiment, overrides=None):
    """Plot style for an experiment, invented on the fly if not configured.

    `overrides` is a figure's optional "styles" block; the same run can appear on
    two figures under different labels/colours (e.g. baseline WSRL is the "5000"
    arm of the warmup sweep and the "CQL" arm of the value-init figure).
    """
    if experiment not in EXPERIMENTS:
        EXPERIMENTS[experiment] = dict(
            label=experiment, color=next(_FALLBACK_COLORS), linestyle="-"
        )
    style = {**EXPERIMENTS[experiment], **(overrides or {}).get(experiment, {})}
    return dict(
        label=style.get("label", experiment),
        color=style.get("color", "#333333"),
        linestyle=style.get("linestyle", "-"),
    )


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _find_step_col(df, name):
    for col in STEP_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"{name}: no step column, looked for {STEP_COLS}, got {list(df.columns)}")


def _find_metric_col(df, metric):
    """Locate a metric column across export formats.

    A parquet/API export names it exactly ("evaluation/average_return"); a chart
    CSV prefixes it with the run name ("wsrl_..._seed0 - evaluation/average_return")
    and adds "__MIN"/"__MAX" band columns we must not pick up.
    """
    if metric in df.columns:
        return metric
    candidates = [
        c for c in df.columns
        if c.endswith(metric) and "__MIN" not in c and "__MAX" not in c
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous columns for {metric}: {candidates} -- keep one run per file"
        )
    return candidates[0] if candidates else None


def _ref_scores(name):
    """(random, expert) reference pair for a file, inferred from its name."""
    lowered = name.lower()
    for env, scores in REF_SCORES.items():
        if env in lowered:
            return scores
    if DEFAULT_ENV is None:
        raise ValueError(
            f"{name}: can't tell which env this is -- put one of {list(REF_SCORES)} "
            f"in the file name, or set DEFAULT_ENV"
        )
    return REF_SCORES[DEFAULT_ENV]


def _load_run(path: Path, shift_offline=True):
    """Load one run file.

    `shift_offline=False` keeps the raw absolute step numbers and the offline
    rows -- used by composite experiments, which show pretraining and
    fine-tuning on one 0..750k axis instead of the usual fine-tuning-only view.
    """
    df = _read(path)
    step_col = _find_step_col(df, path.name)
    raw_col = _find_metric_col(df, RETURN_COL)
    norm_col = _find_metric_col(df, NORM_RETURN_COL)
    if raw_col is None and norm_col is None:
        raise ValueError(
            f"{path.name}: found neither {RETURN_COL!r} nor {NORM_RETURN_COL!r} "
            f"in columns {list(df.columns)}"
        )

    value_cols = [c for c in (raw_col, norm_col) if c is not None]
    df = df[[step_col] + value_cols].dropna()
    if shift_offline:
        df = df[df[step_col] >= OFFLINE_STEPS]
    df = df.sort_values(step_col)
    steps = df[step_col].to_numpy(dtype=float)
    if shift_offline:
        steps = steps - OFFLINE_STEPS

    # Whichever metric the export is missing is reconstructed from the other --
    # normalization is linear in the raw return, so this is exact.
    random_score, expert_score = _ref_scores(path.name)
    if raw_col is not None:
        returns = df[raw_col].to_numpy(dtype=float)
    else:
        norm = df[norm_col].to_numpy(dtype=float)
        returns = random_score + norm / 100.0 * (expert_score - random_score)
    if norm_col is not None and raw_col is not None and not RENORMALIZE_FROM_RAW:
        norm_returns = df[norm_col].to_numpy(dtype=float)
    else:
        norm_returns = 100.0 * (returns - random_score) / (expert_score - random_score)

    return steps, returns, norm_returns


def _seed_files(prefix):
    """{seed: path} for every file named "{prefix}_seed{N}.*"."""
    found = {}
    for path in sorted(DATA_DIR.glob("*")):
        m = FILE_RE.match(path.name)
        if m and m.group("experiment") == prefix:
            found[m.group("seed")] = path
    return found


def _load_composite(name, spec):
    """Join a pretraining run and a fine-tuning run per seed, on absolute steps.

    The two phases are logged as separate wandb runs (pretraining ends where
    fine-tuning resumes from its checkpoint), so a seed's full 0..750k curve is
    the concatenation of the two files.
    """
    pretrain, finetune = _seed_files(spec["pretrain"]), _seed_files(spec["finetune"])
    unpaired = set(pretrain) ^ set(finetune)
    if unpaired:
        print(f"{name}: skipping unpaired seed(s) {sorted(unpaired)} "
              f"(pretrain={sorted(pretrain)}, finetune={sorted(finetune)})")

    runs = {}
    for seed in sorted(set(pretrain) & set(finetune)):
        halves = [
            _load_run(pretrain[seed], shift_offline=False),
            _load_run(finetune[seed], shift_offline=False),
        ]
        merged = [np.concatenate([h[i] for h in halves]) for i in range(3)]
        order = np.argsort(merged[0], kind="stable")
        runs[seed] = tuple(col[order] for col in merged)
    return runs


def load_all_runs(only=None):
    """Load every {experiment}_seed{N} file in DATA_DIR.

    `only` optionally restricts loading to a set of experiment names. Names
    listed in COMPOSITES are assembled from two files per seed instead.
    Returns {experiment: {seed: (steps, returns, norm_returns)}}.
    """
    runs = {}
    for name, spec in COMPOSITES.items():
        if only is None or name in only:
            loaded = _load_composite(name, spec)
            if loaded:
                runs[name] = loaded

    for path in sorted(DATA_DIR.glob("*")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        experiment, seed = m.group("experiment"), m.group("seed")
        if only is not None and experiment not in only:
            continue
        runs.setdefault(experiment, {})[seed] = _load_run(path)
    return runs


def aggregate(seed_runs, value_index, n_points=200):
    """Interpolate each seed's curve onto a common step grid, then mean/std."""
    # Span only where every seed has data, so the ends aren't held flat by
    # np.interp's clamping (composite curves start at the first eval, not 0).
    min_step = max(steps.min() for steps, *_ in seed_runs.values())
    max_step = min(steps.max() for steps, *_ in seed_runs.values())
    grid = np.linspace(min_step, max_step, n_points)
    interped = np.stack(
        [np.interp(grid, run[0], run[value_index]) for run in seed_runs.values()]
    )
    return grid, interped.mean(axis=0), interped.std(axis=0)


def plot_group(runs, experiments, title, outfile, metric="normalized",
               required=False, styles=None, xlabel="Online Fine-Tuning Steps",
               phase_line=None):
    if metric not in METRICS:
        raise ValueError(f"{outfile}: unknown metric {metric!r}, expected one of {list(METRICS)}")
    value_index = METRICS[metric]["index"]
    ylabel = METRICS[metric]["ylabel"]

    missing = [name for name in experiments if name not in runs]
    present = [name for name in experiments if name in runs]
    if missing:
        print(f"{outfile}: no data for {missing}")
    if not present or (missing and required):
        print(f"skipping {outfile}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in present:
        grid, mean, std = aggregate(runs[name], value_index)
        style = style_for(name, styles)
        ax.plot(grid, mean, label=style["label"], color=style["color"],
                linestyle=style["linestyle"], solid_capstyle="round")
        ax.fill_between(grid, mean - std, mean + std, color=style["color"],
                        alpha=0.18, linewidth=0)

    if phase_line is not None:
        ax.axvline(phase_line, color="#C00000", linestyle="--", linewidth=2,
                   zorder=1)
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for text, dx, ha in (("Pretraining", -0.012, "right"),
                             ("Finetuning", 0.012, "left")):
            ax.text(phase_line + dx * span, 0.965, text, transform=ax.get_xaxis_transform(),
                    ha=ha, va="top", fontsize=13, style="italic", color="#333333")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    # Steps run to 500k -- "200k" beats "200000" on a projected axis.
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    )
    ax.set_xlim(left=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="best")
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / outfile, dpi=DPI)
    plt.close(fig)
    print(f"saved {OUT_DIR / outfile}")


def build_return_table(runs):
    rows = []
    for experiment, seed_runs in runs.items():
        for seed, (steps, returns, norm_returns) in sorted(seed_runs.items()):
            final_raw = returns[-FINAL_WINDOW:].mean()
            final_norm = norm_returns[-FINAL_WINDOW:].mean()
            rows.append(dict(experiment=experiment, seed=seed, final_return=final_raw,
                              final_normalized_return=final_norm))
    table = pd.DataFrame(rows).sort_values(["experiment", "seed"])

    summary = (
        table.groupby("experiment")[["final_return", "final_normalized_return"]]
        .agg(["mean", "std"])
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "final_returns_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "final_returns_summary.csv")
    print(f"saved {OUT_DIR / 'final_returns_per_seed.csv'}")
    print(f"saved {OUT_DIR / 'final_returns_summary.csv'}")
    print()
    print(f"Per-seed final returns (mean of last {FINAL_WINDOW} eval points):")
    print(table.to_string(index=False))
    print()
    print("Per-experiment summary (mean +/- std across seeds):")
    print(summary.to_string())


def main():
    wanted = {name for spec in PLOTS for name in spec["experiments"]}
    runs = load_all_runs(only=wanted)
    if not runs:
        print(f"no run files found in {DATA_DIR} for experiments {sorted(wanted)}")
        found = sorted({m.group("experiment") for p in DATA_DIR.glob("*")
                        if (m := FILE_RE.match(p.name))})
        if found:
            print(f"available experiments in {DATA_DIR}: {found}")
        return

    print(f"loaded: " + ", ".join(
        f"{name} ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})"
        for name, seeds in sorted(runs.items())
    ))

    for spec in PLOTS:
        plot_group(runs, **spec)

    build_return_table(runs)


if __name__ == "__main__":
    main()
