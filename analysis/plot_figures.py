"""
Builds paper-compatible Figure 17 / Figure 9 style plots and a final
return table for halfcheetah-medium (WSRL vs IQL vs CQL, no-retention
and retention variants) from per-run wandb history exports.

Expected input (one file per run) in analysis/data/, named:
    wsrl_seed{N}.parquet          - WSRL, no retention
    iql_seed{N}.parquet           - IQL, no retention
    cql_seed{N}.parquet           - CQL, no retention
    iql_retain_seed{N}.parquet    - IQL, retains offline data
    cql_retain_seed{N}.parquet    - CQL, retains offline data

Each file must have a "_step" column and the two eval metric columns
"evaluation/average_return" and "evaluation/average_normalized_return"
(a plain wandb run-history export, e.g. via run.history(pandas=True)
or the same format as wandb's parquet artifact download). ".csv"
files with the same columns are also accepted.

Usage:
    python analysis/plot_figures.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Approximate final normalized scores (0-100 scale) hand-read off the WSRL
# paper's Figure 17, halfcheetah-medium-replay-v2 panel, at its 300k-step
# budget (Zhou et al. 2025, "Efficient Online RL Fine-Tuning Need Not
# Retain Offline Data"). The paper does not test halfcheetah-medium-v0 at
# all (only random/expert/medium-replay), so medium-replay is the closest
# available reference, not a matched dataset -- treat these as ballpark
# figures for the presentation, not ground truth.
PAPER_REFERENCE = {
    "dataset": "halfcheetah-medium-replay-v2",
    "dataset_caveat": (
        "closest available panel in the paper's Figure 17 -- the paper does not "
        "test halfcheetah-medium-v0 at all, only random/expert/medium-replay"
    ),
    "step_budget": 300_000,
    "values": {"wsrl": 79.0, "cql": 50.0, "iql": 37.0},
}

FILE_RE = re.compile(r"^(?P<algo>[a-z_]+)_seed(?P<seed>\d+)\.(?:parquet|csv)$")

STYLE = {
    "wsrl": dict(label="WSRL", color="#1b9e77", linestyle="-"),
    "iql": dict(label="IQL", color="#d95f02", linestyle="-"),
    "cql": dict(label="CQL", color="#7570b3", linestyle="-"),
    "iql_retain": dict(label="IQL (retains data)", color="#d95f02", linestyle="--"),
    "cql_retain": dict(label="CQL (retains data)", color="#7570b3", linestyle="--"),
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_run(path: Path):
    df = _read(path)
    cols = ["_step", RETURN_COL, NORM_RETURN_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}")

    df = df[cols].dropna()
    df = df[df["_step"] >= OFFLINE_STEPS].sort_values("_step")
    steps = df["_step"].to_numpy(dtype=float) - OFFLINE_STEPS
    returns = df[RETURN_COL].to_numpy(dtype=float)
    norm_returns = df[NORM_RETURN_COL].to_numpy(dtype=float)
    return steps, returns, norm_returns


def load_all_runs():
    runs = {}
    for path in sorted(DATA_DIR.glob("*")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        algo, seed = m.group("algo"), m.group("seed")
        runs.setdefault(algo, {})[seed] = _load_run(path)
    return runs


def aggregate(seed_runs, value_index, n_points=200):
    """Interpolate each seed's curve onto a common step grid, then mean/std."""
    max_step = min(steps.max() for steps, *_ in seed_runs.values())
    grid = np.linspace(0, max_step, n_points)
    interped = np.stack(
        [np.interp(grid, run[0], run[value_index]) for run in seed_runs.values()]
    )
    return grid, interped.mean(axis=0), interped.std(axis=0)


def plot_group(runs, algos, title, outfile, value_index=2, ylabel="Normalized Score"):
    missing = [algo for algo in algos if algo not in runs]
    if missing:
        print(f"skipping {outfile}: no data for {missing}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for algo in algos:
        grid, mean, std = aggregate(runs[algo], value_index)
        style = STYLE[algo]
        ax.plot(grid, mean, label=style["label"], color=style["color"],
                linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=style["color"], alpha=0.15)

    ax.set_xlabel("Online Fine-Tuning Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / outfile, dpi=200)
    plt.close(fig)
    print(f"saved {OUT_DIR / outfile}")


def build_return_table(runs):
    rows = []
    for algo, seed_runs in runs.items():
        for seed, (steps, returns, norm_returns) in sorted(seed_runs.items()):
            final_raw = returns[-FINAL_WINDOW:].mean()
            final_norm = norm_returns[-FINAL_WINDOW:].mean()
            rows.append(dict(algo=algo, seed=seed, final_return=final_raw,
                              final_normalized_return=final_norm))
    table = pd.DataFrame(rows).sort_values(["algo", "seed"])

    summary = (
        table.groupby("algo")[["final_return", "final_normalized_return"]]
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
    print("Per-algorithm summary (mean +/- std across seeds):")
    print(summary.to_string())


def build_paper_comparison(runs):
    ref = PAPER_REFERENCE["values"]
    rows = []
    our_max_step = 0.0
    for algo, paper_value in ref.items():
        if algo not in runs:
            continue
        norm_values = [nr[-FINAL_WINDOW:].mean() for _, _, nr in runs[algo].values()]
        our_max_step = max(our_max_step, max(steps.max() for steps, _, _ in runs[algo].values()))
        ours_mean = float(np.mean(norm_values))
        ours_std = float(np.std(norm_values))
        rows.append(dict(
            algo=algo,
            ours_normalized_return_mean=ours_mean,
            ours_normalized_return_std=ours_std,
            paper_normalized_return_approx=paper_value,
            delta=ours_mean - paper_value,
        ))
    table = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "paper_comparison.csv", index=False)
    print(f"saved {OUT_DIR / 'paper_comparison.csv'}")
    print()
    print(f"Ours (halfcheetah-medium-v0, {our_max_step:.0f} online steps) vs "
          f"paper (Figure 17, {PAPER_REFERENCE['dataset']}, {PAPER_REFERENCE['step_budget']} online steps):")
    print(f"Caveat: {PAPER_REFERENCE['dataset_caveat']}.")
    print(table.to_string(index=False))


def main():
    runs = load_all_runs()
    if not runs:
        print(f"no run files found in {DATA_DIR}")
        return

    # Figure 17 style: no-retention comparison, normalized return.
    plot_group(
        runs, algos=["wsrl", "iql", "cql"],
        title="Halfcheetah-mediumv0: WSRL vs IQL vs CQL (no retention)",
        outfile="fig17_halfcheetah_mediumv0.png",
        value_index=2, ylabel="Normalized Score",
    )

    # Same comparison in raw-return units.
    plot_group(
        runs, algos=["wsrl", "iql", "cql"],
        title="Halfcheetah-mediumv0: WSRL vs IQL vs CQL (no retention, raw)",
        outfile="fig17_halfcheetah_mediumv0_raw.png",
        value_index=1, ylabel="Average Return",
    )

    # Figure 9 style: WSRL (no retention) vs IQL/CQL retaining offline data.
    plot_group(
        runs, algos=["wsrl", "iql_retain", "cql_retain"],
        title="Halfcheetah-mediumv0: WSRL vs IQL/CQL (retains offline data)",
        outfile="fig9_halfcheetah_mediumv0.png",
        value_index=2, ylabel="Normalized Score",
    )

    build_return_table(runs)
    build_paper_comparison(runs)


if __name__ == "__main__":
    main()
