# Handoff — WSRL lab, Presentation 2 (29 July 2026)

Carry-over context from the chat that produced Presentation 1 (15 July) and the
humanoid/ablation work that followed. Written 29 July 2026, presentation is
**today**.

---

## 1. Project basics

- **Course:** Offline-to-Online RL lab, 6 weeks, groups of 3, $150 GCP credits.
- **Our paper: NR#5 — WSRL**, "Efficient Online RL Fine-Tuning Need Not Retain
  Offline Data" (Zhou et al., ICLR 2025).
- **Team:** Niladri Mitra (5977224), Nitishkumar Solpure (5975944),
  Kartik Nagraj Nayak (5975875).
- **wandb:** entity `fryan-nr`; project `NR5` (locomotion), `HUM1` (humanoid).
- **Repo:** branch `chetaahgoesmeow`, remote `Nitishkumar-S/wsrl`.
- **Reference docs in repo root** (both gitignored, but present locally):
  - `Copy of offline_to_online_rl_project_intro.pptx.md` — the assignment brief.
  - `Efficient Online Reinforcement Learning.md` — the WSRL paper as markdown.
    Note: figure *images* are stripped, only captions survive. Paper Figure 17
    (MuJoCo locomotion) had to be read from a screenshot.

## 2. What the assignment requires for 29 July

From the "Project Presentations" slide of the intro deck, verbatim scope:

- Summarize what you have done **since the last presentation**
- For the "Reproduction Set": explain your changes to the original code, and
  explain the differences between your results and the paper's
- If you have already touched the "Extension Set" (Minari Humanoid /
  HumanoidStandup) and "Advanced Option" (OGBench HumanoidMaze): explain that work

They explicitly say the deck does **not** need to be fancy — "efficient and easy
to follow".

Standing requirements that apply to every report/presentation (from the
"Evaluation rules" + "Practical success rules" slides):

- Final return table — **raw AND normalized** returns
- Learning curves with an **online-step x-axis**
- Seed info (≥3 seeds unless compute prevents it — we use 3)
- Compute budget
- A "what changed from the paper" table

## 3. Presentation 1 (15 July) — the pattern to follow

File: `Presentation 15_07.pptx` (repo root, gitignored). 13 slides:

| # | Slide |
|---|---|
| 1 | Title — paper name, 3 names + matriculation numbers, date |
| 2 | Learning Curves for HalfCheetah — raw + normalized PNGs side by side, summary table (mean/std per algo) |
| 3 | Comparison with Paper Result — paper's figure next to ours, delta table |
| 4 | Return Table per Seed for HalfCheetah |
| 5 | Learning Curves for Walker2D — + table incl. "Paper's Return (Approx)" |
| 6 | Learning Curves for Hopper |
| 7 | Comparison with Paper Result (Hopper) |
| 8 | Return Table per Seed for Hopper |
| 9 | Compute Budget — HalfCheetah + Hopper (GPU/CPU/RAM/disk, run time per seed per algo) |
| 10 | Compute Budget — Walker2D |
| 11 | Hyperparameters Used — 15-row table, ours vs paper inline |
| 12 | Next Steps |
| 13 | Thank You |

Pattern per environment: **learning curves → paper comparison → per-seed table**,
then compute budget and hyperparameters at the end. Reuse this for Humanoid.

### Presentation 1 numbers (already presented, for reference)

HalfCheetah-medium-v0, 500k online steps, 3 seeds — normalized return:

| Algo | Ours | Paper (approx, medium-replay) | Delta |
|---|---|---|---|
| wsrl | 84.02 ± 5.00 | 79.0 | +5.0 |
| cql  | 85.07 ± 3.10 | 50.0 | +35.1 |
| iql  | 52.65 ± 1.94 | 37.0 | +15.6 |

Hopper-medium-v0 (normalized): cql 104.38 ± 5.31, iql 55.30 ± 13.68,
wsrl 59.85 ± 17.86.
Walker2D (from the deck): cql 116.95 ± 6.87, iql 111.99 ± 5.47, wsrl 73.86 ± 11.10.

Archived artifacts for the HalfCheetah run live in `analysis/data/mine/` and
`analysis/figures/mine/`. The **top level** of `analysis/data/` currently holds
the **Hopper** parquets, and `plot_figures.py` is currently configured for Hopper.

## 4. Caveats established in Presentation 1 — still true, reuse them

These are the honest "differences from the paper" talking points. They were well
received and should carry forward:

1. **Dataset library differs.** We use **Minari** (`mujoco/halfcheetah/medium-v0`),
   the paper uses **D4RL** (`halfcheetah-medium-v2`). Different libraries,
   different versioning — `-v0` vs `-v2` is not a version bump of the same thing.
2. **Dataset variant differs.** Paper's Figure 17 has only `random`, `expert`,
   `medium-replay` panels — **no `medium` panel at all**. Our closest reference
   is `medium-replay`, so every paper comparison number is approximate and
   cross-variant.
3. **Step budget differs.** Paper plots 300k online steps; presentation-1 runs
   went to 500k.
4. **Paper numbers are hand-read off a figure**, not a published table. The paper
   gives no numeric table for MuJoCo locomotion.
5. **UTD mismatch (real config bug we found).** `launch_iql_finetune.sh` and
   `launch_cql_finetune.sh` never pass `--utd`, so they silently fall back to
   `finetune.py`'s default of **1**, while the paper uses **4 for all methods**
   and WSRL's launch script sets 4. `--use_redq` only adds the 10-Q ensemble +
   layer norm — it does **not** set UTD. Slide 11 of deck 1 reports this openly
   as "4 / 1(paper has 4) / 4".

### Abandoned thread — do not restart without a reason

We tried to reproduce the paper's exact D4RL `-v2` results. It requires
installing the `d4rl` fork + `mujoco-py` + `mujoco210`, and hits a real
gymnasium/legacy-gym incompatibility (`assert isinstance(env, gymnasium.Env)`
fails because `TruncationWrapper` is a classic-gym wrapper). We got as far as a
working dataset load, then **deliberately reverted** (commit `27fe71a`) because
the assignment mandates Minari, not D4RL. The cluster still has the packages
installed — harmless, purely additive, no need to uninstall.

## 5. What changed since 15 July

Three commits: `0cf47f5` (new changes), `a2f1041` (humanoid and runners),
`1e3332f` (normalised score update).

### 5.1 Humanoid extension (the "Extension Set" deliverable)

- **Env:** `mujoco/humanoid/medium-v0` (Minari). Also referenced:
  `simple-v0`, `expert-v0` for the data-quality ablation.
- **`wsrl/envs/env_common.py`** — `get_env_type()` now recognizes `humanoid` as
  `locomotion`.
- **`finetune.py`** — added a `humanoid` branch to
  `get_locomotion_normalized_score()`. No published Humanoid-v5/Minari reference
  pair exists, so it was **measured**: `random_score = 105.712692`,
  `expert_score = 8602.9` (random policy over 20 episodes; expert = mean return
  of `mujoco/humanoid/expert-v0` over 1197 episodes). Produced by
  `analysis/measure_humanoid_scores.py`.
- **`wsrl/envs/minari_dataset.py`** — **important correctness fix**:
  `_rescale_actions_to_agent_space()`. `make_gym_env()` wraps locomotion envs in
  `RescaleAction(env, -0.99999, +0.99999)`, so online the agent emits ~[-1,1],
  but Minari stores *raw* env actions. Invisible on halfcheetah/hopper/walker
  (already `Box(-1,1)`) but **Humanoid-v5 is `Box(-0.4, 0.4)`** — without this,
  offline data and online rollouts live on different scales and the policy's
  outputs would be shrunk 2.5x the moment online fine-tuning starts. Good
  slide material: a humanoid-specific bug that simple locomotion never exposes.
- **`experiments/configs/train_config.py`** — added `locomotion_{cql,iql,wsrl}_wide`
  configs with `[512, 512]` hidden dims (vs `[256, 256]`), motivated by
  Humanoid's 348-dim observation vs halfcheetah's 17.
- **`finetune.py`** — new `--algo_name` flag: overrides the label used for the
  wandb run name and checkpoint dir, so a run can use `--agent cql` while still
  being labelled `wsrl`.

### 5.2 WSRL is now launched as the CQL agent

`run_all.sh` and `run_humanoid.sh` both launch WSRL as
`--agent cql --algo_name wsrl --online_use_cql_loss=True`. The
`locomotion_wsrl` config is CQL-shaped so the `cql_*` agent_kwargs are present.
Do **not** re-pass `--config` when doing this. absl takes the *last* occurrence
of a repeated flag, which is what makes the override work.

### 5.3 Runner scripts

- **`run_all.sh`** (tracked) — now `mujoco/halfcheetah/expert-v0`, 300k online
  steps, seeds 0/1/2. WSRL via the `--agent cql` trick. **IQL at `--utd 1`,
  CQL at `--utd 4`** (deliberately left mismatched; noted in deck 1).
- **`run_humanoid.sh`** (⚠️ **gitignored, local only**) — 4-phase humanoid suite,
  ~380 lines, with concurrency control (`MAX_JOBS`), per-arm logging, failure
  collection, `DRY_RUN=1`, and checkpoint-resume to keep the suite on one GPU.
  - `phase0` — headline WSRL/IQL/CQL, 3 seeds, **all three at UTD 4** (unlike
    `run_all.sh`). Must finish first: leaves the 250k checkpoints others resume from.
  - `phase1` — A1 warmup steps (0 / 20000), A2 retention (0.25 / 0.5),
    A6 online CQL loss on/off. 2 seeds.
  - `phase2` — A4 UTD (1 / 8), A9 batch size (1024), A5 no-REDQ-ensemble
    (full 550k runs, can't resume — arch changes; WSRL excluded because its
    ensemble comes from the config, not `--use_redq`).
  - `phase3` — A7 pretraining budget (0 / 100k), A8 data quality
    (simple-v0 / expert-v0). Most expensive, no resume. "Cut this first."
  - Key constraint baked in: `replay_buffer_capacity=320000`, because the
    inherited default of 2e6 × 348-dim × float64 × 2 ≈ **11 GB host RAM** on
    Humanoid (invisible on halfcheetah's 17 dims).
- **`run_ablation_warmup.sh`** (⚠️ **gitignored, local only**) — standalone A1
  warmup sweep (`wsrl_nowarmup`=0, `wsrl_longwarmup`=20000), resumes from
  phase0's 250k checkpoints.
- **`run_retention_baselines.sh`** (tracked) — older halfcheetah retention runs
  (`--offline_data_ratio 0.5`), for the paper's Figure-9-style comparison. Never
  produced data as of deck 1; `plot_figures.py` skips `fig9_*` cleanly when the
  `iql_retain`/`cql_retain` files are absent.

⚠️ **Because the two humanoid runners are gitignored, `git pull` on the cluster
will NOT deliver them.** They must be `scp`'d or pasted manually.

⚠️ **Arm-label constraint:** `plot_figures.py` matches `^([a-z_]+)_seed(\d+)$`,
so `--algo_name` labels must be lowercase letters/underscores only. Any digit
silently breaks the export match. This is why labels are `wsrl_nowarmup` /
`wsrl_longwarmup`, not `wsrl_warmup0` / `wsrl_warmup20000`.

## 6. Analysis workflow

`analysis/plot_figures.py` is the single entry point. It reads **one file per
run** from `analysis/data/`, named `<algo>_seed<N>.parquet` (or `.csv`).

Required columns: `_step`, `evaluation/average_return`,
`evaluation/average_normalized_return`. A plain wandb run-history export has
these; the 32 training curves and 4 timer graphs are not needed.

What it does:
- Drops everything before `OFFLINE_STEPS = 250_000` and shifts x to 0, so the
  x-axis is **online fine-tuning steps** (matches the paper's convention).
- Interpolates each seed onto a common grid, plots mean ± std shading.
- Emits `fig17_*.png` (normalized), `fig17_*_raw.png` (raw), `fig9_*.png`
  (retention comparison — skipped with a printed note if data absent).
- Writes `final_returns_per_seed.csv`, `final_returns_summary.csv`,
  `paper_comparison.csv`. "Final" = mean of the last `FINAL_WINDOW = 5` eval
  points, to denoise.
- `PAPER_REFERENCE` holds the hand-read paper values and the caveat string.

Run it with:

```bash
python analysis/plot_figures.py
```

**Local Python works fine for analysis** (pandas/matplotlib/pyarrow all present
on the Windows box) — figures and tables were generated locally throughout. Only
*training* has to happen on the cluster. (The `no-local-python-env` memory note
is overly broad on this point.)

Currently `plot_figures.py` is hardcoded for **Hopper** (titles, output names,
`PAPER_REFERENCE`). For Humanoid it needs those updated — and note the paper has
**no Humanoid panel at all**, so `PAPER_REFERENCE` should be dropped or clearly
marked N/A rather than compared against a locomotion number.

## 7. Repo state (as of this handoff)

- Branch `chetaahgoesmeow`, working tree **clean**, HEAD = `1e3332f`.
- `analysis/data/*.parquet` — Hopper runs, 3 seeds × wsrl/iql/cql, `_step` to 750k
  (= 250k offline + 500k online).
- `analysis/data/mine/` + `analysis/figures/mine/` — archived HalfCheetah
  (presentation 1).
- `analysis/figures/` — current Hopper figures + CSVs.
- **No Humanoid parquets exist locally yet.**
- `*.png` is gitignored, so figures are never committed — only the CSVs are.

## 8. Open questions for the new chat — ask these first

1. **Did the humanoid runs actually complete on the cluster?** No humanoid
   parquets are in `analysis/data/` yet. Which phases finished — phase0 only, or
   any of phase1/2/3? This determines whether Presentation 2 shows a headline
   humanoid comparison, ablations, or just "in progress".
2. **Were the Walker2D/Hopper reproduction runs re-run** at the corrected 300k
   budget, or are the deck-1 numbers still current?
3. **Compute budget figures for the humanoid runs** — GPU type, CPU, RAM,
   wall-clock per seed per algo. Deck 1 has a dedicated slide per environment and
   the assignment explicitly requires budget tracking. The humanoid README notes
   an A6000; confirm.
4. **OGBench HumanoidMaze (Advanced Option)** — touched at all? Deck 1 didn't
   cover it. If untouched, say so explicitly on the "Next Steps" slide.
5. Should the archived HalfCheetah results be **re-shown** in deck 2, or only
   referenced as "presented last time"? The assignment says summarize what's been
   done *since* the last presentation.

## 9. Suggested Presentation 2 outline (adapt once Q1–Q5 are answered)

1. Title — same format, date 29 July 2026
2. Recap / what we've done since 15 July (one slide, bulleted)
3. **Humanoid: setup & what it took** — Minari `mujoco/humanoid/medium-v0`,
   measured normalization pair, the action-rescaling bug, the 11 GB replay-buffer
   trap, the wide-network config. This is the strongest new material.
4. Humanoid learning curves (raw + normalized) + summary table
5. Humanoid per-seed return table
6. Ablations — whichever phases completed. A1 warmup is the paper's headline
   mechanism and the most on-point; A2 retention directly tests the paper's core
   claim.
7. Differences from the paper — reuse §4 caveats + "paper has no Humanoid panel,
   so no reference number exists for this environment"
8. Compute budget — humanoid
9. Hyperparameters — same 15-row format, extended with humanoid-specific rows
10. Next steps / what's left
11. Thank you

## 10. Working preferences carried over from the last chat

- Plots should show **only our own data** — a paper-reference overlay (✕ markers
  + step-budget line) was built and then explicitly rejected. Keep paper
  comparisons in tables, not on the curves.
- Verify claims against the actual code before asserting them (several
  conclusions in the last chat came from tracing `run_all.sh` → launch scripts →
  `train_config.py` → base configs, and that's how the UTD bug surfaced).
- Don't run training locally. Edit here, run on the cluster.
- Ask before large speculative detours (the D4RL thread was correctly abandoned
  when it drifted out of scope).
