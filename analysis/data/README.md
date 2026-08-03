# wandb CSV exports go here

Preferred: download the run's full history as parquet.

For runs where that isn't available, open the run in the wandb UI (project
`NR5`, entity `fryan-nr`), find the **`evaluation/average_return`** chart
(the raw one, not normalized -- see below), click the chart's `...` menu ->
**Download data** -> CSV, and save it here named:

```
wsrl_seed1.csv
wsrl_seed2.csv
iql_seed1.csv
iql_seed2.csv
cql_seed1.csv
cql_seed2.csv
iql_retain_seed1.csv
iql_retain_seed2.csv
cql_retain_seed1.csv
cql_retain_seed2.csv
```

- `wsrl`/`iql`/`cql` = your existing `run_all.sh` runs (no data retention).
- `iql_retain`/`cql_retain` = the new runs from `run_retention_baselines.sh`
  (`--offline_data_ratio 0.5`).

Parquet and CSV can be mixed freely, including across seeds of the same
experiment (`wsrl_seed0.parquet` + `wsrl_seed1.csv` is fine).

Each file just needs a step column and at least one metric column — no need
to clean it up further, `plot_figures.py` handles the rest: it accepts either
`_step` or `Step`, strips the `"{run name} - "` prefix and the `__MIN`/`__MAX`
band columns that the chart CSV adds, and fills in whichever metric is missing.

**Download the raw `average_return` chart, not the normalized one.** Raw return
never changes, so `plot_figures.py` can normalize it locally with one consistent
reference pair (`REF_SCORES`, mirroring `get_locomotion_normalized_score()` in
`finetune.py`). A normalized export is frozen against whatever constants were
live when that run executed, which silently desyncs old and new runs on the same
figure if those constants are ever corrected — this matters for Humanoid, whose
pair was measured locally rather than published. Normalized-only CSVs still work
(the transform is linear, so they're inverted back to raw exactly), it's just the
less robust input. Note `plot_figures.py` infers the env from the **file name**,
so include e.g. `halfcheetah` or `humanoid` in it, or set `DEFAULT_ENV`.

Then run:

```
python analysis/plot_figures.py
```
