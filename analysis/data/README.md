# wandb CSV exports go here

For each run, open it in the wandb UI (project `NR5`, entity `fryan-nr`),
find the `evaluation/average_normalized_return` chart (use
`evaluation/average_return` if normalized isn't available), click the
chart's `...` menu -> **Download data** -> CSV, and save it here named:

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

Each file just needs a step column and the one metric column — no need
to clean it up further, `plot_figures.py` handles the rest.

Then run:

```
python analysis/plot_figures.py
```
