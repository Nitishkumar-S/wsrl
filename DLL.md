# Running Experiments

This document describes how to reproduce all experiments in this repository using the Minari datasets.

---

# 1. Environment Setup

Create a Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip.

```bash
pip install --upgrade pip
```

Install the project dependencies.

```bash
pip install -r requirements.txt
```

Remove any existing JAX installation.

```bash
pip uninstall -y jax jaxlib nvidia-cudnn-cu12 nvidia-cublas-cu12
```

Install the CUDA 12 compatible JAX version.

```bash
pip install "jax[cuda12_pip]==0.4.23" "nvidia-cudnn-cu12<9.0" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify the installation.

```bash
python -c "import jax; print(jax.devices())"
```

---

# 2. Running Experiments

The experiments are launched using the shell scripts under:

```text
experiments/scripts/locomotion/
```

Each command trains one model for a single random seed.

The checkpoint directory can be changed using:

```text
--save_dir
```

The experiment name can be changed using:

```text
--exp_name
```

---

# Hopper

## IQL Baseline

```
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
  --exp_name NR5-hopper-iql-seed-0 \
  --project NR5 \
  --env mujoco/hopper/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/iql
```

replace --seed and --exp_name appropriately to run on different seeds.


## CQL Baseline

```
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
  --exp_name NR5-hopper-cql-seed-0 \
  --project NR5 \
  --env mujoco/hopper/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --utd 4 \
  --save_dir ./checkpoints/iql
```

replace --seed and --exp_name appropriately to run on different seeds.

