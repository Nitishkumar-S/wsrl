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

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


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
  --save_dir ./checkpoints/cql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

## WSRL

### pretraining

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-hopper-cql-pretrain-seed-0 \
  --project NR5 \
  --env mujoco/hopper/medium-v0 \
  --agent cql \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 0 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

```python
def get_config(updates=None):
    # config = sac_config.get_config()
    config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment. Similarly, switch to `iql_config.get_config()` when using an IQL-pretrained policy.

### finetuning

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-hopper-finetune-seed-0 \
  --project NR5 \
  --env mujoco/hopper/medium-v0 \
  --agent sac \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
  --resume_path path_to_checkpoint
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

**Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.


---

# Walker2d

## IQL Baseline

```
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
  --exp_name NR5-walker2d-iql-seed-0 \
  --project NR5 \
  --env mujoco/walker2d/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/iql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


## CQL Baseline

```
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
  --exp_name NR5-walker2d-cql-seed-0 \
  --project NR5 \
  --env mujoco/walker2d/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --utd 4 \
  --save_dir ./checkpoints/cql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

## WSRL

### pretraining

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-walker2d-cql-pretrain-seed-0 \
  --project NR5 \
  --env mujoco/walker2d/medium-v0 \
  --agent cql \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 0 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

```python
def get_config(updates=None):
    # config = sac_config.get_config()
    config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment. Similarly, switch to `iql_config.get_config()` when using an IQL-pretrained policy.

### finetuning

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-walker2d-finetune-seed-0 \
  --project NR5 \
  --env mujoco/walker2d/medium-v0 \
  --agent sac \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
  --resume_path path_to_checkpoint
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

**Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.


---

# HalfCheetha

## IQL Baseline

```
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
  --exp_name NR5-halfcheetah-iql-seed-0 \
  --project NR5 \
  --env mujoco/halfcheetah/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/iql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


## CQL Baseline

```
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
  --exp_name NR5-halfcheetah-cql-seed-0 \
  --project NR5 \
  --env mujoco/halfcheetah/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --utd 4 \
  --save_dir ./checkpoints/cql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

## WSRL

### pretraining

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-halfcheetah-cql-pretrain-seed-0 \
  --project NR5 \
  --env mujoco/walker2d/medium-v0 \
  --agent cql \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 0 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

```python
def get_config(updates=None):
    # config = sac_config.get_config()
    config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment. Similarly, switch to `iql_config.get_config()` when using an IQL-pretrained policy.

### finetuning

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-halfcheetah-finetune-seed-0 \
  --project NR5 \
  --env mujoco/halfcheetah/medium-v0 \
  --agent sac \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/wsrl
  --resume_path path_to_checkpoint
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

**Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.


---

# Humanoid

## IQL Baseline

```
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
  --exp_name NR5-humanoid-iql-seed-0 \
  --project NR5 \
  --env mujoco/humanoid/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/humanoid/iql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


## CQL Baseline

```
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
  --exp_name NR5-humanoid-cql-seed-0 \
  --project NR5 \
  --env mujoco/humanoid/medium-v0 \
  --seed 0 \
  --use_redq \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --utd 4 \
  --save_dir ./checkpoints/humanoid/cql
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

## WSRL

### pretraining

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-humanoid-cql-pretrain-seed-0 \
  --project NR5 \
  --env mujoco/humanoid/medium-v0 \
  --agent cql \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 0 \
  --save_interval 250000 \
  --save_dir ./checkpoints/humanoid/wsrl
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

```python
def get_config(updates=None):
    # config = sac_config.get_config()
    config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment. Similarly, switch to `iql_config.get_config()` when using an IQL-pretrained policy.

### finetuning

```
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-humanoid-finetune-seed-0 \
  --project NR5 \
  --env mujoco/humanoid/medium-v0 \
  --agent sac \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --save_dir ./checkpoints/humanoid/wsrl
  --resume_path path_to_checkpoint
```

**Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

**Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

**Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.


---