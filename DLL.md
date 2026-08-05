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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.


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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified.

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** Before running WSRL with a CQL-pretrained policy, edit `experiments/configs/wsrl_config.py` and change the base configuration from SAC to CQL:

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

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (e.g., `0`, `1`, or `2`) to ensure each run is uniquely identified. 

> **Note:** The `--resume_path` argument should point to the checkpoint generated during the offline pretraining stage.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.


---

# Ablation on Humanoid

## Ablation Study: Warmup Steps

The following experiments evaluate the effect of varying the number of warmup steps used during WSRL fine-tuning.

Warmup values evaluated:

- **1,000** steps
- **5,000** steps
- **20,000** steps

Each experiment was run with **three random seeds (0, 1, and 2)**.

### Example Command (1,000 Warmup Steps)

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-humanoid-wsrl-ablation-1kWarmup-finetune-seed-0 \
  --project NR5 \
  --agent sac \
  --env mujoco/humanoid/medium-v0 \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --warmup_steps 1000 \
  --save_dir ./checkpoints/humanoid/wsrl \
  --resume_path path_to_checkpoint
```

> **Note:** Replace `--warmup_steps` with `5000` or `20000` to reproduce the remaining warmup ablation experiments.

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (`0`, `1`, or `2`).

> **Note:** Update `--resume_path` to point to the corresponding pretrained checkpoint for the selected environment and random seed.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.

---

## Ablation Study: Warmup Buffer Initialisation

This ablation evaluates the effect of using different policies before WSRL fine-tuning.

Each experiment was run with **three random seeds (0, 1, and 2)**.

### Example Command

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-humanoid-wsrl-ablation-RandomWarmup-finetune-seed-0 \
  --project NR5 \
  --agent sac \
  --env mujoco/humanoid/medium-v0 \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --random_warmup True \
  --save_dir ./checkpoints/humanoid/wsrl \
  --resume_path path_to_checkpoint
```

> **Note:** Setting `--random_warmup True` performs the warmup phase using a **random policy** instead of the pretrained policy.

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (`0`, `1`, or `2`).

> **Note:** Update `--resume_path` to point to the corresponding pretrained checkpoint for the selected environment and random seed.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.

---

## Ablation Study: Value Initialisation

This ablation evaluates the effect of initializing WSRL from different offline pretrained agents. The following pretrained policies were considered:

- **IQL**
- **CQL**

For each pretrained agent, the experiment consists of two stages:

1. **Offline pretraining**
2. **WSRL fine-tuning**

Each experiment was run with **three random seeds (0, 1, and 2)**.

---

### IQL Pretraining

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-iql-pretraining-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent iql \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 0 \
    --utd 1 \
    --save_dir ./checkpoints/humanoid/wsrl
```

### IQL Fine-tuning

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-iql-finetuning-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent sac \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 500000 \
    --save_dir ./checkpoints/humanoid/wsrl \
    --resume_path path_to_checkpoint
```

> **Note:** Before running the pre-training stage, edit `experiments/configs/wsrl_config.py` and set the base configuration to `iql_config.get_config()`.

---

### CQL Pretraining

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-cql-pretraining-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent cql \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 0 \
    --save_dir ./checkpoints/humanoid/wsrl
```

### CQL Fine-tuning

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-cql-finetuning-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent sac \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 500000 \
    --save_dir ./checkpoints/humanoid/wsrl \
    --resume_path path_to_cql_pretrained_checkpoint
```

> **Note:** Before running the pre-training stage, edit `experiments/configs/wsrl_config.py` and set the base configuration to  `cql_config.get_config()`.

---

### Notes

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (`0`, `1`, or `2`).

> **Note:** Update `--resume_path` to point to the pretrained checkpoint generated during the corresponding offline pretraining stage.

> **Note:** During offline pretraining, set the `--agent` argument to the desired offline RL algorithm (e.g., `iql` or `cql`). During WSRL fine-tuning, always use `--agent sac`, as WSRL performs online fine-tuning using the SAC agent initialized from the pretrained checkpoint.

> **Note:** Ensure that the active configuration in `experiments/configs/wsrl_config.py` matches the agent used (SAC, IQL, or CQL).
```python
def get_config(updates=None):
    # config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```


---

# Humanoid Specific Ablations


## Ablation Study: Retaining Offline Data During Fine-tuning

This ablation investigates the effect of retaining a fraction of the offline dataset during the online fine-tuning stage. While the original WSRL algorithm discards the offline dataset after pretraining (`offline_data_ratio = 0`), this study evaluates whether retaining some offline data can improve performance on the challenging **Humanoid** environment, which has a high-dimensional state and action space due to its many degrees of freedom (DoF).

The following offline data retention ratios were evaluated:

- **0.00** (original WSRL; no offline data retained)
- **0.25**
- **0.50**
- **0.75**

Each experiment was run with **three random seeds (0, 1, and 2)**.

### Example Command (25% Offline Data Retained)

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
  --exp_name NR5-humanoid-wsrl-ablation-retain25-finetune-seed-0 \
  --agent sac \
  --env mujoco/humanoid/medium-v0 \
  --seed 0 \
  --num_offline_steps 250000 \
  --num_online_steps 500000 \
  --save_interval 250000 \
  --offline_data_ratio 0.25 \
  --save_dir ./checkpoints/humanoid/wsrl \
  --resume_path path_to_checkpoint
```

> **Note:** Set `--offline_data_ratio` to `0`, `0.25`, `0.5`, or `0.75` to reproduce the different offline data retention ablation experiments.

> **Note:** An `offline_data_ratio` of `0` corresponds to the original WSRL algorithm, where the offline dataset is completely discarded after offline pretraining.

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (`0`, `1`, or `2`).

> **Note:** Update `--resume_path` to point to the corresponding pretrained checkpoint for the selected environment and random seed.

> **Note:** Before finetuning WSRL, edit `experiments/configs/wsrl_config.py` and change the base configuration to SAC:

```python
def get_config(updates=None):
    config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```

Save the file before launching the experiment.

---

## Ablation Study: Reward Configuration

This ablation investigates how modifying the reward function during both offline pretraining and WSRL fine-tuning affects performance on the **Humanoid** environment. The reward is manipulated using the `--reward_scale` and `--reward_bias` arguments to emphasize either the forward locomotion reward or the alive reward.

For each reward setting, experiments consist of two stages:

1. **Offline pretraining**
2. **WSRL fine-tuning**

Each experiment was run with **three random seeds (0, 1, and 2)**.

### Reward Configurations

| Reward Scale | Reward Bias | Description |
|--------------|------------:|-------------|
| **1.0** | **0.0** | Default reward (no modification). |
| **1.0** | **-5.0** | Primarily forward locomotion reward with the alive reward effectively removed. |
| **0.0** | **5.0** | Alive reward only; forward locomotion reward is removed. |
| **1.0** | **20.0** | Places greater emphasis on staying alive than moving forward. |
| **5.0** | **-20.0** | Places greater emphasis on forward locomotion than staying alive. |

### Example: Offline Pretraining

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-reward-scale1-bias-5-pretraining-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent cql \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 0 \
    --reward_scale 1.0 \
    --reward_bias -5.0 \
    --save_dir ./checkpoints/humanoid/wsrl
```

### Example: WSRL Fine-tuning

```bash
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5-humanoid-wsrl-ablation-reward-scale1-bias-5-finetuning-seed-0 \
    --env mujoco/humanoid/medium-v0 \
    --agent sac \
    --seed 0 \
    --num_offline_steps 250000 \
    --num_online_steps 500000 \
    --reward_scale 1.0 \
    --reward_bias -5.0 \
    --save_dir ./checkpoints/humanoid/wsrl \
    --resume_path path_to_pretrained_checkpoint
```

### Notes

> **Note:** Replace `--reward_scale` and `--reward_bias` with the desired reward configuration listed above to reproduce the corresponding ablation experiment.

> **Note:** Replace the values of `--seed` and `--exp_name` appropriately when running experiments with different random seeds (`0`, `1`, or `2`).

> **Note:** Update `--resume_path` to point to the checkpoint generated during the corresponding offline pretraining stage.

> **Note:** During offline pretraining, use `--agent cql`. During WSRL fine-tuning, use `--agent sac`, initializing from the pretrained checkpoint.

> **Note:** Ensure that the active configuration in `experiments/configs/wsrl_config.py` matches the agent used (SAC or CQL).
```python
def get_config(updates=None):
    # config = sac_config.get_config()
    # config = cql_config.get_config()
    # config = iql_config.get_config()
```
