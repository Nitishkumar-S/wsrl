#!/bin/bash

set -euo pipefail

# -------------------------
# Logging
# -------------------------
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOGFILE")
exec 2>&1

echo "========================================"
echo "Experiment started: $(date)"
echo "Logging to: $LOGFILE"
echo "========================================"

# -------------------------
# Shutdown on exit
# -------------------------
cleanup() {
    status=$?

    echo ""
    echo "========================================"
    echo "Finished at: $(date)"
    echo "Exit status: $status"
    echo "Syncing filesystem..."
    sync

    echo "Shutting down VM..."
    sudo shutdown -h now
}

trap cleanup EXIT

# -------------------------
# Activate virtual environment
# -------------------------
source /home/niladrimitra066/venv/bin/activate

# Optional: verify Python
python --version

# -------------------------
# Run experiments
# -------------------------
# halfcheetah/expert-v0, 300k online steps, seeds 0/1/2.
# CQL now get --utd 4 explicitly: launch_iql_finetune.sh /
# launch_cql_finetune.sh never set --utd, so without this they silently
# fall back to finetune.py's default of 1, while WSRL (and the paper's
# baselines) run at UTD=4. --use_redq alone only adds the 10-Q ensemble
# + layer norm, it does not set UTD.
#
# WSRL overrides launch_wsrl_finetune.sh's --agent sac with --agent cql below
# (absl takes the last occurrence of a repeated flag). --algo_name wsrl keeps
# the wandb run name and checkpoint dir reading "wsrl" instead of the agent
# name. The locomotion_wsrl config in experiments/configs/train_config.py is
# CQL-shaped so the cql_* agent_kwargs are present -- do not re-pass --config
# here. --online_use_cql_loss=True is finetune.py's default, passed explicitly
# because keeping the CQL regularizer on during the online phase is deliberate.

# --- seed 0 ---

# 1. WSRL
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 0 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/wsrl \
    --batch_size 256 \
    --agent cql \
    --algo_name wsrl \
    --online_use_cql_loss=True

# 2. IQL
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 0 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/iql \
    --use_redq \
    --utd 1

# 3. CQL
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 0 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/cql \
    --use_redq \
    --utd 4

# --- seed 1 ---

# 1. WSRL
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 1 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/wsrl \
    --batch_size 256 \
    --agent cql \
    --algo_name wsrl \
    --online_use_cql_loss=True

# 2. IQL
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 1 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/iql \
    --use_redq \
    --utd 1

# 3. CQL
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 1 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/cql \
    --use_redq \
    --utd 4

# --- seed 2 ---

# 1. WSRL
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 2 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/wsrl \
    --batch_size 256 \
    --agent cql \
    --algo_name wsrl \
    --online_use_cql_loss=True

# 2. IQL
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 2 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/iql \
    --use_redq \
    --utd 1

# 3. CQL
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/expert-v0 \
    --project NR5 \
    --seed 2 \
    --num_online_steps 300000 \
    --save_interval 250000 \
    --save_dir /home/niladrimitra066/wsrl/checkpoints/cql \
    --use_redq \
    --utd 4
