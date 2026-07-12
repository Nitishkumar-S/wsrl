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
# Activate Conda
# -------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wsrl

# -------------------------
# Run experiments
# -------------------------

# 1. WSRL
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/medium-v0 \
    --project NR5 \
    --seed 0 \
    --save_interval 250000 \
    --save_dir ./checkpoints/wsrl \
    --batch_size 256

# 2. IQL
bash experiments/scripts/locomotion/launch_iql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/medium-v0 \
    --project NR5 \
    --seed 0 \
    --save_interval 250000 \
    --save_dir ./checkpoints/iql \
    --use_redq

# 3. CQL
bash experiments/scripts/locomotion/launch_cql_finetune.sh \
    --exp_name NR5 \
    --env mujoco/halfcheetah/medium-v0 \
    --project NR5 \
    --seed 0 \
    --save_interval 250000 \
    --save_dir ./checkpoints/cql \
    --use_redq