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
echo "Retention-baseline experiments started: $(date)"
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

python --version

# -------------------------
# IQL and CQL with offline data retention enabled.
# Same env/ensemble-Q/layer-norm setup as run_all.sh's no-retention
# runs, but with offline_data_ratio > 0 so each online batch mixes in
# offline transitions (online_sampling_method defaults to "mixed").
# Pairs with the existing no-retention IQL/CQL/WSRL runs to reproduce
# the "retains offline data" comparison (paper Fig. 9 / Fig. 17-style)
# using only WSRL, IQL, and CQL.
# -------------------------

for SEED in 1 2; do

    # IQL (retains offline data)
    bash experiments/scripts/locomotion/launch_iql_finetune.sh \
        --exp_name NR5retain \
        --env mujoco/halfcheetah/medium-v0 \
        --project NR5 \
        --seed "$SEED" \
        --save_interval 250000 \
        --save_dir /home/niladrimitra066/wsrl/checkpoints/iql_retain \
        --use_redq \
        --utd 4 \
        --offline_data_ratio 0.5

    # CQL (retains offline data)
    bash experiments/scripts/locomotion/launch_cql_finetune.sh \
        --exp_name NR5retain \
        --env mujoco/halfcheetah/medium-v0 \
        --project NR5 \
        --seed "$SEED" \
        --save_interval 250000 \
        --save_dir /home/niladrimitra066/wsrl/checkpoints/cql_retain \
        --use_redq \
        --utd 4 \
        --offline_data_ratio 0.5

done
