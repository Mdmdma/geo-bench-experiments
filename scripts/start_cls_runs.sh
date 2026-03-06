#!/bin/bash
# Launches all classification dataset runs in tmux session cls_all (one window per dataset).
# Usage:
#   bash scripts/start_cls_runs.sh <experiment_dir>
# Or set EXP directly in this file.

EXP="${1:-}"

if [[ -z "$EXP" ]]; then
    echo "ERROR: No experiment directory specified."
    echo "Usage: bash scripts/start_cls_runs.sh <path/to/experiment_dir>"
    exit 1
fi

if [[ ! -d "$EXP" ]]; then
    echo "ERROR: Experiment directory does not exist: $EXP"
    exit 1
fi

DATASETS=(m-brick-kiln m-bigearthnet m-eurosat m-so2sat m-pv4ger m-forestnet)

# Filter to only datasets that actually exist in the experiment dir
AVAILABLE=()
for dataset in "${DATASETS[@]}"; do
    if [[ -d "$EXP/$dataset/seed_0" ]]; then
        AVAILABLE+=("$dataset")
    else
        echo "WARNING: Skipping $dataset (not found in $EXP)"
    fi
done

if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
    echo "ERROR: No matching dataset directories found under $EXP"
    exit 1
fi

# Kill existing session if present
tmux kill-session -t cls_all 2>/dev/null || true

# Create session with first dataset
tmux new-session -d -s cls_all -n "${AVAILABLE[0]}" \
    "bash \"${EXP}/${AVAILABLE[0]}/seed_0/run.sh\" 2>&1 | tee \"${EXP}/${AVAILABLE[0]}/seed_0/train.log\""

# Add remaining datasets as new windows
for dataset in "${AVAILABLE[@]:1}"; do
    tmux new-window -t cls_all -n "$dataset" \
        "bash \"${EXP}/${dataset}/seed_0/run.sh\" 2>&1 | tee \"${EXP}/${dataset}/seed_0/train.log\""
done

echo "Started tmux session 'cls_all' with the following windows:"
tmux list-windows -t cls_all
echo ""
echo "Attach with: tmux attach -t cls_all"
echo "Switch windows: Ctrl-b <number>"
