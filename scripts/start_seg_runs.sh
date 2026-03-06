#!/bin/bash
# Launches all segmentation dataset runs in tmux session seg_all (one window per dataset).
EXP="/scratch3/merler/geo-bench-experiments/experiments/prithvi_segmentation/1.00x_train_prithvi_v2_seg_segmentation_v1.0_03-04-2026_14:37:22prithvi_eo_v2_100_tl"

DATASETS=(m-NeonTree m-SA-crop-type m-cashew-plant m-chesapeake m-nz-cattle m-pv4ger-seg)

# Kill existing session if present
tmux kill-session -t seg_all 2>/dev/null || true

# Create session with first dataset
tmux new-session -d -s seg_all -n "${DATASETS[0]}" \
    "bash \"${EXP}/${DATASETS[0]}/seed_0/run.sh\" 2>&1 | tee \"${EXP}/${DATASETS[0]}/seed_0/train.log\""

# Add remaining datasets as new windows
for dataset in "${DATASETS[@]:1}"; do
    tmux new-window -t seg_all -n "$dataset" \
        "bash \"${EXP}/${dataset}/seed_0/run.sh\" 2>&1 | tee \"${EXP}/${dataset}/seed_0/train.log\""
done

echo "Started tmux session 'seg_all' with the following windows:"
tmux list-windows -t seg_all
echo ""
echo "Attach with: tmux attach -t seg_all"
echo "Switch windows: Ctrl-b <number>"
