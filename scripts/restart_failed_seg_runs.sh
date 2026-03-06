#!/bin/bash
# Restarts failed segmentation dataset runs in tmux session seg_all.
EXP="/scratch3/merler/geo-bench-experiments/experiments/prithvi_segmentation/1.00x_train_prithvi_v2_seg_segmentation_v1.0_03-03-2026_15:52:49prithvi_eo_v2_100_tl"

FAILED=(m-NeonTree m-SA-crop-type m-cashew-plant m-chesapeake)

for dataset in "${FAILED[@]}"; do
    # Kill existing window with this name if present
    tmux kill-window -t "seg_all:${dataset}" 2>/dev/null || true
    tmux new-window -t seg_all -n "$dataset" \
        "bash \"${EXP}/${dataset}/seed_0/run.sh\" 2>&1 | tee \"${EXP}/${dataset}/seed_0/train.log\""
    echo "Restarted: $dataset"
done

echo ""
echo "Windows now running:"
tmux list-windows -t seg_all
