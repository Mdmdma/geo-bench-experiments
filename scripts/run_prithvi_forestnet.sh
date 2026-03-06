#!/usr/bin/env bash
# Run a Prithvi-EO v2 experiment on m-forestnet in one call.
# Usage: bash scripts/run_prithvi_forestnet.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export GEO_BENCH_DIR=/scratch3/merler/geo-bench/datasets

TASK_CONFIG=geobench_exp/configs/prithvi_forestnet_task.yaml
MODEL_CONFIG=geobench_exp/configs/model_configs/classification/prithvi_v2_100_all_bands.yaml
TMUX_SESSION=forestnet_prithvi

# ── 1. Ensure required packages are present ───────────────────────────────────
echo "==> Checking / installing dependencies..."
uv pip install --quiet kornia lightning torchgeo omegaconf hydra-core terratorch

# ── 2. Generate the job directory ─────────────────────────────────────────────
echo "==> Generating experiment..."
uv run geobench_exp-gen_exp \
    --task_config_path "$TASK_CONFIG" \
    --model_config_path "$MODEL_CONFIG"

# Locate the most-recently generated job dir
JOB_DIR=$(ls -dt "$REPO_DIR/experiments/prithvi_forestnet"/*/m-forestnet/seed_0 | head -1)
echo "==> Job dir: $JOB_DIR"

# ── 3. Launch in a (new or replaced) tmux session ────────────────────────────
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" \
    "bash $JOB_DIR/run.sh 2>&1 | tee $JOB_DIR/log.out; echo \"EXIT:\$?\" >> $JOB_DIR/log.out"

echo "==> Training started in tmux session '$TMUX_SESSION'."
echo "    Attach : tmux attach -t $TMUX_SESSION"
echo "    Logs   : tail -f $JOB_DIR/log.out"
