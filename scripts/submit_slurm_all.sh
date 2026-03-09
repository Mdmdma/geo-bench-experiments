#!/bin/bash
# Load eth proxy to give the instance access to the internet.
# This is only needed for logging to wandb in an online run
module load eth_proxy 
# Submit all classification and segmentation benchmark runs to SLURM via sbatch.
#
# Usage:
#   bash scripts/submit_slurm_all.sh [tiny|100m]
#
# Model sizes:
#   tiny  - prithvi_eo_v2_tiny_tl  (default, for debugging / fast iteration)
#   100m  - prithvi_eo_v2_100_tl   (for real evaluation)

set -euo pipefail

# ─── Model selection ──────────────────────────────────────────────────────────
MODEL="${1:-tiny}"   # Change to "100m" for full evaluation

case "$MODEL" in
  tiny)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_tiny_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_tiny_all_bands.yaml"
    ;;
  100m)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_100_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_100_tl.yaml"
    ;;
  *)
    echo "ERROR: Unknown model '$MODEL'. Use 'tiny' or '100m'."
    exit 1
    ;;
esac

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO="/scratch3/merler/geo-bench-experiments"
export GEO_BENCH_DIR="/scratch3/merler/geo-bench/datasets"

CLS_TASK_CONFIG="$REPO/geobench_exp/configs/prithvi_cls_all_task.yaml"
SEG_TASK_CONFIG="$REPO/geobench_exp/configs/prithvi_seg_all_task.yaml"

# ─── SLURM settings ───────────────────────────────────────────────────────────
SBATCH_ARGS=(
  --account=es_schin
  --time=06:00:00
  --ntasks=1
  --cpus-per-task=10
  --mem-per-cpu=2G
  --gpus=1
  --gres=gpumem:11G
)

# ─── Generate experiment job directories ──────────────────────────────────────
echo ">>> Generating classification experiments (model=$MODEL)..."
cd "$REPO"
uv run geobench_exp-gen_exp \
  --task_config_path "$CLS_TASK_CONFIG" \
  --model_config_path "$CLS_MODEL_CONFIG"

echo ">>> Generating segmentation experiments (model=$MODEL)..."
uv run geobench_exp-gen_exp \
  --task_config_path "$SEG_TASK_CONFIG" \
  --model_config_path "$SEG_MODEL_CONFIG"

# ─── Helper: submit all seed_0/run.sh jobs under an experiment dir ─────────────
submit_jobs() {
  local exp_root="$1"
  local label="$2"

  # Find the most recently created experiment directory under exp_root
  local exp_dir
  exp_dir=$(ls -td "$exp_root"/*/  2>/dev/null | head -1)
  exp_dir="${exp_dir%/}"

  if [[ -z "$exp_dir" ]]; then
    echo "ERROR: No experiment directories found under $exp_root"
    return 1
  fi

  echo ">>> Submitting $label jobs from: $exp_dir"

  for run_sh in "$exp_dir"/*/seed_0/run.sh; do
    if [[ ! -f "$run_sh" ]]; then
      continue
    fi
    dataset=$(basename "$(dirname "$(dirname "$run_sh")")")
    log_file="$(dirname "$run_sh")/slurm_%j.log"

    job_id=$(sbatch \
      "${SBATCH_ARGS[@]}" \
      --job-name="geobench_${dataset}" \
      --output="$log_file" \
      --wrap="bash $run_sh" \
      | awk '{print $NF}')

    echo "  Submitted $dataset → job $job_id  (log: $(dirname "$run_sh")/slurm_${job_id}.log)"
  done
}

# ─── Submit ───────────────────────────────────────────────────────────────────
submit_jobs "$REPO/experiments/prithvi_cls_all" "classification"
submit_jobs "$REPO/experiments/prithvi_seg_all" "segmentation"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R'"
