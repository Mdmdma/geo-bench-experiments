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
#   300m  - prithvi_eo_v2_300_tl   (300M model, batch_size=1, 12h)
#   600m  - prithvi_eo_v2_600_tl   (largest model, batch_size=1, 12h)

set -euo pipefail

# ─── Data directory ───────────────────────────────────────────────────────────
# Set this to the root folder containing classification_v1.0/ and segmentation_v1.0/
DATA_DIR="/cluster/scratch/merler/geobench"

# ─── Model selection ──────────────────────────────────────────────────────────
MODEL="${1:-tiny}"   # Change to "100m" for full evaluation

case "$MODEL" in
  tiny)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_tiny_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_tiny_all_bands.yaml"
    GPU_MEM="11G"
    JOB_TIME="06:00:00"
    CLS_BATCH_SIZE=32
    SEG_BATCH_SIZE=8
    ;;
  100m)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_100_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_100_tl.yaml"
    GPU_MEM="11G"
    JOB_TIME="06:00:00"
    CLS_BATCH_SIZE=32
    SEG_BATCH_SIZE=8
    ;;
  300m)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_300_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_300_tl.yaml"
    GPU_MEM="11G"
    JOB_TIME="12:00:00"
    CLS_BATCH_SIZE=8
    SEG_BATCH_SIZE=8
    ;;
  600m)
    CLS_MODEL_CONFIG="geobench_exp/configs/model_configs/classification/prithvi_v2_600_all_bands.yaml"
    SEG_MODEL_CONFIG="geobench_exp/configs/model_configs/segmentation/prithvi_v2_600_tl.yaml"
    GPU_MEM="11G"
    JOB_TIME="12:00:00"
    CLS_BATCH_SIZE=4
    SEG_BATCH_SIZE=4
    ;;
  *)
    echo "ERROR: Unknown model '$MODEL'. Use 'tiny', '100m', '300m', or '600m'."
    exit 1
    ;;
esac

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO="/cluster/home/merler/geo-bench-experiments"
EXPERIMENTS_DIR="/cluster/scratch/merler/experiments"
export GEO_BENCH_DIR="$DATA_DIR"

# Patch benchmark_dir in temp copies of the task configs so no YAML needs editing
CLS_TASK_CONFIG=$(mktemp --suffix=.yaml)
SEG_TASK_CONFIG=$(mktemp --suffix=.yaml)
CLS_MODEL_CONFIG_TMP=$(mktemp --suffix=.yaml)
SEG_MODEL_CONFIG_TMP=$(mktemp --suffix=.yaml)
trap 'rm -f "$CLS_TASK_CONFIG" "$SEG_TASK_CONFIG" "$CLS_MODEL_CONFIG_TMP" "$SEG_MODEL_CONFIG_TMP"' EXIT

sed "s|generate_experiment_dir:.*|generate_experiment_dir: $EXPERIMENTS_DIR/prithvi_cls_all|; s|benchmark_dir:.*|benchmark_dir: $DATA_DIR/classification_v1.0|" \
    "$REPO/geobench_exp/configs/prithvi_cls_all_task.yaml" > "$CLS_TASK_CONFIG"
sed "s|generate_experiment_dir:.*|generate_experiment_dir: $EXPERIMENTS_DIR/prithvi_seg_all|; s|benchmark_dir:.*|benchmark_dir: $DATA_DIR/segmentation_v1.0|" \
    "$REPO/geobench_exp/configs/prithvi_seg_all_task.yaml" > "$SEG_TASK_CONFIG"

sed "s|batch_size:.*|batch_size: $CLS_BATCH_SIZE|" "$REPO/$CLS_MODEL_CONFIG" > "$CLS_MODEL_CONFIG_TMP"
sed "s|batch_size:.*|batch_size: $SEG_BATCH_SIZE|" "$REPO/$SEG_MODEL_CONFIG" > "$SEG_MODEL_CONFIG_TMP"

# ─── SLURM settings ───────────────────────────────────────────────────────────
SBATCH_ARGS=(
  --account=es_schin
  --time=$JOB_TIME
  --ntasks=1
  --cpus-per-task=4
  --mem-per-cpu=3G
  --gpus=1
  --gres=gpumem:$GPU_MEM
)

# ─── Generate experiment job directories ──────────────────────────────────────
echo ">>> Generating classification experiments (model=$MODEL, batch_size=$CLS_BATCH_SIZE)..."
cd "$REPO"
uv run geobench_exp-gen_exp \
  --task_config_path "$CLS_TASK_CONFIG" \
  --model_config_path "$CLS_MODEL_CONFIG_TMP"

echo ">>> Generating segmentation experiments (model=$MODEL, batch_size=$SEG_BATCH_SIZE)..."
uv run geobench_exp-gen_exp \
  --task_config_path "$SEG_TASK_CONFIG" \
  --model_config_path "$SEG_MODEL_CONFIG_TMP"

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
submit_jobs "$EXPERIMENTS_DIR/prithvi_cls_all" "classification"
submit_jobs "$EXPERIMENTS_DIR/prithvi_seg_all" "segmentation"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R'"
