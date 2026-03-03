# Copilot Instructions for geo-bench-experiments

## Project Purpose
Experiment harness for benchmarking geospatial foundation models on the [GEO-Bench](https://github.com/ServiceNow/geo-bench) dataset suite. Supports classification and segmentation tasks across multiple satellite/remote-sensing datasets.

## Architecture Overview

**Two-phase experiment workflow:**
1. **Generate** – `geobench_exp-gen_exp` reads a task config + model config, merges them with OmegaConf, iterates over every dataset in the benchmark, and writes one `Job` directory per (dataset × seed). Each directory contains `config.yaml`, `task_specs.pkl`, and an executable `run.sh`.
2. **Run** – `geobench_exp-run_exp --job_dir <path>` loads the `Job`, instantiates model/datamodule via Hydra's `instantiate()`, then calls `trainer.fit()` + `trainer.test()`.

**Key files:**
- [geobench_exp/generate_experiment.py](../geobench_exp/generate_experiment.py) – phase-1 entry point
- [geobench_exp/run_experiment.py](../geobench_exp/run_experiment.py) – phase-2 entry point
- [geobench_exp/job.py](../geobench_exp/job.py) – `Job` class: manages directory layout, config/task_specs persistence, metrics retrieval
- [geobench_exp/torch_toolbox/model.py](../geobench_exp/torch_toolbox/model.py) – `GeoBenchBaseModule` (Lightning), `GeoBenchClassifier`, `GeoBenchSegmentation`
- [geobench_exp/torch_toolbox/model_utils.py](../geobench_exp/torch_toolbox/model_utils.py) – `generate_trainer()`: configures early-stopping, checkpointing, CSV/WandB loggers
- [geobench_exp/torch_toolbox/dataset.py](../geobench_exp/torch_toolbox/dataset.py) – `DataModule`, kornia/torchgeo transforms

## Config System
Configs are split into **task configs** (`geobench_exp/configs/classification_task.yaml`, `segmentation_task.yaml`) merged with **model configs** (`geobench_exp/configs/model_configs/**`). OmegaConf merges them; Hydra `instantiate()` is used for `model`, `datamodule`, and `trainer` blocks.

- `_target_` keys are mandatory for Hydra instantiation (e.g. `_target_: geobench_exp.torch_toolbox.model.GeoBenchClassifier`)
- `band_names: "all"` is resolved to the full band list from `TaskSpecifications` before saving the per-job config
- `in_channels` is auto-set from the band count when `band_names: "all"`
- WandB logging is opt-in: add a `wandb:` section to the config; omitting it uses CSV-only logging

### Selecting which datasets to run: `ignore_tasks`

**Despite its misleading name, `ignore_tasks` is an allow-list (keep-list), not an exclude-list.**

If `ignore_tasks` is set, `generate_experiment.py` will generate jobs **only** for the datasets listed. If `ignore_tasks` is absent or `null`, jobs are generated for **all** datasets in the benchmark.

Usage in a task config YAML:
```yaml
experiment:
  # Run only these two datasets (all others are skipped):
  ignore_tasks: ["m-eurosat", "m-forestnet"]
```

To run every dataset, simply omit the key or comment it out:
```yaml
experiment:
  # ignore_tasks: [...]   # commented out → all datasets are benchmarked
```

Available classification dataset names (from `classification_v1.0`):
`m-brick-kiln`, `m-bigearthnet`, `m-eurosat`, `m-so2sat`, `m-pv4ger`, `m-forestnet`, `m-cashew-plantation`, `m-SA-crop-type`, `m-nz-cattle`, `m-NeonTree`

## Adding a New Model
1. Create a YAML under `geobench_exp/configs/model_configs/<task>/yourmodel.yaml` with the appropriate `_target_`, backbone/decoder keys, optimizer, and `batch_size` override.
2. If the model class doesn't exist, subclass `GeoBenchBaseModule` and implement `configure_the_model()` to assign `self.model`.
3. Add the input size to `get_desired_input_sizes()` in [dataset.py](../geobench_exp/torch_toolbox/dataset.py) if not already present.

## Metric Tracking Convention
The monitored checkpoint/early-stopping metric is dataset-specific (hardcoded in `generate_trainer()`):
- Classification datasets → `val_Accuracy`
- Multi-label (`m-bigearthnet`) → `val_F1Score`
- Segmentation datasets → `val_Jaccard`
Override with `early_stopping_metric` in the model config.

## Validation Loop Design
`validation_step` handles **both** val and test splits via `dataloader_idx` (0 = val, 1 = test). Test metrics are therefore computed and logged during `on_validation_epoch_end`, not just in `test_step`.

## Developer Workflows

**Environment:** the project uses a `uv`-managed virtual environment (`.venv/` at the repo root). All commands must be prefixed with `uv run` or run after `source .venv/bin/activate`. Never use conda for this project.

**Known dependency issues:**
- If a package is missing at runtime, install it with `uv pip install <package>` — do **not** edit `pyproject.toml` unless adding a permanent dependency.

**Install (editable):**
```bash
uv sync           # create/update .venv from pyproject.toml
uv pip install -e ".[dev]"
pre-commit install
```

**Run tests** (uses test HDF5 data in `tests/data/`):
```bash
uv run pytest                          # all standard tests
uv run pytest --optional               # includes slow/optional tests
uv run pytest -m "not slow"            # skip slow tests
```

**Lint / format:**
```bash
uv run black geobench_exp tests
uv run isort geobench_exp tests
uv run flake8 geobench_exp tests       # max-line-length=120
```

**Generate + run a quick experiment:**
```bash
export GEO_BENCH_DIR=/scratch3/merler/geo-bench/datasets
uv run geobench_exp-gen_exp \
  --task_config_path geobench_exp/configs/classification_task.yaml \
  --model_config_path geobench_exp/configs/model_configs/classification/resnet18.yaml

# Then execute any generated run.sh inside a tmux session so the user can
# detach/reattach and monitor live progress at any time:
tmux new-session -s experiment       # create a named session
sh experiments/0.01x_train_.../m-eurosat/seed_0/run.sh
# Detach with Ctrl-b d; reattach with: tmux attach -t experiment
```

**CRITICAL — always export `GEO_BENCH_DIR` before generating experiments:**
`job.py` captures `GEO_BENCH_DIR` at generation time and writes it into the generated `run.sh`. If the variable is not set when `gen_exp` runs, the `export` line is omitted from `run.sh`. At runtime, `TaskSpecifications.get_dataset_dir()` then resolves paths using geobench's hardcoded default (`~/dataset/geobench/...`), causing a `ValueError: dataset_dir ... does not exist`.

- **Always generate with the variable set:**
  ```bash
  export GEO_BENCH_DIR=/scratch3/merler/geo-bench/datasets
  uv run geobench_exp-gen_exp ...
  ```
- **If a run fails with that error on an already-generated job**, add the missing line to `run.sh` manually:
  ```bash
  # Insert after the #!/bin/bash shebang, before the cd line:
  export GEO_BENCH_DIR=/scratch3/merler/geo-bench/datasets
  ```

**Running long / background experiments — always use tmux, never nohup:**
- Start a named session: `tmux new-session -s <name>`
- Split pane to watch GPU live: `Ctrl-b %` then `watch -n2 nvidia-smi`
- Detach without killing: `Ctrl-b d`
- Reattach later: `tmux attach -t <name>`
- List sessions: `tmux ls`
- Tail the CSV metrics from outside the session:
  ```bash
  tail -f experiments/.../csv_logs/version_0/metrics.csv
  ```
- Do **not** use `nohup ... &` — it hides output and makes progress monitoring harder.

## Key External Dependencies
- `geo-benchmark` (`geobench`) – `TaskSpecifications`, `task_iterator`, `Sample` objects from HDF5 data
- `segmentation-models-pytorch` – decoder library for segmentation (installed from GitHub main)
- `torchgeo` – pretrained weights and geo-specific transforms
- `lightning` (PyTorch Lightning) – training loop
- `omegaconf` + `hydra-core` – config merging and class instantiation
- `kornia` – image augmentation pipeline
- `terratorch` – Prithvi-EO v2 backbone factory (installed via `uv pip install terratorch`)
- `wandb` – optional experiment tracking (add a `wandb:` section to the task config to enable)
