"""Create job directories from which the experiments will be run."""

import argparse
import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from geobench.task import task_iterator
from omegaconf import OmegaConf

from .job import Job



def generate_experiment_name(config: dict) -> str:
    """Generate the name of the directory for the experiment.

    Args:
        config: dictionary containing config

    Returns:
        experiment_name: name of the experiment directory
    """
    benchmark_dir = config["experiment"]["benchmark_dir"]

    experiment_prefix = f"{config['experiment']['experiment_name'] or 'experiment'}_{os.path.basename(benchmark_dir)}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    if "weights" in config["model"]:
        experiment_prefix += f"{config['model']['weights']}"
    if "model" in config["model"]:
        experiment_prefix += f"{config['model']['model']}"
    elif "encoder_type" in config["model"] and "decoder_type" in config["model"]:
        experiment_prefix += f"{config['model']['encoder_type']}_{config['model']['decoder_type']}"
    elif "prithvi_variant" in config["model"]:
        experiment_prefix += f"{config['model']['prithvi_variant']}"
    else:
        # fallback: derive a name from the _target_ class path
        experiment_prefix += config["model"].get("_target_", "unknown").split(".")[-1]

    if config["experiment"]["experiment_name"] is not None:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"]) / experiment_prefix
    else:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"])  # type: ignore[no-redef]

    return experiment_dir


def _count_channels(band_names: list, bands_info_list: list) -> int:
    """Return total number of input channels, accounting for MultiBand entries.

    A MultiBand (e.g. HyperSpectralBands) contributes ``band_info.n_bands``
    channels when unpacked by ``pack_to_3d``; all other bands contribute 1.
    """
    from geobench.dataset import MultiBand

    info_by_name = {bi.name: bi for bi in bands_info_list}
    total = 0
    for name in band_names:
        bi = info_by_name.get(name)
        if bi is not None and isinstance(bi, MultiBand) and bi.n_bands:
            total += bi.n_bands
        else:
            total += 1
    return total


def get_band_names(config: Dict[str, Any], task_specs) -> Dict[str, Any]:
    """Get the appropriate band names for experiments.

    Args:
        config: dictionary containing config
        task_specs: task specifications

    Returns:
        config: dictionary containing config
    """
    if config["datamodule"]["band_names"] == "all":
        band_names = [band_info.name for band_info in task_specs.bands_info]
        config["datamodule"]["band_names"] = band_names
        config["model"]["in_channels"] = _count_channels(band_names, task_specs.bands_info)
    else:
        config["model"]["in_channels"] = _count_channels(
            config["datamodule"]["band_names"], task_specs.bands_info
        )

    return config


def experiment_generator(task_config_path: str, model_config_path: str) -> None:
    """Generate job directories for experiments.

    Args:
        task_config_path: path to task config file
        model_config_path: path to model config file
    """
    # Load the task and model configurations
    task_config = OmegaConf.load(task_config_path)
    model_config = OmegaConf.load(model_config_path)

    # Merge the configurations
    config = OmegaConf.merge(task_config, model_config)

    experiment_dir = generate_experiment_name(config)
    benchmark_dir = config["experiment"]["benchmark_dir"]

    for task_specs in task_iterator(
        benchmark_name=config["experiment"]["benchmark_name"],
        benchmark_dir=benchmark_dir,
        ignore_task=config["experiment"].get("ignore_tasks", None),
    ):
        print(task_specs.dataset_name)
        task_config = copy.deepcopy(config)
        task_config = get_band_names(task_config, task_specs)

        for i in range(config["experiment"]["num_seeds"]):
            # set seed to be used in experiment
            task_config["experiment"]["seed"] = i
            job_dir = experiment_dir / task_specs.dataset_name / f"seed_{i}"
            job = Job(job_dir)
            job.save_config(task_config)
            job.save_task_specs(task_specs)
            job.write_script(job_dir=str(job_dir))


def start() -> None:
    """Start generating job directories for experiments."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="create_job_dirs.py",
        description="Generate experiment directory structure based on user-defined configr",
    )
    parser.add_argument(
        "--task_config_path",
        help="The path to the task configuration file.",
        required=True,
    )
    parser.add_argument(
        "--model_config_path",
        help="The path to the model configuration file.",
        required=True,
    )

    args = parser.parse_args()

    experiment_generator(task_config_path=args.task_config_path, model_config_path=args.model_config_path)


if __name__ == "__main__":
    start()
