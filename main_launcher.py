#!/usr/bin/env python
# coding: utf-8
"""
Launches training jobs on the cluster.
"""

print("Doing imports")

import inspect
import os
from pathlib import Path
import shlex
import subprocess

from beartype import beartype
from beartype.typing import *
import fire  # type: ignore[import]
import ujson as json
import numpy as np
import rich
from tqdm import tqdm  # type: ignore[import]

import bart_relative_attention
import bart_modified
import our_datasets

print("Done with imports")

CONFIG = dict(
    data_name="349_6_6_200000.json.pkl",    
    inf_num=6,
    abs_pos_embs_mode=bart_modified.AbsPosEmbsModes.learned_pos_embs,
    rel_pos_embs_mode=bart_relative_attention.RelPosEmbsChoices.no_rel_pos_embs,
    num_rel_pos_embs=64,

    ###########################################################################
    # Changes often
    ###########################################################################
    freeform_options=[True, False],
    max_level_training=6,
    do_log_results=True,

    ###########################################################################
    # Changes with model size
    ###########################################################################
    batch_size=256,

    ###########################################################################
    # Model Stuff
    ###########################################################################
    # Small config
    h_size=64,
    n_layers=2,
    n_heads=4,
)

def cmd_list_from_config(command_start: list[str], python_start: list[str], config: dict[str, Any]):
    cmd = command_start.copy()
    python_args = python_start.copy()

    for k, v in config.items():
        python_args.append(f"--{k}")
        python_args.append(str(v))
    
    python_ouptut = shlex.join(python_args)
    cmd.append(python_ouptut)
    return cmd


class GpuChoices:
    rtx_8000 = "rtx8000"
    a100 = "a100"
    choices = [rtx_8000, a100]


@beartype
def main(*, dataset_mode: str, n_runs: int, gpu: str = GpuChoices.rtx_8000, max_epochs: int = 150, test_run: bool = False):
    """
    Table of Contents:
    - Get proper variable names from CLI abbrevs
    - Validate the CLI argument values
    - Handle the dataset related options
    - Handle the GPU related options
    - Define some constants & generate the seeds
    - Loop per job:
    --- Fill in the custom config options & generate the command
    --- Launch the job / command
    """

    all_parameters = locals().copy()
    assert all_parameters.keys() == inspect.signature(main).parameters.keys(), (
        all_parameters.keys() - inspect.signature(main).parameters.keys(),
        inspect.signature(main).parameters.keys() - all_parameters.keys(),
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get proper variable names from abbrevs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    name = dataset_mode
    gpu_mode = gpu

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset_mode_choices = ["oracle", "basic"]
    gpu_mode_choices = GpuChoices.choices
    assert dataset_mode in dataset_mode_choices, f"Unknown dataset_mode {dataset_mode}, expected one of {dataset_mode_choices}"
    assert gpu_mode in gpu_mode_choices, f"Unknown gpu_mode {gpu_mode}, expected one of {gpu_mode_choices}"
    assert n_runs <= 30, f"n_runs must be <= 30, got {n_runs = }"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handle the dataset option
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if dataset_mode == "oracle":
        dataset_mode = our_datasets.DatasetTypesChoices.oracle_basic_dataset
    elif dataset_mode == "basic":
        dataset_mode = our_datasets.DatasetTypesChoices.most_basic_dataset
    else:
        raise ValueError(dataset_mode)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handle the gpu option
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    SHARED_COMMAND_START = ["sbatch", "--signal=SIGUSR1@90"]
    if gpu_mode == GpuChoices.rtx_8000:
        COMMAND_START = SHARED_COMMAND_START + [
            "--gres=gpu:1", "--mem=48G", "--cpus-per-task=12"
        ]
    elif gpu_mode == GpuChoices.a100:
        COMMAND_START = SHARED_COMMAND_START + [
            "--reservation=DGXA100", "--gres=gpu:a100:1", "--mem=100G", "--cpus-per-task=16" 
        ]
    else:
        raise ValueError(gpu_mode)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define some constants & generate the seeds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NUMPY_SEED_SEED_SHUFFLER: Final[int] = 453345
    SEED_MAX: Final[int] = 10000
    PROJECT_ROOT : Final[Path] = Path("/home/mila/g/gagnonju/SelfLearnedExplanations")
    # We do the addition thing so it doesn't mess with the interpolation thing at the end
    FOLDER_NAME : Final[str] = name
    FOLDER : Final[Path] = PROJECT_ROOT / "log_results" / FOLDER_NAME
    assert not FOLDER.exists(), f"Folder {FOLDER} already exists"
    FOLDER.mkdir(parents=False)
    BASHRC = str(Path(os.environ["HOME"], ".bashrc"))
    PYTHON_START: Final[list[str]] = ["python", str(PROJECT_ROOT / "main.py")]
    
    with (FOLDER / "shared_config.json").open("w") as f:
        json.dump(CONFIG, f, indent=4)

    np.random.seed(NUMPY_SEED_SEED_SHUFFLER)
    seeds = np.random.permutation(SEED_MAX)[:n_runs]
    seeds.sort()
    def default(x):
        if isinstance(x, Path):
            return str(x)
        raise ValueError((type(x), x))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop per job
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx, seed in enumerate(seeds):

        job_name = f"{name}_{idx}"
        sub_folder = FOLDER / job_name
        sub_folder.mkdir(parents=False)
        loop_specific_cmd = ["-J", job_name]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fill in the custom config options & generate the command
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        config = CONFIG.copy()
        assert config["do_log_results"]
        config["seed"] = int(seed)
        config["dataset_type"] = dataset_mode
        config["path_log_results"] = str(sub_folder / "predictions.jsonl")
        config["max_epochs"] = max_epochs
        checkpoints_folder = sub_folder / "checkpoints"
        config["checkpoints_folder"] = str(checkpoints_folder)
        checkpoints_folder.mkdir(parents=False)
        config["extra_info_file"] = str(sub_folder / "extra_info.json")

        with (sub_folder / "specific_config.json").open("w") as f:
            json.dump(config, f, indent=4, default=default) # type: ignore[call-arg]

        cmd = cmd_list_from_config(
            COMMAND_START + loop_specific_cmd + ["--wrap"], 
            PYTHON_START, 
            config,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Launch the job
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if test_run:
            rich.print(f"Would run with seed {seed} and command:\n\t{cmd}")
        else:
            rich.print(f"Running with seed {seed} and command:\n\t{cmd}")
            subprocess.Popen(cmd)
        


if __name__ == "__main__":
    fire.Fire(main)
