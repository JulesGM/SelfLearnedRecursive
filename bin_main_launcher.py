#!/usr/bin/env python
# coding: utf-8
"""
Launches training jobs on the cluster.
"""

print("Doing imports")

import inspect
import os
from pathlib import Path
import random
import shlex
import subprocess
print("Done with stdlib")

from beartype import beartype
from beartype.typing import *
import fire  # type: ignore[import]
import json
import rich
print("Done with third party")

import data_datasets
import general_shared_constants
import general_utils
print("Done with first party")

print("Done with imports")


SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT : Final[Path] = SCRIPT_DIR
PYTHON_START: Final[list[str]] = [str(PROJECT_ROOT / "bin_main.py")]

CONFIG = dict(
    data_name="349_6_6_200000.json.pkl",    
    subset_path=SCRIPT_DIR / "data/subsets/subset_10000_seed_453345_of_349_6_6_200000.json",
    inf_num=6,
    abs_pos_embs_mode=general_shared_constants.AbsPosEmbsModes.learned_pos_embs,
    rel_pos_embs_mode=general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs,
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

def cmd_list_from_config(command_start: list[str], python_start: list[str], config: dict[str, Any], wrap_mode: bool):
    cmd = command_start.copy()
    python_args = python_start.copy()
    for k, v in config.items():
        python_args.append(f"--{k}")
        python_args.append(str(v))
    
    if wrap_mode:
        cmd.append("--wrap")    
        cmd.append(shlex.join(["python"] + python_args))
    else:
        cmd.extend(["load_env_then_run.sh"] + python_args)
    return cmd

def random_subset(max_val, num_vals):
    numbers = list(range(max_val))
    random.shuffle(numbers)
    return numbers[:num_vals]

class GpuChoices:
    rtx_8000 = "rtx8000"
    a100 = "a100"
    choices = [rtx_8000, a100]


def json_default(x):
    if isinstance(x, Path):
        return str(x)
    raise ValueError((type(x), x))


def make_command_start(folder_stdout, folder_stderr, gpu_mode):
    shared_command_start = [
        "sbatch", 
        "--signal=SIGUSR1@90", 
        "--output", str(folder_stdout / "slurm-%j.out"), 
        "--error", str(folder_stderr / "slurm-%j.err")
    ]

    if gpu_mode == GpuChoices.rtx_8000:
        command_start = shared_command_start + [
            "--gres=gpu:1", "--mem=48G", "--cpus-per-task=12"
        ]
    elif gpu_mode == GpuChoices.a100:
        command_start = shared_command_start + [
            "--reservation=DGXA100", "--gres=gpu:a100:1", "--mem=100G", "--cpus-per-task=16" 
        ]
    else:
        raise ValueError(gpu_mode)
    return command_start
    
PathType = Union[str, Path]

class App:
    @beartype
    def main(self, *, dataset_mode: str, n_runs: int, gpu: str = GpuChoices.rtx_8000, max_epochs: int = 150, test_run: bool = False):
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
        assert all_parameters.keys() == inspect.signature(App.main).parameters.keys(), (
            all_parameters.keys() - inspect.signature(App.main).parameters.keys(),
            inspect.signature(App.main).parameters.keys() - all_parameters.keys(),
        )

        rich.print("\n[bold]Parameters:")
        general_utils.print_dict(all_parameters)

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
            dataset_mode = general_shared_constants.DatasetTypesChoices.oracle_basic_dataset
        elif dataset_mode == "basic":
            dataset_mode = general_shared_constants.DatasetTypesChoices.most_basic_dataset
        else:
            raise ValueError(dataset_mode)

        assert Path(cast(str, CONFIG["subset_path"])).exists(), (
            f"subset_path {CONFIG['subset_path']} does not exist"
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define some constants & generate the seeds
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SEED_SEED_SHUFFLER: Final[int] = 453345
        SEED_MAX: Final[int] = 10000
        
        # We do the addition thing so it doesn't mess with the interpolation thing at the end
        FOLDER_NAME : Final[str] = name
        FOLDER : Final[Path] = PROJECT_ROOT / "log_results" / FOLDER_NAME
        assert not FOLDER.exists(), f"Folder \"{FOLDER}\" already exists"
        
        FOLDER_STDOUT = FOLDER / "stdout"
        FOLDER_STDERR = FOLDER / "stderr"

        if not test_run:
            FOLDER.mkdir(parents=False)
            FOLDER_STDOUT.mkdir(parents=False)
            FOLDER_STDERR.mkdir(parents=False)

            with (FOLDER / "shared_config.json").open("w") as f:
                json.dump(CONFIG, f, indent=4, default=json_default)

        random.seed(SEED_SEED_SHUFFLER)
        seeds = random_subset(SEED_MAX, n_runs)
        seeds.sort()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Handle the gpu option
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        command_start = make_command_start(FOLDER_STDOUT, FOLDER_STDERR, gpu_mode)
        

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop per job
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        processes: List[subprocess.Popen] = []
        for idx, seed in enumerate(seeds):

            job_name = f"{name}_{idx}"
            sub_folder = FOLDER / job_name
            if not test_run:
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
            if not test_run:
                checkpoints_folder.mkdir(parents=False)
            config["extra_info_file"] = str(sub_folder / "extra_info.json")

            if not test_run:
                with (sub_folder / "specific_config.json").open("w") as f:
                    json.dump(config, f, indent=4, default=json_default) # type: ignore[call-arg]

            WRAP_MODE = True
            cmd = cmd_list_from_config(
                command_start + loop_specific_cmd, 
                PYTHON_START, 
                config,
                WRAP_MODE,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Launch the job
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if test_run:
                rich.print(f"Would run with seed {seed} and command:\n\t{cmd}")
            else:
                rich.print(f"Running with seed {seed} and command:\n\t{cmd}")
                processes.append(subprocess.Popen(cmd))
            
        processes = [p.wait() for p in processes]  # type: ignore[misc]
        
        print()
        rich.print(f"STDOUT at \"[cyan]{FOLDER_STDOUT}[/]\"")
        rich.print(f"STDERR at \"[cyan]{FOLDER_STDERR}[/]\"")


    @beartype
    def resume(
        self, 
        path_to_specific_conf: PathType, 
        gpu_mode : str = GpuChoices.a100, 
        original_job_name: Optional[str] = None, 
        folder_stdout: Optional[PathType] = None, 
        folder_stderr: Optional[PathType] = None
    ):
        """
        Good meta info would be:
        - folder_stdout
        - folder_stderr
        - job_name
        """

        all_parameters = locals().copy()
        assert all_parameters.keys() == inspect.signature(App.resume).parameters.keys(), (
            all_parameters.keys() - inspect.signature(App.resume).parameters.keys(),
            inspect.signature(App.resume).parameters.keys() - all_parameters.keys(),
        )

        rich.print("\n[bold]Parameters:")
        general_utils.print_dict(all_parameters)

        assert Path(path_to_specific_conf).exists(), f"path_to_specific_conf {path_to_specific_conf} does not exist"
        assert gpu_mode in GpuChoices.choices, f"Unknown gpu_mode {gpu_mode}, expected one of {GpuChoices.choices}"

        ###############################################################################
        # We default to the values that would be given by our launch script
        ###############################################################################
        if original_job_name is None:
            original_job_name = Path(path_to_specific_conf).parent.name

        if folder_stdout is None:
            folder_stdout = Path(path_to_specific_conf).parent.parent / "stdout"

        if folder_stderr is None:
            folder_stderr = Path(path_to_specific_conf).parent.parent / "stderr"

        assert Path(folder_stdout).exists(), f"folder_stdout {folder_stdout} does not exist"
        assert Path(folder_stderr).exists(), f"folder_stderr {folder_stderr} does not exist"

        ###############################################################################
        # We build the command
        ###############################################################################
        with open(path_to_specific_conf) as f:
            config = json.load(f)
        
        command_start = make_command_start(
            folder_stdout, folder_stderr, gpu_mode,
        )
        
        loop_specific_cmd = ["-J", "resumed_" + original_job_name]

        WRAP_MODE = True
        cmd = cmd_list_from_config(
            command_start + loop_specific_cmd, 
            PYTHON_START, 
            config,
            WRAP_MODE,
        )

        ###############################################################################
        # We run the command
        ###############################################################################
        rich.print("\n[bold]Resuming:")
        rich.print(cmd)

        process = subprocess.Popen(cmd)
        process.wait()

        print()
        rich.print(f"STDOUT at \"[cyan]{folder_stdout}[/]\"")
        rich.print(f"STDERR at \"[cyan]{folder_stderr}[/]\"")        





if __name__ == "__main__":
    fire.Fire(App)
