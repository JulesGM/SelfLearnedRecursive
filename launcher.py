print("Doing imports")
import random
import shlex
import subprocess

from beartype.typing import *
import fire
import jsonlines as jsonl  # type: ignore[import]
import numpy as np
import torch
import rich
from tqdm import tqdm  # type: ignore[import]

import modded_bart
import bart_rel_att
import our_datasets
print("Done with imports")

CONFIG = dict(
    data_name="349_6_6_10000.json.pkl",    
    inf_num=6,
    abs_pos_embs_mode=modded_bart.AbsPosEmbsModes.learned_pos_embs,
    rel_pos_embs_mode=bart_rel_att.RelPosEmbsChoices.no_rel_pos_embs,
    num_rel_pos_embs=64,

    ###########################################################################
    # Changes often
    ###########################################################################
    freeform_options=[True, False],
    dataset_type=our_datasets.DatasetTypesChoices.oracle_basic_dataset,
    max_level_training=6,
    do_log_results=True,
    # path_log_results="log_results/lel.txt",

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

def main():
    N_RUNS = 1
    SEED_MAX = 10000
    PATH_LOG_RESULTS_PREFIX = "log_results/{ds_type}_{idx}.jsonl"
    COMMAND_START = [
        "sbatch", "--gres=gpu:1", "--mem=48G", "--cpus-per-task=12", "--wrap", 
    ]
    PYTHON_START = [
        "python", "/home/mila/g/gagnonju/SelfLearnedExplanations/main.py",
    ]

    seeds = np.random.permutation(SEED_MAX)[:N_RUNS]
    seeds.sort()

    for idx, seed in enumerate(seeds):
        config = CONFIG.copy()
        assert config["do_log_results"]
        config["seed"] = seed
        config["path_log_results"] = PATH_LOG_RESULTS_PREFIX.format(
            ds_type=config["dataset_type"], 
            idx=idx
        )
        cmd = cmd_list_from_config(COMMAND_START, PYTHON_START, config)
        rich.print(f"Running with seed {seed} and command:\n\t{cmd}")
        subprocess.Popen(cmd)


if __name__ == "__main__":
    fire.Fire(main)