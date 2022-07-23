#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Doing imports")
import collections
import concurrent.futures
import math
import multiprocessing
from pathlib import Path
import itertools
import shlex
import subprocess
import time
import re

import beartype
from beartype.typing import *
import h5py  # type: ignore[import]
import fire  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import more_itertools
import numpy as np
import pretty_traceback
# import ujson as json
import orjson as json
import jsonlines as jsonl  # type: ignore[import]
import rich
from tqdm import tqdm  # type: ignore[import]

pretty_traceback.install()
print("Done with other imports")

import data_tokenizer
import general_utils

print("Done with torch")

SIZE_HUMAN_NAMES = {
        0: "B",
        1: "KB",
        2: "MB",
        3: "GB",
    }

def to_human_size(size: int) -> str:
    if size == 0:
        return "0 B"

    exponent = int(math.log(size, 1000))
    mantissa = size / 1000 ** exponent
    return f"{mantissa:.2f} {SIZE_HUMAN_NAMES[exponent]}"

def by_last_number(value: Union[str, Path]) -> int:
    """
    Returns the first number in a string as an int.
    Useful for sorting by the first number in a string.
    """
    return int(re.findall(r"\d+", str(value))[-1])

def cmd(command: list[str]) -> list[str]:
    return subprocess.check_output(command).decode("utf-8").strip().split("\n")

def only_one(it: Iterable):
    iterated = iter(it)
    good = next(iterated)
    for bad in iterated:
        raise ValueError("Expected only one item, got more than one.")
    return good

def check_len(it: Sequence, expected_len: int) -> Sequence:
    if not len(it) == expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(it)}.")
    return it

def count_lines(path: Path) -> int:
    return int(check_len(only_one(cmd(["wc", "-l", str(path)])).split(), 2)[0])

def count_lines_list(paths: list[Path]) -> dict[Path, int]:
    with concurrent.futures.ThreadPoolExecutor() as tp:
        futures = {file: tp.submit(lambda: count_lines(file)) for file in paths}
        return {file: future.result() for file, future in futures.items()}
    
def md5sum(path: Path) -> str:
    return only_one(cmd(["md5sum", str(path)])).strip().split()[0]


########################################################################################
# List the output files of an experiment
########################################################################################
def main(
    name,
    ignore_bads=False,
    max_length=60,
    test_run=False,
):
    general_utils.check_and_print_args(locals().copy(), main)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find and filter the directory pathss
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert name in {"basic", "oracle"}
    target_dir = Path("log_results") / name
    directories = list(target_dir.iterdir())
    active = []
    for file in tqdm(directories):
        target = file / "predictions.h5"
        if target.exists():
            active.append(target)

    del directories
    rich.print(f"Unfiltered directories: {len(active)}")
    rich.print(active)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the data & find the smallest epoch no
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    files = [h5py.File(file, "r") for file in active]
    bad_ones = set()
    min_no = min(min(file["predictions"].shape[0] for file in files), max_length)
    print("Min no:", min_no)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ... Only load the files that share the same keys & are the most recent
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not all([
        np.all(files[0]["input_samples"][:min_no] == 
        files[i]["input_samples"][:min_no]) for i in range(len(files))
    ]): 
        for i in range(len(files)):
            qty = np.sum(
                [
                    files[0]["input_samples"][:min_no] == 
                    files[i]["input_samples"][:min_no] 
                ]
            )
            
            size = active[i].stat().st_size
            if qty == 0:
                bad_ones.add(active[i])

            print(f"{i}:")
            print(f" - {qty}")
            print(f" - {str(active[i]) = }")
            print(f" - {to_human_size(size)}")
            print()

        if len(bad_ones) > 0 and not ignore_bads:
            rich.print("Bad ones:")
            for bad in sorted(bad_ones, key=lambda x: by_last_number(x.parent)):
                rich.print(f"- {bad}, size: {to_human_size(bad.stat().st_size)}")

            print(shlex.join(["rm", "-i", *[str(x) for x in bad_ones]]))
            raise ValueError(f"Found {len(bad_ones)} bad ones")
        else:
            active = [path for path in active if path not in bad_ones]

    del files    
    rich.print(f"Filtered directories: {len(active)}")
    rich.print(active)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Finally open the correct files & check the keys are identical
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    files = [h5py.File(file, "r") for file in active]
    assert all([
        np.all(files[1]["input_samples"][:min_no] == 
        files[i]["input_samples"][:min_no]) for i in range(len(files))
    ]), "Failed still somehow"
    tokenizer = data_tokenizer.ArithmeticTokenizer()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the labels associated to the inputs.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the data & compute agreement
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    agreement_per_epoch = []
    for epoch_idx in tqdm(range(min_no)):
        agreement_per_sample = []
        arrays = [file["predictions"][epoch_idx] for file in files]
        for samples in zip(*arrays):
            text_samples = [tokenizer.decode(sample, ignore_special_symbols=True) for sample in samples]
            top = collections.Counter(text_samples).most_common(1)
            agreement = top[0][1] / len(text_samples)
            agreement_per_sample.append(agreement)
    
        print(f"Epoch {epoch_idx} Agreement:", np.mean(agreement_per_sample))
        agreement_per_epoch.append(np.mean(agreement_per_sample))
    print(f"Averaged agreement: {np.mean(agreement_per_epoch)}")

    if not test_run:
        validation_accuracy_per_epoch = []
        with h5py.File("agreement.h5", "w") as f:
            f.attrs.update(dict(
                average_agreement_per_epoch=agreement_per_epoch,
                average_validation_accuracy_per_epoch=np.mean(validation_accuracy_per_epoch),
                files_used=active,
                file_md5s=[md5sum(file) for file in active],
            ), f)

            f.create_dataset(
                "validation_accuracy_per_epoch", 
                data=validation_accuracy_per_epoch,
            )


if __name__ == "__main__":
    fire.Fire(main)


