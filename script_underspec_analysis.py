#!/usr/bin/env python
# coding: utf-8


print("Doing imports")
import collections
import concurrent.futures
import enum
import math
import multiprocessing
from pathlib import Path
import itertools
import json
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
import rich
from tqdm import tqdm  # type: ignore[import]

pretty_traceback.install()
print("Done with other imports")

import data_tokenizer
import general_utils

print("Done with torch")
SCRIPT_DIR = Path(__file__).absolute().parent

H5_INPUT_IDS_KEY = "input_samples"
H5_LABEL_IDS_KEY = "label_ids"
H5_PREDICTIONS_KEY = "predictions"

def md5sum(path: Path) -> str:
    return general_utils.only_one(general_utils.cmd(["md5sum", str(path)])).strip().split()[0]

def extract_pred_oracle(string: str) -> Optional[int]:
    string = string.replace(" ", "").strip()
    numbers = re.findall(r"\-?\d+", string.replace(" ", ""))
    if numbers:
        return numbers[-1]
    else:
        return None

extract_pred_basic = extract_pred_oracle

class Modes(str, enum.Enum):
    oracle = "oracle"
    basic = "basic"

########################################################################################
# List the output files of an experiment
########################################################################################
def main(
    name=Modes.oracle.value,
    ignore_bads=True,
    max_length=60,
    test_run=False,
):
    general_utils.check_and_print_args(locals().copy(), main)
    
    if name == Modes.oracle:
        extract_pred_fn = extract_pred_oracle
    elif name == Modes.basic:
        extract_pred_fn = extract_pred_basic
    else:
        raise ValueError(f"Unknown name `{name}`, must be one of {list(Modes)}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find and filter the directory pathss
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert name in {"basic", "oracle"}
    target_dir = SCRIPT_DIR / "log_results" / str(name)
    directories = list(target_dir.iterdir())
    active = []
    for file in tqdm(directories):
        target = file / "predictions.h5"
        if target.exists():
            active.append(target)

    del directories
    print()
    rich.print(f"[bold]Unfiltered directories: {len(active)}")
    general_utils.print_list(general_utils.sort_iterable_text(active))
    print()

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
        np.all(files[0][H5_INPUT_IDS_KEY][:min_no] == 
        files[i][H5_INPUT_IDS_KEY][:min_no]) for i in range(len(files))
    ]): 
        for i in range(len(files)):
            qty = np.sum(
                [
                    files[0][H5_INPUT_IDS_KEY][:min_no] == 
                    files[i][H5_INPUT_IDS_KEY][:min_no] 
                ]
            )
            
            size = active[i].stat().st_size
            if qty == 0:
                bad_ones.add(active[i])

            print(f"{i}:")
            print(f" - {qty}")
            print(f" - {str(active[i]) = }")
            print(f" - {general_utils.to_human_size(size)}")
            print()
        
        assert not ignore_bads or len(bad_ones) == 0

        if len(bad_ones) > 0 and not ignore_bads:
            rich.print("Bad ones:")
            for bad in general_utils.sort_iterable_text(bad_ones):
                rich.print(f"\t- {bad}, size: {general_utils.to_human_size(bad.stat().st_size)}")

            print(shlex.join(["rm", "-i", *[str(x) for x in bad_ones]]))
            raise ValueError(f"Found {len(bad_ones)} bad ones")
        else:
            active = [path for path in active if path not in bad_ones]

    [file.close() for file in files]
    del files    
    print()
    rich.print(f"[bold]Filtered directories: {len(active)}")
    general_utils.print_list(general_utils.sort_iterable_text(active))
    print()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Finally open the correct files & check the keys are identical
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    files = [h5py.File(file, "r") for file in active]
    assert all([
        np.all(files[1][H5_INPUT_IDS_KEY][:min_no] == 
        files[i][H5_INPUT_IDS_KEY][:min_no]) for i in range(len(files))
    ]), "Failed still somehow"
    assert all([H5_LABEL_IDS_KEY in file for file in files]), [file.keys() for file in files]
    
    print()
    rich.print("[bold]Passed the checks.")
    print()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Accuracies
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("[bold]Computing accuracies...")
    tokenizer = data_tokenizer.ArithmeticTokenizer()

    labels = []
    for file in tqdm(files, desc="Decoding labels"):
        for sample in file[H5_LABEL_IDS_KEY]:
            label = tokenizer.decode(sample, ignore_special_symbols=True).replace(" ", "")
            int(label)  # Labels should be valid integers. This "useless" conversion checks this.
            labels.append(label)

    accuracies = np.zeros((len(active), min_no), dtype=np.float64)
    for epoch_idx in range(min_no):
        for file_idx, file in enumerate(files):
            values = []
            for label, pred in zip(labels, file[H5_PREDICTIONS_KEY][epoch_idx]):
                pred_str = tokenizer.decode(pred, ignore_special_symbols=True)
                maybe_pred_num = extract_pred_fn(pred_str)
                values.append(maybe_pred_num == label)
            accuracies[file_idx, epoch_idx] = np.mean(values)
        rich.print(f"{epoch_idx = }: {np.mean(accuracies[:, epoch_idx]):0.1%}")
    
    per_epoch_acc = np.mean(accuracies, axis=0)
    rich.print(f"Per epoch accuracy: {per_epoch_acc}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute agreement
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    agreement_per_epoch = []
    for epoch_idx in tqdm(range(min_no), desc="Computing agreement scores per epoch"):
        agreement_per_sample = []
        arrays = [file["predictions"][epoch_idx] for file in files]
        for samples in zip(*arrays):
            text_samples = [extract_pred_fn(tokenizer.decode(sample, ignore_special_symbols=True)) for sample in samples]
            top = collections.Counter(text_samples).most_common(1)
            agreement = top[0][1] / len(text_samples)
            agreement_per_sample.append(agreement)
    
        print(f"Epoch {epoch_idx} Agreement: {np.mean(agreement_per_sample):0.1%}, Accuracy: {per_epoch_acc[epoch_idx]:0.1%}")
        agreement_per_epoch.append(np.mean(agreement_per_sample))
    print(f"Averaged agreement: {np.mean(agreement_per_epoch):0.1%}")

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

            f.create_dataset(
                "accuracies",
                data=accuracies
            )


if __name__ == "__main__":
    fire.Fire(main)



# %%
