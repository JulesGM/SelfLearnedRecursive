#!/usr/bin/env python
# coding: utf-8

"""

"""

print("Doing imports")
import collections
import concurrent.futures
import enum
import math
import multiprocessing
import multiprocessing.pool as mp_pool
from pathlib import Path
import itertools
import json
import shlex
import subprocess
import time
import re

from beartype import beartype
from beartype.typing import *
import h5py  # type: ignore[import]
import fire  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import more_itertools
import numpy as np
import pretty_traceback  # type: ignore[import]
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


def compute_accuracy(path, min_no, tokenizer, labels, extract_pred_fn) -> tuple[np.ndarray, float]:
    start = time.process_time()
    accuracies = np.zeros(min_no, dtype=np.float64)
    with h5py.File(path, "r") as file:
        for epoch_idx in range(min_no):
            values = []
            for label, pred in more_itertools.zip_equal(
                labels, file[H5_PREDICTIONS_KEY][epoch_idx]):

                pred_str = tokenizer.decode(pred, ignore_special_symbols=True)
                maybe_pred_num = extract_pred_fn(pred_str)
                values.append(maybe_pred_num == label)
            accuracies[epoch_idx] = np.mean(values)

    return accuracies, time.process_time() - start


def compute_agreement(
    epoch_idx: int,  # Small
    tokenizer,   # Small
    extract_pred_fn: Callable,  # Small
    paths: List[Path],  # Small 
    per_epoch_acc = None,  # Big (if not None), not currently used
):
    top_1_agreement_per_sample = []
    pairwise_agreement_per_sample = []
    files = [h5py.File(path, "r") for path in paths]
    arrays = [file["predictions"][epoch_idx] for file in files]

    for samples in zip(*arrays):
        text_samples = [extract_pred_fn(tokenizer.decode(
            sample, ignore_special_symbols=True)) for sample in samples]
        top = collections.Counter(text_samples).most_common(1)
        top_1_agreement = top[0][1] / len(text_samples)
        top_1_agreement_per_sample.append(top_1_agreement)
        
        pairwise_good = 0
        pairwise_total = 0
        for i in range(len(text_samples)):
            for j in range(i + 1, len(text_samples)):
                pairwise_good += text_samples[i] == text_samples[j]
                pairwise_total += 1
        
        pairwise_agreement_per_sample.append(pairwise_good / pairwise_total)

    # print(f"Epoch {epoch_idx}:")
    # if per_epoch_acc:
    #     print(f"\t- Accuracy:            {per_epoch_acc[epoch_idx]:0.1%}")
    # print(f"\t- Pairwise Agreement:  {np.mean(pairwise_agreement_per_sample):0.1%}")
    # print(f"\t- Top-1 Agreement:     {np.mean(top_1_agreement_per_sample):0.1%}")
    # print()
    
    return top_1_agreement_per_sample, pairwise_agreement_per_sample


class Modes(str, enum.Enum):
    oracle = "oracle"
    basic = "basic"

########################################################################################
# List the output files of an experiment
########################################################################################
@beartype
def main(
    name: str = Modes.basic.value,
    ignore_bads: bool = True,
    max_length: int = 60,
    test_run: bool = False,
    n_cpus: int = 12,
):
    general_utils.check_and_print_args(locals().copy(), main)
    
    if name == Modes.oracle:
        extract_pred_fn = extract_pred_oracle
    elif name == Modes.basic:
        extract_pred_fn = extract_pred_basic
    else:
        raise ValueError(f"Unknown name `{name}`, must be one of {list(Modes)}")

    target_file = SCRIPT_DIR / f"agreement_{name}.h5"
    assert not target_file.exists(), f"File {target_file} already exists."

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
    ]), "Some different input ids"

    assert all([
        np.all(files[1][H5_LABEL_IDS_KEY][:min_no] == 
        files[i][H5_LABEL_IDS_KEY][:min_no]) for i in range(len(files))
    ]), "Some different label ids"

    print()
    rich.print("[bold]Passed the checks.")
    print()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Accuracies
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("[bold]Computing accuracies...")
    tokenizer = data_tokenizer.ArithmeticTokenizer()

    labels = []
    accuracies = np.zeros((len(active), min_no), dtype=np.float64)

    for sample in files[0][H5_LABEL_IDS_KEY]:
        label = tokenizer.decode(sample, ignore_special_symbols=True).replace(" ", "")
        int(label)  # Labels should be valid integers. This "useless" conversion checks this.
        labels.append(label)
    
    promises = []
    accuracies_list = []
    times = []

    cpus_to_use = min(n_cpus, len(active))
    start = time.perf_counter()
    with multiprocessing.Pool(cpus_to_use) as pool:
        for i, path in enumerate(active):
            promises.append(
                pool.apply_async(
                    compute_accuracy, (path, min_no, tokenizer, labels, extract_pred_fn)
                )
            )
        
        for promise in tqdm(promises):
            acc, time_ = promise.get()
            accuracies_list.append(acc)
            times.append(time_)
    
    computation_time = time.perf_counter() - start

    rich.print(f"[bold]Perf analysis")
    rich.print(f"\t- Computation time: {computation_time}s")
    print("{cpus_to_use} cpus used")
    print()

    accuracies = np.array(accuracies_list)
    assert accuracies.shape == (len(active), min_no), f"{accuracies.shape = }, {(len(active), min_no) = }"

    for epoch_idx in range(min_no):
        rich.print(f"{epoch_idx = }: {np.mean(accuracies[:, epoch_idx]):0.1%}")
    
    per_epoch_acc = np.mean(accuracies, axis=0)
    rich.print(f"Per epoch accuracy: {per_epoch_acc}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute agreement
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print()
    rich.print("[bold]Computing agreement.")
    
    top_1_all = []
    pairwise_all = []

    start = time.perf_counter()
    cpus_to_use = min(n_cpus, min_no)
    with multiprocessing.Pool(cpus_to_use) as pool:
        promises = []
        for epoch_idx in range(min_no):
            promises.append(pool.apply_async(
                compute_agreement, 
                (epoch_idx, tokenizer, extract_pred_fn, active,), 
                dict(per_epoch_acc=None)
            ))

        for promise in tqdm(promises, desc="Computing agreement scores per epoch"):
            top_1_agreement, pairwise_agreement = promise.get()
            top_1_all.append(top_1_agreement)
            pairwise_all.append(pairwise_agreement)

    np_top_1_all = np.array(top_1_all, dtype=np.float64)
    np_pairwise_all = np.array(pairwise_all, dtype=np.float64)

    duration = time.perf_counter() - start
    rich.print(f"[bold]Done computing agreement.[/bold]")
    rich.print(f"\t- Computation time: {duration:0.0f}s")
    rich.print(f"\t- CPUs used: {cpus_to_use}")

    print()
    rich.print(f"[bold]Averaged Top-1 Agreement: {np.mean(np_top_1_all):0.1%}")
    rich.print(f"[bold]Averaged Pairwise Agreement: {np.mean(np_pairwise_all):0.1%}")
    print()

    rich.print(f"[bold]Saving results to {target_file} ...")
    if not test_run:
        with h5py.File(target_file, "w") as f:
            f.attrs.update(dict(
                files_used=[str(path) for path in active],
                file_md5s=[md5sum(path) for path in active],
            ))

            f.create_dataset(
                "top-1_all", 
                data=np_top_1_all,
            )

            f.create_dataset(
                "pairwise_all", 
                data=np_pairwise_all, 
            )

            f.create_dataset(
                "accuracy_per_epoch", 
                data=accuracies,
            )
    rich.print("[bold]All done.")


if __name__ == "__main__":
    fire.Fire(main)



# %%
