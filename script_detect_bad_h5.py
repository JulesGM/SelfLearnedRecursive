
#!/usr/bin/env python3
# coding: utf-8

"""
- Checks if the h5 files have labels.
- Checks if the h5 files have inputs.
- Checks if all the h5 files have the same inputs.
- Checks if all the h5 files have the same labels.

"""

print("Doing imports.")
# Stdlib
import math
from pathlib import Path
import itertools
import json
import shlex
import subprocess
import time
import re

# Third Party
from beartype import beartype
from beartype.typing import *
import h5py  # type: ignore[import]
import fire  # type: ignore[import]
import more_itertools
import numpy as np
import pretty_traceback  # type: ignore[import]
import rich
import torch
from tqdm import tqdm  # type: ignore[import]

# First Party
import general_utils
print("Done with imports.")

pretty_traceback.install()

SCRIPT_DIR = Path(__file__).absolute().parent

H5_INPUT_IDS_KEY = "input_samples"
H5_LABEL_IDS_KEY = "label_ids"
H5_PREDICTIONS_KEY = "predictions"


def length_stats(h5):
    lengths = (h5[H5_PREDICTIONS_KEY][:] != 0).cumsum(axis=2).max(axis=2)
    assert lengths.shape == h5[H5_PREDICTIONS_KEY].shape[:2]
    return lengths.mean(), lengths.std(), lengths.max()


def detect_bads(h5_paths: Sequence[Union[Path, str]], num_epochs: int):
    h5_paths = cast(Sequence[Path], [Path(h5_path) for h5_path in h5_paths])
    files = [h5py.File(path, "r") for path in h5_paths]

    all_inputs_are_the_same = all([
        np.all(files[0][H5_INPUT_IDS_KEY][:num_epochs] == 
        files[i][H5_INPUT_IDS_KEY][:num_epochs]) 
        for i in range(len(files))
    ])
    
    all_labels_are_the_same = all([
        np.all(files[0][H5_LABEL_IDS_KEY][:num_epochs] == 
        files[i][H5_LABEL_IDS_KEY][:num_epochs]) 
        for i in range(len(files))
    ])

    assert all_inputs_are_the_same
    rich.print("[bold green]All files had the same inputs.")
    assert all_labels_are_the_same
    rich.print("[bold green]All files had the same labels.")

    print()
    rich.print("[bold]Doing prediction length stats.")
    means = []
    stds = []
    maxes = []

    for i, (file, path) in enumerate(more_itertools.zip_equal(tqdm(files), h5_paths)):
        size = path.stat().st_size
        mean, std, max_ = length_stats(file)
        
        means.append(mean)
        stds.append(std)
        maxes.append(max_)

        rich.print(f" - {general_utils.shorten_path(path)}")
        rich.print(f" - {general_utils.to_human_size(size)}")
        rich.print(f" - {file['predictions'].shape}")
        rich.print(f" - {mean = :.1f}")
        rich.print(f" - {std = :.1f}")
        rich.print(f" - {max_ = }")
        print()
        assert max_ == file['predictions'].shape[2]

    print()
    rich.print("[bold]Means:")
    rich.print(means)
    print()
    rich.print("[bold]Stds:")
    rich.print(stds)
    print()
    rich.print("[bold]Maxes:")
    rich.print(maxes)


@beartype
def main(
    directory: Union[Path, str] = SCRIPT_DIR / "log_results" / "oracle",
    max_epochs: int = 60,
):
    general_utils.check_and_print_args(locals().copy(), main)

    directory = Path(directory)
    assert directory.exists(), directory

    h5_paths = general_utils.sort_iterable_text(list(directory.glob("**/predictions.h5")))
    print()
    rich.print(f"[bold]All paths: [/bold]({len(h5_paths)})")
    general_utils.print_list(h5_paths)
    print()

    assert h5_paths
    bad_ones = detect_bads(h5_paths, num_epochs=max_epochs)
    good_ones = set(h5_paths) - bad_ones

    print()
    rich.print(f"[bold]Good ones: [/bold]({len(good_ones)}/{len(h5_paths)})")
    # general_utils.print_list(general_utils.sort_iterable_text(good_ones))
    print()
    rich.print(f"[bold]Bad ones: [/bold]({len(bad_ones)}/{len(h5_paths)})")
    general_utils.print_list(general_utils.sort_iterable_text(bad_ones))
    print()

    dont_have_label_ids = set()
    keys = []
    for path in h5_paths:
        with h5py.File(path, "r") as file:
            keys.extend(file.keys())
            if H5_LABEL_IDS_KEY not in file:
                dont_have_label_ids.add(path)


    rich.print(f"[bold]Keys:")
    general_utils.print_dict(dict(Counter(keys).items()))

    print()
    rich.print(
        f"[bold]Don't have label_ids: [/bold]"
        f"({len(dont_have_label_ids)}/{len(h5_paths)})"
    )
    general_utils.print_list(general_utils.sort_iterable_text(dont_have_label_ids))
    print()

    

if __name__ == "__main__":
    fire.Fire(main)