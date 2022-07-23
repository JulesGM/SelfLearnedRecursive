#!/usr/bin/env python3
# coding: utf-8

from beartype import beartype
import hashlib
from pathlib import Path
import pickle
import random
import time
from typing import *

import fire  # type: ignore[import]
import numpy as np
import rich

import general_utils

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]


SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"


def md5sum(filename: Union[str, Path]):
    with open(filename, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

@beartype
def read_subset_file(
    target_dataset: Union[str, Path], filepath: Union[str, Path]
):

    with open(filepath, "r") as fin:
        subset_conf = json.load(fin)

    md5_read = subset_conf["orig_pkl_md5"]
    md5sum_computed = md5sum(target_dataset)
    if md5_read != md5sum_computed:
        raise ValueError(
            f"The md5sum of the original dataset does "
            f"not match the md5sum of the subset file.\n"
            f"Original: {md5_read}\n"
            f"Subset: {md5sum_computed}"
        )

    # Convert the keys to ints.
    for entry_type in ["subset_ids", "subset_str"]:
        new_dict = {}
        for k in subset_conf[entry_type]:
            new_dict[int(k)] = subset_conf[entry_type][k]
        subset_conf[entry_type] = new_dict

    return subset_conf["subset_ids"], subset_conf["subset_str"]


@beartype
def main(
    # qty_desired_per_Level: int = 5000,
    # target: Union[str, Path] = DATA_DIR / "349_6_6_10000.json.pkl",
    
    qty_desired_per_Level: int = 10000,
    target: Union[str, Path] = DATA_DIR / "349_6_6_200000.json.pkl",

    seed: int = 453345,
):
    general_utils.check_and_print_args(locals().copy(), main)

    random.seed(seed)
    np.random.seed(seed)
    target = Path(target)  # type: ignore[no-redef]
    assert target.exists(), target

    print("Loading the file.")
    start = time.perf_counter()
    with open(target, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded the file in {time.perf_counter() - start} seconds.")

    rich.print(f"{obj.keys() = }")
    rich.print(f"{obj['config'] = }")
    rich.print(f"{obj['data'].keys() = }")
    rich.print(f"{obj['data']['eval'].keys() = }")
    for key in obj["data"]["eval"]:
        assert isinstance(key, int), type(key)

    lengths = {k: len(v) for k, v in obj["data"]["eval"].items()}
    
    for key in lengths:
        assert isinstance(key, int), type(key)

    rich.print(f"eval {lengths = }")
    rich.print(obj["data"]["eval"][6][0]["input_str"])

    subset_ids = {
        k: np.random.permutation(l)[:qty_desired_per_Level].tolist()
        for k, l in lengths.items()
    }

    for key in subset_ids:
        assert isinstance(key, int), type(key)

    subset_str = {}
    for level_no, level_content in obj["data"]["eval"].items():
        assert isinstance(level_no, int), type(level_no)
        subset_str[level_no] = [
            level_content[idx]["input_str"] for idx in subset_ids[level_no]
        ]

    for key in subset_str:
        assert isinstance(key, int), type(key)

    without_exts = target.name.split(".", 1)[0]
    output_target = (
        target.parent
        / "subsets"
        / f"subset_{qty_desired_per_Level}_seed_{seed}_of_{without_exts}.json"
    )
    print(f"{output_target = }")

    original_json_path = target.parent / target.stem
    output_dict = dict(
        subset_ids=subset_ids,
        subset_str=subset_str,
        original_qties=lengths,
        seed=seed,
        orig_pkl_md5=md5sum(target),
        orig_json_md5=md5sum(original_json_path) if original_json_path.exists() else None,
        orig_pkl_path=str(target),
        orig_json_path=str(original_json_path) if original_json_path.exists() else None,
    )

    with open(output_target, "w") as fout:
        json.dump(output_dict, fout)


if __name__ == "__main__":
    fire.Fire(main)
