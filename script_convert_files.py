#!/usr/bin/env python
# coding: utf-8

import inspect
import os
import multiprocessing
from pathlib import Path
import time

import fire  # type: ignore[import]
import jsonlines as jsonl  # type: ignore[import]
import msgpack  # type: ignore[import]
import msgpack_numpy  # type: ignore[import]
import numpy as np
import rich
from tqdm import tqdm  # type: ignore[import]

from beartype import beartype
from beartype.typing import *

import general_utils

SCRIPT_DIR = Path(__file__).absolute().parent

msgpack_numpy.patch()


def convert(input_path):
    with jsonl.open(input_path) as f:
        content = [x for x in f]

    for i in range(len(content)):
        for k, v in content[i]["results"].items():
            content[i]["results"][k] = np.asarray(v["True"]["per_batch"], dtype=np.int64)

    with open(input_path.parent / "numpy_predictions.msgpack", "wb") as f:
        msgpack.pack(content, f)

@beartype
def main(
    path: Union[str, Path] = SCRIPT_DIR / "log_results/oracle/"
):

    all_arguments = locals().copy()
    assert all_arguments.keys() == inspect.signature(main).parameters.keys()
    rich.print("[bold]Arguments:")
    general_utils.print_dict(all_arguments)

    path = Path(path)
    directories = list(path.iterdir())
    active = []

    for file in tqdm(directories):
        target = file / "predictions.jsonl"
        if target.exists():
            active.append(target)

    rich.print("[bold]Directories:")
    rich.print(active)

    rich.print("[bold]Converting.")
    rich.print(f"{os.cpu_count() = }")
    start = time.perf_counter()
    with multiprocessing.Pool() as pool:
        list(pool.map(convert, active))
        pool.close()
        pool.join()
    duration = time.perf_counter() - start
    rich.print(f"[bold]Done in {duration:.2f} seconds.")
    rich.print(f" - This is {duration / os.cpu_count()} s/cpu.")
    rich.print(f" - This is {duration / len(active)} s/file")
    rich.print(f" - This is {duration / (os.cpu_count() * len(active))} s/(cpu * file)")



if __name__ == "__main__":
    fire.Fire(main)
