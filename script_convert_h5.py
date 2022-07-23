#!/usr/bin/env python
# coding: utf-8

import collections
import concurrent.futures
import enum
import inspect
import os
import multiprocessing
import itertools
from pathlib import Path
import queue
import subprocess
import time
import traceback

from beartype import beartype
from beartype.typing import *
import fire  # type: ignore[import]
import jsonlines as jsonl  # type: ignore[import]
import h5py  # type: ignore[import]
import pretty_traceback  # type: ignore
import numpy as np
import rich
from tqdm import tqdm  # type: ignore[import]

import data_tokenizer
import general_utils

pretty_traceback.install()


SCRIPT_DIR = Path(__file__).absolute().parent

def _convert(
    input_path: Union[str, Path],
    tokenizer: data_tokenizer.ArithmeticTokenizer,
    max_epochs: int = None,
    verbose: bool = False,
    queue = None,
):
    if verbose:
        print("Working.")


    ###########################################################################
    # Read the data
    ###########################################################################
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}.h5"

    if verbose:
        rich.print("[bold]Counting lines of the jsonl.")
    num_lines = int(
        subprocess.check_output(["wc", "-l", str(input_path)])
        .strip()
        .decode()
        .split()[0]
    )

    if max_epochs:
        assert max_epochs <= num_lines, (max_epochs, num_lines)
        target_qty = min(max_epochs, num_lines)
    else:
        target_qty = num_lines

    if verbose:
        rich.print(f"[bold]Fewer than {num_lines} lines.")
        print()

    if verbose:
        rich.print("[bold]Reading jsonl.")

    with jsonl.open(input_path) as f:
        iterable = f
        if max_epochs:
            iterable = itertools.islice(f, target_qty)

        if verbose:
            iterable = tqdm(iterable, total=target_qty)
        content = [x for x in iterable]

    del iterable
    del target_qty  # Not meant to be used again.
    del max_epochs  # Not meant to be used again.
    del num_lines  # Not meant to be used again.

    if verbose:
        rich.print("[bold]Done reading jsonl.")


    ###########################################################################
    # Prep the data
    ###########################################################################
    # We ignore pytorch lightning's "sanity test" tiny zeroth epoch.
    if content[0]["epoch"] == 0 and content[1]["epoch"] == 0:
        content = content[1:]
    
    # We make sure that all epochs have the same number of samples.
    assert all([len(content[i]) == len(content[0]) for i in range(1, len(content))]), (
        [len(content[i]) for i in range(len(content))]
    )

    # If an epoch happens twice, we remove the second one. 
    # This happens we think when training is interrupted and restarted, PL 
    # starts with an eval pass.
    new_content = []
    epochs_seen = set()
    for epoch in content:
        if epoch["epoch"] in epochs_seen:
            continue
        epochs_seen.add(epoch["epoch"])
        new_content.append(epoch)
    content = new_content

    for i in range(len(content)):
        content[i]["results"] = {
            k: content[i]["results"][k]["True"]["per_batch"] 
            for k in content[i]["results"]
        }

    sorted_keys: list[str] = list(content[0]["results"].keys())
    sorted_keys.sort()
    for epoch_content in content:
        assert sorted_keys == sorted(epoch_content["results"].keys())

    del epochs_seen  # Not meant to be used again.
    del new_content  # Not meant to be used again.
    

    ###########################################################################
    # Write the data
    ###########################################################################
    num_samples = len(content[1]["results"])
    num_epochs = len(content)
    len_seqs_output = max([
        max([
            len(samples)
            for samples in epochs["results"].values()
        ]) for epochs in content
    ])

    if verbose:
        rich.print(f"[bold]Doing h5py.")

    with h5py.File(output_path, "w") as output_file:
        if verbose:
            rich.print("Creating datasets.")

        output_file.create_dataset(
            "input_samples", 
            shape=(num_samples, len_seqs_output), 
            dtype=np.int64,
        )
        output_file.create_dataset(
            "predictions",
            shape=(num_epochs, num_samples, len_seqs_output),
            dtype=np.int64,
        )
        input_samples = output_file["input_samples"]
        predictions = output_file["predictions"]
        
        if verbose:
            rich.print("Writing data.")
        
        num_samples = None
        epochs_seen_list: list[int] = []  
        for entry_idx in range(len(content)):
            real_epoch = content[entry_idx]["epoch"]
            assert real_epoch == entry_idx, (real_epoch, entry_idx)

            # If we're after the 0th epoch
            # Then the saved keys should be the same as the keys of the current epoch
            for input_idx, k in enumerate(sorted_keys):
                # Make sure that we're only adding keys if we're in the zeroth epoch
                if input_idx >= len(sorted_keys):
                    assert real_epoch == 0, real_epoch

                    sorted_keys.append(k)
                    tokenized = tokenizer.encode(k, return_tensors=None)
                    input_samples[input_idx] = tokenized + [
                        tokenizer.pad_token_id
                    ] * max(len_seqs_output - len(tokenized), 0)

                value = content[entry_idx]["results"][k]
                predictions[real_epoch, input_idx] = value + [
                    tokenizer.pad_token_id
                ] * max(len_seqs_output - len(value), 0)

            epochs_seen_list.append(content[entry_idx]["epoch"])

        # Make sure that we only have one of each key
        assert len(set(epochs_seen_list)) == len(epochs_seen_list), collections.Counter(epochs_seen_list)
        assert epochs_seen_list == list(range(len(epochs_seen_list))), epochs_seen_list

        if verbose:
            rich.print("Writing attrs")

        predictions.attrs.create("epochs", epochs_seen_list, dtype=np.int64)
        
    if verbose:
        rich.print("[bold]Done h5py.")

    if queue:
        queue.put(None)


class _ConvertFunctor:
    def __init__(self, tokenizer, max_epochs):
        self._tokenizer = tokenizer
        self._max_epochs: Optional[int] = max_epochs
        if max_epochs:
            rich.print(f"[red]{max_epochs = }")

    def __call__(self, input_path: Union[str, Path], queue=None, verbose=False):
        return _convert(
            input_path=input_path, 
            tokenizer=self._tokenizer, 
            max_epochs=self._max_epochs, 
            verbose=verbose, 
            queue=queue,
        )

class ProcessWithException(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class LaunchMethods(str, enum.Enum):
    launch_all = "launch_all"
    launch_few = "launch_few"
    launch_one = "launch_one"

def handle_exceptions(processes):
    if any(process.exception for process in processes):
        rich.print("[bold red]Got at least an exception")
        for process in processes:
            if process.exception:
                rich.print(f"{vars(process.exception) = }")
            print()
    else:
        rich.print("[bold green]No exceptions")

@beartype
def main(
    path: Union[str, Path] = SCRIPT_DIR / "log_results/basic/",
    n_cpus: int = int(os.getenv("SLURM_CPUS_ON_NODE", os.cpu_count())),
    test_run: bool = False,
    method=LaunchMethods.launch_few,
    max_epochs: Optional[int] = 60,
    max_cpus: int = 6,
):
    general_utils.check_and_print_args(locals().copy(), main)

    method = LaunchMethods(method)
    tokenizer = data_tokenizer.ArithmeticTokenizer()

    #######################################################################
    # Prepare the paths
    #######################################################################
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
    rich.print(f"{n_cpus = }")
    n_cpus = min(min(n_cpus, len(active)), max_cpus)  # Likely already done by multiprocessing.Pool
    functor = _ConvertFunctor(tokenizer, max_epochs=max_epochs)

    print("Starting multiprocessing.")
    start = time.perf_counter()
    if not test_run:
        PROCESS_CLASS: TypeAlias = ProcessWithException

        if method == LaunchMethods.launch_all:
            #######################################################################
            # Launch all processes
            #######################################################################
            processes: list[PROCESS_CLASS] = []
            for path in active:
                process = PROCESS_CLASS(target=functor, args=(path, None,))
                processes.append(process)
                process.start()
            
            for process in tqdm(processes):
                process.join()

            handle_exceptions(processes)

        elif method == LaunchMethods.launch_few:
            #######################################################################
            # Only launch n_cpus processes
            #######################################################################
            processes: list[multiprocessing.Process] = []  # type: ignore[no-redef]
            queue = multiprocessing.Queue(n_cpus)  # type: ignore[var-annotated]
            [queue.put(None) for _ in range(n_cpus)]  # type: ignore[func-returns-value]

            for i, path in enumerate(tqdm(active)):
                queue.get()
                print(f"Started {i}")
                process = ProcessWithException(target=functor, args=(path, queue,))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            handle_exceptions(processes)
        elif method == LaunchMethods.launch_one:
            for path in tqdm(active):
                functor(path, queue=None, verbose=True)

        else:
            raise ValueError(f"Unknown method: {method}")


    #######################################################################
    # Print some stats
    #######################################################################
    duration = time.perf_counter() - start
    print("Done with multiprocessing.")
    rich.print(f"[bold]Done in {duration:.2f} seconds.")
    rich.print(f" - This is {duration / os.cpu_count()} s/cpu.")
    rich.print(f" - This is {duration / len(active)} s/file")
    rich.print(f" - This is {duration / (os.cpu_count() * len(active))} s/(cpu * file)")


if __name__ == "__main__":
    fire.Fire(main)
