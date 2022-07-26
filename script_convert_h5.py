#!/usr/bin/env python
# coding: utf-8

"""
Converts the predictions.jsonl files generated by the training scripts to much faster h5 files.

The format of the H5 is:
    - "input_samples": num_samples x max_len_input_samples
    - "predictions": num_epochs x num_samples x max_len_predictions
The root of the H5 has an "epochs" attr, of the epochs. This is uselesss now, 
as epochs are just range(num_epochs) now.
"""

import collections
import concurrent.futures
import enum
import inspect
import os
import multiprocessing
import multiprocessing.pool as mp_pool
import queue as threading_queue
import itertools
from pathlib import Path
import queue
import subprocess
import time
import threading
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
import script_add_label_info_to_subset_h5
pretty_traceback.install()


SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"

PathType = Union[Path, str]

def _convert(
    input_path: Union[str, Path],
    tokenizer: data_tokenizer.ArithmeticTokenizer,
    input_ids: np.ndarray,
    max_epochs: int = None,
    verbose: bool = False,
    queue = None,
):
    start_time = time.process_time()
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

        assert tokenizer.pad_token_id == 0, tokenizer.pad_token_id
        
        output_file.create_dataset(
            "input_samples", 
            data=input_ids,
        )

        output_file.create_dataset(
            "predictions",
            shape=(num_epochs, num_samples, len_seqs_output),
            dtype=np.int64,
        )
        predictions = output_file["predictions"]
        
        if verbose:
            rich.print("Writing data.")
        
        num_samples = None
        epochs_seen_list: list[int] = []  
        for entry_idx in range(len(content)):
            real_epoch = content[entry_idx]["epoch"]

            ######################################################
            # DON'T REMOVE THIS CHECK
            assert real_epoch == entry_idx, (real_epoch, entry_idx)
            ######################################################

            # If we're after the 0th epoch
            # Then the saved keys should be the same as the keys of the current epoch
            for input_idx, k in enumerate(sorted_keys):
                # Make sure that we're only adding keys if we're in the zeroth epoch
                if real_epoch == 0:
                    tokenized = tokenizer.encode(k, return_tensors=None)
                    input_ids_gen = tokenized + [
                        tokenizer.pad_token_id
                    ] * max(input_ids[input_idx].shape[0] - len(tokenized), 0)
                    
                    assert np.all(input_ids[input_idx] == input_ids_gen)

                # The predictions are already encoded
                prediction = content[entry_idx]["results"][k]
                predictions[real_epoch, input_idx] = prediction + [
                    tokenizer.pad_token_id
                ] * max(len_seqs_output - len(prediction), 0)

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
    
    return time.process_time() - start_time

class _ConvertFunctor:
    def __init__(self, tokenizer, max_epochs, input_ids, label_ids):
        self._tokenizer = tokenizer
        self._max_epochs: Optional[int] = max_epochs
        self._input_ids = input_ids
        if max_epochs:
            rich.print(f"[red]{max_epochs = }")

    def __call__(self, input_path: Union[str, Path], queue=None, verbose=False):
        return _convert(
            input_path=input_path, 
            tokenizer=self._tokenizer, 
            max_epochs=self._max_epochs, 
            input_ids=self._input_ids,
            verbose=verbose, 
            queue=queue,
        )


class LaunchMethods(str, enum.Enum):
    launch_all = "launch_all"
    launch_few = "launch_few"
    launch_one = "launch_one"
    launch_pool = "launch_pool"

class ThreadOrProcess(str, enum.Enum):
    thread = "thread"
    process = "process"

@beartype
def main(
    path: Union[str, Path] = SCRIPT_DIR / "log_results/oracle/",
    n_cpus: int = 12,
    test_run: bool = False,
    method=LaunchMethods.launch_pool,
    max_epochs: Optional[int] = 60,
    thread_or_process: str = ThreadOrProcess.process,
    subset_path: PathType = DATA_DIR / "subsets/subset_10000_seed_453345_of_349_6_6_200000.json", 
    data_path: PathType = DATA_DIR / "349_6_6_200000.json.pkl",
    
):
    general_utils.check_and_print_args(locals().copy(), main)

    assert data_path.suffix == ".pkl", f"{data_path} is not a pickle file."
    assert subset_path.suffix == ".json", f"{subset_path} is not a json file."
    assert subset_path.exists(), f"{subset_path} does not exist."
    assert data_path.exists(), f"{data_path} does not exist."


    if thread_or_process == ThreadOrProcess.thread:
        thread_or_process_type = threading.Thread
    elif thread_or_process == ThreadOrProcess.process:
        thread_or_process_type = multiprocessing.Process
    else:
        raise ValueError(f"Unknown thread_or_process: {thread_or_process}, should be one of {[x.value for x in list(ThreadOrProcess)]}")

    method = LaunchMethods(method)
    tokenizer = data_tokenizer.ArithmeticTokenizer()

    #######################################################################
    # Prepare the paths
    #######################################################################
    TARGET_FILE_NAME = "predictions.jsonl"
    path = Path(path)
    active = sorted(path.glob(f"**/{TARGET_FILE_NAME}"))
    
    rich.print(f"[bold]File paths: {general_utils.shorten_path(path)}")
    for path in general_utils.sort_iterable_text(active):
        rich.print(f"\t- {general_utils.shorten_path(path)}")
    print()
    assert active, path

    rich.print("[bold]Converting.")
    rich.print(f"{n_cpus = }")
    n_cpus = min(n_cpus, len(active))  # Likely already done by multiprocessing.Pool

    ###########################################################################
    # Prep the inputs and labels
    ###########################################################################
    print()
    rich.print("[bold]Preparing inputs and labels.")
    sorted_by_keys = script_add_label_info_to_subset_h5.build_eval_subset_sorted_by_keys(data_path, subset_path)
    
    label_ids = script_add_label_info_to_subset_h5.tokenize_pad_numpify(
        tokenizer, sorted_by_keys.values(), script_add_label_info_to_subset_h5.NODE_VALUE_STR_KEY
    )

    print("3")
    input_ids = script_add_label_info_to_subset_h5.tokenize_pad_numpify(tokenizer, sorted_by_keys.keys())

    ###########################################################################
    # Do the multiprocessing
    ###########################################################################
    print()
    rich.print("[bold]Starting multiprocessing.")
    convert_functor = _ConvertFunctor(tokenizer, max_epochs=max_epochs, input_ids=input_ids, label_ids=label_ids)
    start = time.perf_counter()
    if not test_run:
        if method == LaunchMethods.launch_few:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Only launch n_cpus processes
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            processes: list[thread_or_process_type] = []  # type: ignore[no-redef]
            if thread_or_process_type == threading.Thread:
                queue = threading_queue.Queue(n_cpus)
            elif thread_or_process_type == multiprocessing.Process:
                queue = multiprocessing.Queue(n_cpus)
            else:
                raise ValueError(thread_or_process_type)

            [queue.put(None) for _ in range(n_cpus)]  # type: ignore[func-returns-value]

            for i, path in enumerate(tqdm(active, desc="Running jobs")):
                queue.get()
                print(f"Started {i}")
                process = thread_or_process_type(target=convert_functor, args=(path, queue))
                processes.append(process)
                process.start()

            for process in tqdm(processes, desc="Joining"):
                process.join()
                
        elif method == LaunchMethods.launch_pool:
            #######################################################################
            # Launch all processes
            #######################################################################
            if thread_or_process_type == threading.Thread:
                PoolType = mp_pool.ThreadPool
            elif thread_or_process_type == multiprocessing.Process:
                PoolType = multiprocessing.Pool
            
            promises = []
            times = []
            with PoolType(n_cpus) as pool:
                for path in active:
                    promises.append(pool.apply_async(convert_functor, (path,)))
                
                for promise in tqdm(promises, desc="Running conversion jobs."):
                    times.append(promise.get())

        elif method == LaunchMethods.launch_one:
            for path in tqdm(active):
                convert_functor(path, queue=None, verbose=True)

        else:
            raise ValueError(f"Unknown method: {method}")
    duration = time.perf_counter() - start
    print()
    rich.print("[bold]Done running jobs. Writing label_ids and input_ids.")

    def write_label_ids(input_path: Path):
        output_path = input_path.parent / f"{input_path.stem}.h5"

        with h5py.File(output_path, "r+") as f:
            f.create_dataset("label_ids", data=label_ids)

    with mp_pool.ThreadPool(n_cpus) as pool:
        promises = [pool.apply_async(write_label_ids, (path,)) for path in active]
        for promise in tqdm(promises, desc="Writing `label_ids`."):
            promise.get()

    #######################################################################
    # Print some stats
    #######################################################################
    duration_w_more_stuff = time.perf_counter() - start
    print("Done with multiprocessing.")
    rich.print(f"[bold]Done in {duration:.2f} seconds.")
    rich.print(f" - This is {duration / n_cpus} s/cpu.")
    rich.print(f" - This is {duration / len(active)} s/file")
    rich.print(f" - Average time of one file was {np.mean(times):.1f} seconds.")
    rich.print(f" - The linear time would have been {np.sum(times):.1f} seconds.")
    rich.print(f" - This is an improvement of {np.mean(times) / duration:0.1f} times.")
    print(f"{duration_w_more_stuff:.2f} seconds with more stuff.")

if __name__ == "__main__":
    fire.Fire(main)
