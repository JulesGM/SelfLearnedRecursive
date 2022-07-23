#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import re
import sys
from typing import *

from beartype import beartype
import collections
import fire  # type: ignore[import]
import h5py  # type: ignore[import]
import numpy as np
import pickle
import rich
import time
from tqdm import tqdm  # type: ignore[import]

import data_generation_arithmetic
import data_tokenizer
import general_utils
import script_data_subset_selection

SCRIPT_DIR = Path(__file__).absolute().parent
DATA_DIR = SCRIPT_DIR / "data"

H5_INPUT_IDS_KEY = "input_samples"
H5_LABEL_IDS_KEY = "label_ids"
H5_PREDICTIONS_KEY = "predictions"
MAIN_DATASET_EVAL_KEY = "eval"
MAIN_DATASET_DATA_KEY = "data"
NODE_VALUE_STR_KEY = "value"
NODE_INPUT_STR_KEY = "input_str"
SUBSET_INPUT_IDS_KEY = "subset_ids"

PathType = Union[str, Path]

def find_last(str_: str, char: str) -> int:
    assert len(char) == 1, f"\"{char}\""
    return - 1 - str_[::-1].find(char)


def tokenize_pad_numpify(tokenizer, strings, key=None):
    
    if key:
        take_fn = lambda x: x[key]
    else:
        take_fn = lambda x: x

    tokenized = [tokenizer.encode(take_fn(x), no_eos=False, return_tensors=None) for x in tqdm(strings)]
    max_len = max(len(x) for x in tokenized)
    padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in tokenized]

    return np.array(padded)


def main(
    predictions_h5_path: PathType = SCRIPT_DIR / "log_results/oracle/oracle_1/predictions.h5",
    subset_path: PathType = DATA_DIR / "subsets/subset_10000_seed_453345_of_349_6_6_200000.json", 
    data_path: PathType = DATA_DIR / "349_6_6_200000.json.pkl",
):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Argument checking
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    general_utils.check_and_print_args(locals().copy(), main)

    predictions_h5_path = Path(predictions_h5_path)
    subset_path = Path(subset_path)
    data_path = Path(data_path)

    assert predictions_h5_path.exists(), f"{predictions_h5_path} does not exist"
    assert predictions_h5_path.is_file(), f"Path {predictions_h5_path} is not a file"
    assert predictions_h5_path.suffix == ".h5", f"Path {predictions_h5_path} is not a .h5 file"

    predictions = h5py.File(predictions_h5_path, "r+")
    if H5_LABEL_IDS_KEY in predictions:
        del predictions[H5_LABEL_IDS_KEY]

    if H5_PREDICTIONS_KEY in predictions:
        del predictions[H5_PREDICTIONS_KEY]
    # assert H5_INPUT_IDS_KEY in predictions, (
    #     f"{predictions_h5_path} does not have an '{H5_INPUT_IDS_KEY}' "
    #     f"dataset. {list(predictions.keys())}"
    # )

    assert subset_path.exists(), f"Path {subset_path} does not exist"
    assert subset_path.is_file(), f"Path {subset_path} is not a file"
    assert subset_path.suffix == ".json", f"Path {subset_path} is not a json file"

    subset_size = sum(len(x) for x in json.loads(subset_path.read_text()
    )[SUBSET_INPUT_IDS_KEY].values())
    assert subset_size == predictions[H5_INPUT_IDS_KEY].shape[0], (
        f"{subset_path} has a different number of samples than {predictions_h5_path}"
    )

    assert data_path.exists(), f"Path {data_path} does not exist"
    assert data_path.is_file(), f"Path {subset_path} is not a file"
    assert data_path.suffix == ".pkl", f"Path {subset_path} is not a pkl file"
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the eval ds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("[bold]1. Loading the eval dataset ...")
    start = time.perf_counter()
    dataset_dict = pickle.loads(data_path.read_bytes())
    end = time.perf_counter()
    print(f"Loaded the pkl dataset in {end - start:.2f} seconds")
    valid_ds = dataset_dict[MAIN_DATASET_DATA_KEY][MAIN_DATASET_EVAL_KEY]
    del dataset_dict

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the subset file and apply it to the eval ds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("\n[bold]2. Loading the subset file and applying it to the eval dataset ...")
    subset_indices, subset_str = script_data_subset_selection.read_subset_file(
        data_path, subset_path
    )
    valid_ds_subset = {}
    for level, nodes in tqdm(valid_ds.items(), desc="Applying the subset"):
        assert isinstance(level, int)
        valid_ds_subset[level] = [nodes[idx] for idx in subset_indices[level]]
    del valid_ds, subset_indices, subset_str
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # extract the order of the nodes in the subset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("\n[bold]3. Extracting the order of the nodes in the subset ...")
    subset_per_key = {}
    for level, node_list in valid_ds_subset.items():
        for node in node_list:
            subset_per_key[node[NODE_INPUT_STR_KEY]] = node
    sorted_by_keys = dict(sorted(subset_per_key.items(), key=lambda item: item[0]))
    del subset_per_key

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make sure the order is the same in the predictions h5 file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("\n[bold]4. Make sure the order is the same in the predictions h5 file ...")
    tokenizer = data_tokenizer.ArithmeticTokenizer()
    input_ids = tokenize_pad_numpify(tokenizer, sorted_by_keys.keys())

    if np.all(predictions[H5_INPUT_IDS_KEY][:] == 0):
        del predictions[H5_INPUT_IDS_KEY]
        predictions.create_dataset(H5_INPUT_IDS_KEY, data=input_ids)
        added_inputs = True
    else:
        assert np.all(predictions[H5_INPUT_IDS_KEY][:] == input_ids), (
            f"{predictions_h5_path} has a different order of nodes than {subset_path}"
        )
        added_inputs = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the labels and tokenize them, then save them.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich.print("\n[bold]5. Compute the labels and tokenize them, then save them ...")
    labels_np = tokenize_pad_numpify(
        tokenizer, sorted_by_keys.values(), NODE_VALUE_STR_KEY)
    predictions.create_dataset(H5_LABEL_IDS_KEY, data=labels_np)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check accuracy
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if added_inputs:
        levels = [data_generation_arithmetic.tree_depth_from_str(x) for x in sorted_by_keys.values()]
        assert predictions[H5_PREDICTIONS_KEY].shape[1] ==  len(predictions[H5_LABEL_IDS_KEY]), (
            predictions[H5_PREDICTIONS_KEY].shape[1],  len(predictions[H5_LABEL_IDS_KEY]),
        )

        for epoch in range(predictions[H5_PREDICTIONS_KEY].shape[0]):
            good_bad_dict = collections.defaultdict(list)
            good_bad_all = []
        
            for pred, label, level in tqdm(
                zip(predictions[H5_PREDICTIONS_KEY][epoch], predictions[H5_LABEL_IDS_KEY], levels), 
                total=len(predictions[H5_PREDICTIONS_KEY][epoch]),
                desc=f"Epoch {epoch} - Checking accuracy",
            ):
                pred_decoded = tokenizer.decode(pred, ignore_special_symbols=True).replace(" ", "")
                
                if "=" in pred_decoded:
                    # Find last equal
                    pred_decoded_after_equal = pred_decoded[find_last(pred_decoded, "=") + 1:]
                    # Find last number
                    pred_decoded_digits_list = re.findall(r"\d+", pred_decoded_after_equal)
                    # There might not be a number
                    if pred_decoded_digits_list:
                        pred_decoded_digits = pred_decoded_digits_list[0]
                        pred_label = tokenizer.decode(label, ignore_special_symbols=True).replace(" ", "")
                        good = int(pred_decoded_digits == pred_label)
                        good_bad_dict[epoch].append(good)
                        good_bad_all.append(good)

            accuracies = {level: np.mean(vals) for level, vals in sorted(good_bad_dict.items(), key=lambda item: item[0])}
            rich.print(f"[bold green]{epoch} - Accuracy: ")
            rich.print(f"\t- {accuracies}")
            rich.print(f"\t- {np.mean(good_bad_all):.1%}")
    rich.print(f"Done with \"{predictions_h5_path}\"")
    predictions.close()

if __name__ == "__main__":
    fire.Fire(main)