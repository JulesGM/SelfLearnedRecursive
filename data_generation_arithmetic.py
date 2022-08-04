#!/usr/bin/env python
# coding: utf-8
"""
Generate a dataset for the artihmetic dataset.

Usage: data_generation_arithmetic.py main [kwargs]

Args:

    --max_depth (default 6): 
        How deep the equations can go. This guarantees that at 
    least one path from root to leaf will be that long.

    --max_total_length (default 349):
        The maximum length of the *label* in number of tokens. 
    If a generated entry is shorter, it will be ignored and a
    new one will be generated.

    --max_answer_length (default 6):
        The maximum length of a value in number of tokens.
    This can be the final value or the intermediate value.

    --max_qty_per_level (default 200000):
        Number of data points to generate per equation depth level.

The dataset can be loaded with `load_dataset`.

If you just want to experiment with this dataset, you should probably
pre-tokenize the entries and save them to HDF5, calling
`Node.get_input_str` to get the equations, and `Node.get_value` on
the top level strings to get non recursive values.

"""

import collections
import copy
import dataclasses
import logging
import math
from pathlib import Path
import pickle
import time
from typing import *

from beartype import beartype
import fire  # type: ignore[import]
import numpy as np
import torch

try:
    import pretty_traceback  # type: ignore
    pretty_traceback.install()
except ImportError:
    pass

import random
import rich
from tqdm import tqdm  # type: ignore
import json

import general_utils
import data_tokenizer


LOGGER = logging.getLogger(__name__)
DEBUG = True
SCRIPT_DIR = Path(__file__).absolute().parent

OPMAP: Final[dict[str, Callable[[Union[str, int], Union[str, int]], str]]] = {
    "+": lambda x, y: str(int(x) + int(y)),
    "*": lambda x, y: str(int(x) * int(y)),
    "-": lambda x, y: str(int(x) - int(y)),
}

###############################################################################
# Parse max tree depths from strings or ids
###############################################################################
def tree_depth_from_str(tree_string: str) -> int:
    """
    Computes the depth of an equation tree of the dataset.
    Does a cumulative sum where `(` are ones and `)` are -1,
    and returns the max of the resulting array.

    The `from_ids` version is *considerably* faster, 
    being vectorized.
    """

    assert isinstance(tree_string, str)

    bool_test = np.fromiter(
        (char == "(" for char in tree_string if char in ["(", ")"]), np.int64
    )
    cum_sum = np.cumsum(2 * bool_test - 1)
    assert cum_sum[-1] == 0
    val = np.max(cum_sum)

    return val


def tree_depth_from_ids(
    ids: Union[np.ndarray, list, "torch.Tensor"], 
    tokenizer: data_tokenizer.ArithmeticTokenizer,
):
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().detach().numpy()
    
    if not isinstance(ids, np.ndarray):
        ids = tokenizer.pad_array(ids)

    ids = cast(np.ndarray, ids)

    assert ids.dtype == np.int64, ids.dtype

    lparen_idx = tokenizer.token_to_idx["("]
    rparen_idx = tokenizer.token_to_idx[")"]
    
    up_downs = ids.copy()
    up_downs[ids == lparen_idx] = 1
    up_downs[ids == rparen_idx] = -1
    up_downs[np.logical_and(ids != lparen_idx, ids != rparen_idx)] = 0
    counts = up_downs.cumsum(axis=-1)

    return counts.max(axis=-1)


###############################################################################
# Utilities to load the dataset and prep the entries for training or inference.
# prep_input_data used to have more stuff I believe.
###############################################################################
@beartype
def prep_input_data(
    tokenizer: data_tokenizer.Tokenizer,
    input_str: str,
    pseudo_without_head: str,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids = tokenizer(input_str, return_tensors="np", no_eos=False)
    decoder_input_ids = tokenizer(pseudo_without_head, return_tensors="np", no_eos=False)

    # The following is to silence type checkers. Only done once, it's ok.
    assert isinstance(input_ids, np.ndarray), type(input_ids)  
    assert isinstance(decoder_input_ids, np.ndarray), type(decoder_input_ids)

    return input_ids, decoder_input_ids


def load_dataset(
    json_path: Union[str, Path], pkl_path: Union[str, Path], splits=None,
) -> tuple[DefaultDict[str, DefaultDict[int, list["Node"]]], "EquationConfig"]:
    LOGGER.debug(f"Reading and parsing dataset from {json_path} or from {pkl_path}")

    ACCEPTABLE_SPLITS = {"train", "eval"}
    if splits is None:
        splits = ACCEPTABLE_SPLITS

    ###########################################################################
    # Either load the dataset from a json file or from a pickle file.
    # -------------------------------------------------------------------------
    # The data should be identical, the data in the pickle file is a dict of
    # basic types (the same as in the json file). We use pickle because
    # it was faster in practice.
    ###########################################################################
    dicts: dict[str, Any]
    if pkl_path:
        rich.print(f'[bold]Loading PKL data file:[/] "{pkl_path}"\n')
        with open(pkl_path, "rb") as f:
            dicts = pickle.load(f)
    else:
        rich.print(f'[bold]Loading JSON data file:[/] "{json_path}"\n')
        assert json_path
        with open(json_path, "r") as f:
            dicts = json.loads(f.read())
        
        # Convert the keys to ints
        for split in dicts:
            for level in dicts[split]:
                dicts[split][int(level)] = dicts[split][level]
                del dicts[split][level]

    rich.print("Done loading file.")

    ###########################################################################
    # Parse the dataset.
    ###########################################################################
    rich.print(f"Parsing structures from the dicts.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do a sanity check about the structure of the data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data : dict[str, dict[int, list[dict[str, Any]]]]= dicts["data"]
    assert isinstance(data, dict)
    assert set(data.keys()) == ACCEPTABLE_SPLITS, (data.keys(), ACCEPTABLE_SPLITS)

    for split in data.values():  # type: ignore[assignment]
        print(split.keys())  # type: ignore[attr-defined]  # The levels.
        # spit is a dict of levels to list of nodes.
        assert isinstance(split, dict)
        # split[1] is a list of nodes
        assert isinstance(split[1], list)
        # split[1][0] is a Node
        assert isinstance(split[1][0], dict)
        assert "op" in split[1][0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the nodes of the dataset.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("Building nodes")
    output: DefaultDict[str, DefaultDict[int, list[Node]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for split in splits:
        split_data = data[split]
        for level_idx, level_list in tqdm(
            split_data.items(), desc=f"Building nodes for {split}"
        ):
            for node_dict in tqdm(level_list, desc=f"building level {level_idx} nodes"):
                output[split][level_idx].append(Node.from_json_dict(node_dict))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the config object.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    config = dicts["config"]
    config_obj = EquationConfig.from_json_dict(config)


    LOGGER.debug(f"Done loading dataset.")
    
    return output, config_obj


###############################################################################
# Node class
###############################################################################
class Node:
    __slots__ = (
        "_children",
        "_complexity_level",
        "_input_str",
        "_pseudo_value",
        "_op",
        "_oracle_str",
        "_oracle_without_top_val",
        "_value",
        "_root_complexity_level",
    )

    def __copy__(self):
        raise RuntimeError("Nodes are not shallow-copyable.")

    def __deepcopy__(self, memo):

        assert self._root_complexity_level is None, (
            "self._root_complexity_level is not None, this is suspiscious behavior. "
            "It is not currently conserved by the deep copy. It could be though. "
        )
        assert (
            self._pseudo_value is None
        ), "self._pseudo_value is not None, this is suspiscious."

        return Node(
            op=self._op,
            children=[copy.deepcopy(c, memo) for c in self._children]
            if self._children
            else None,
            value=self._value,
            complexity_level=self._complexity_level,
        )

    def __init__(
        self,
        op: Optional[str],
        children: Optional[List["Node"]],
        value: str,
        complexity_level: int,
    ):
        self._op: Optional[str] = op
        self._children: List["Node"] = [] if children is None else children
        self._value = str(value)
        self._input_str: Optional[str] = None
        self._oracle_str: Optional[str] = None
        self._oracle_without_top_val: Optional[str] = None
        self._pseudo_value: Optional[int] = None
        self._complexity_level: Optional[int] = complexity_level
        self._root_complexity_level: Optional[int] = None

    def get_complexity_level(self) -> int:
        assert self._complexity_level is not None, "Complexity level is not set."
        assert isinstance(self._complexity_level, int), type(self._complexity_level)
        return self._complexity_level

    def get_root_complexity_level(self) -> int:
        assert isinstance(self._root_complexity_level, int), type(self._root_complexity_level)
        return self._root_complexity_level

    def set_complexity_level(self, value: int) -> None:
        assert self._complexity_level is None
        self._complexity_level = value

    def set_root_complexity_level(self, value: int) -> None:
        assert self._root_complexity_level is None
        self._root_complexity_level = value

    def reset_pseudo_values(self) -> None:
        self._pseudo_value = None
        if self.get_children():
            for child in self.get_children():
                child.reset_pseudo_values()

    def to_json_dict(self) -> Dict[str, Any]:
        oracle_str, oracle_without_top_val = self.get_oracle_str()
        assert self.get_root_complexity_level() is not None
        return {
            "op": self.get_op(),
            "children": [child.to_json_dict() for child in self.get_children()]
            if self.get_children()
            else None,
            "value": self.get_value(),
            "input_str": self.get_input_str(),
            "oracle_str": oracle_str,
            "oracle_without_top_val": oracle_without_top_val,
            "complexity_level": self.get_complexity_level(),
            "root_complexity_level": self.get_root_complexity_level(),
        }
    
    def get_ident(self) -> str:
        """ Return a unique identifyer.
        `input_str`s are unique identifiers and always should be. 
        The only preoccupation is the efficiency of the equality funciton. We'll 
        see about that but it shouldn't be too bad.
        `Node` objects cache their get_input_str, it's not lazily computed, so 
        that part is pretty fast. 
        """
        return self.get_input_str()

    @classmethod
    def from_json_dict(cls, json_dict) -> "Node":
        node = cls(
            op=json_dict["op"],
            children=(
                [
                    Node.from_json_dict(child_json)
                    for child_json in json_dict["children"]
                ]
                if json_dict["children"]
                else None
            ),
            value=json_dict["value"],
            complexity_level=json_dict["complexity_level"],
        )
        node._root_complexity_level = json_dict["root_complexity_level"]
        node._input_str = json_dict["input_str"]
        node._oracle_str = json_dict["oracle_str"]
        node._oracle_without_top_val = json_dict["oracle_without_top_val"]
        return node

    def get_op(self) -> str:
        assert isinstance(self._op, str)
        return self._op

    def get_children(self) -> list["Node"]:
        return self._children

    def get_value(self) -> str:
        assert isinstance(self._value, str), type(self._value)
        return self._value

    def get_input_str(self) -> str:
        # Multiple calls should always return the same thing
        if self._input_str is None:
            if self.get_children():
                a = self.get_children()[0].get_input_str()
                b = self.get_children()[1].get_input_str()
                assert len(self.get_children()) == 2, len(self.get_children())
                self._input_str = f"({a} {self.get_op()} {b})"
            else:
                self._input_str = f"{self.get_value()}"
        return self._input_str

    def get_oracle_str(self) -> tuple[str, str]:
        """
        The situation is that if we do generation with the scratch pad fed in,
        we need to be given the left part of the final top most equation without
        the right value.
        """
        if self._oracle_str is None or self._oracle_without_top_val is None:
            if self.get_children():
                assert len(self.get_children()) == 2, len(self.get_children())
                a_str, _ = self.get_children()[0].get_oracle_str()
                b_str, _ = self.get_children()[1].get_oracle_str()
                self._oracle_str = f"({a_str} {self._op} {b_str} = {self.get_value()})"
                self._oracle_without_top_val = f"({a_str} {self._op} {b_str} = "
            else:
                self._oracle_str = f"{self.get_value()}"
                self._oracle_without_top_val = ""

        return self._oracle_str, self._oracle_without_top_val

    def get_pseudo_topsort_query(self) -> str:
        """
        The situation is that if we do generation with the scratch pad fed in,
        we need to be given the left part of the final top most equation without
        the right value.
        """
        if self.get_children():
            assert len(self.get_children()) == 2, len(self.get_children())
            assert self.get_children()[0].get_pseudo_value() is not None
            assert self.get_children()[1].get_pseudo_value() is not None

            a_str = self.get_children()[0].get_pseudo_topsort()
            assert a_str
            b_str = self.get_children()[1].get_pseudo_topsort()
            assert b_str
            _pseudo_without_top_val = f"( {a_str} {self._op} {b_str} = "
        else:
            # If we don't have children, then we use the real value
            _pseudo_without_top_val = str(self.get_value())

        return _pseudo_without_top_val

    def get_pseudo_topsort(self):
        """
        The situation is that if we do generation with the scratch pad fed in,
        we need to be given the left part of the final top most equation without
        the right value.
        """
        query = self.get_pseudo_topsort_query()
        if self.get_children():
            return f"{query}{self.get_pseudo_value()} )"
        else:
            return self.get_value()

    def set_pseudo_value(self, value) -> None:
        assert self.get_children()
        self._pseudo_value = value

    def get_pseudo_value(self) -> str:
        if not self.get_children():
            return self.get_value()
        
        assert isinstance(self._pseudo_value, str), type(self._pseudo_value)
        return self._pseudo_value

    def get_pseudo(
        self,
        head_type: str,
        conc_mode: str,
        logging_info: "PredLogger",
        tokenizer: data_tokenizer.Tokenizer,
    ):
        """

        head_types are eaither "pred" or "oracle".
        The `head` being the right side of the top most equation.
        We want a "pred" head if we are composing the pseudo label tree.
        We want an "oracle" head at the very top, during training.
        We return "pseudo_without_head" for generation when the scratch pad is fed in.

        """
        assert head_type in ["pred", "oracle"]
        if self.get_children():
            if conc_mode == "yield":
                a_str, _, masked_pseudo_a = yield from self.get_children()[
                    0
                ].get_pseudo(
                    head_type="pred",
                    conc_mode=conc_mode,
                    logging_info=logging_info,
                    tokenizer=tokenizer,
                )
                b_str, _, masked_pseudo_b = yield from self.get_children()[
                    1
                ].get_pseudo(
                    head_type="pred",
                    conc_mode=conc_mode,
                    logging_info=logging_info,
                    tokenizer=tokenizer,
                )
            elif conc_mode == "top_sort":
                a_str, _, masked_pseudo_a = self.get_children()[0].get_pseudo(
                    head_type="pred",
                    conc_mode=conc_mode,
                    logging_info=logging_info,
                    tokenizer=tokenizer,
                )
                b_str, _, masked_pseudo_b = self.get_children()[1].get_pseudo(
                    head_type="pred",
                    conc_mode=conc_mode,
                    logging_info=logging_info,
                    tokenizer=tokenizer,
                )
            else:
                raise ValueError(f"Unknown conc_mode: {conc_mode}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Fetch the answer to the problem, either predicted or oracle
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            pseudo_without_head = f"({a_str} {self.get_op()} {b_str} = "
            if head_type == "oracle":
                head = self.get_value()
            elif head_type == "pred":
                if conc_mode == "yield":
                    head = yield dict(
                        input_str=self.get_input_str(),
                        pseudo_without_head=pseudo_without_head,
                        logging_info=logging_info,
                    )
                else:
                    raise ValueError(f"Unknown conc_mode: {conc_mode}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Tokenize it
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            pseudo_str = f"({a_str} {self.get_op()} {b_str} = {head})"
            potential_head = tokenizer(
                str(head),
                return_tensors=None,
                no_eos=True,
                strip_special_symbols=True,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Log the accuracy of the different levels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if head_type == "pred":
                tokenized_oracle_head = tokenizer(
                    str(self.get_value()),
                    return_tensors=None,
                    no_eos=True,
                    strip_special_symbols=True,
                )
                logging_info.level_accuracy[self._complexity_level].count_total += 1
                if potential_head == tokenized_oracle_head:
                    logging_info.level_accuracy[self._complexity_level].count_good += 1

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Prepare the label, where the scratchpad should be masked
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if head_type == "pred":
                maybe_head_tokens = [-100] * len(potential_head) + [-100]
            elif head_type == "oracle":
                tokenized = tokenizer(")", return_tensors=None, no_eos=True)
                assert isinstance(tokenized, list), type(tokenized)
                assert isinstance(potential_head, list), type(potential_head)
                maybe_head_tokens = potential_head + tokenized
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

            if head_type == "oracle":
                equal_token_ids = tokenizer("=", return_tensors=None, no_eos=True)
            else:
                equal_token_ids = len(tokenizer("=", return_tensors=None, no_eos=True)) * [-100]

            masked_pseudo = (
                len(tokenizer("(", return_tensors=None, no_eos=True)) * [-100]
                + masked_pseudo_a
                + len(tokenizer(self.get_op(), return_tensors=None, no_eos=True)) * [-100]
                + masked_pseudo_b
                + equal_token_ids
                + maybe_head_tokens
            )

        else:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If we are of the zeroth level, just return the real,
            # & don't alter the accuracy statistics (obviously).
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            pseudo_str = f"{self.get_value()}"
            pseudo_without_head = f""
            masked_pseudo = [-100] * len(tokenizer(pseudo_str, return_tensors=None, no_eos=True))

        assert isinstance(pseudo_str, str), type(pseudo_str)
        assert isinstance(pseudo_without_head, str), type(pseudo_without_head)

        return pseudo_str, pseudo_without_head, masked_pseudo

    def __repr__(self) -> str:
        return f"Node({self.get_oracle_str()})"


###############################################################################
# Node utility functions
###############################################################################
def get_all_desc(root: "Node") -> Generator["Node", None, None]:
    """
    Get all descendants of a node.
    Travels in depth first order.
    """
    work_stack = [root]
    while work_stack:
        node = work_stack.pop()
        yield node

        if node.get_children():
            work_stack.extend(node.get_children())


def multiple_get_all_desc(iterable: Iterable["Node"]) -> Generator["Node", None, None]:
    """
    Get all descendants of an iterable containnig nodes.
    """
    for node in iterable:
        yield from get_all_desc(node)


def all_nodes_have_complexity_levels(all_nodes: Iterable["Node"]) -> bool:
    """
    Check if all nodes have a complexity level.
    """
    for node in all_nodes:
        if node.get_complexity_level() is None:
            return False
    return True


def all_nodes_have_unique_ids(all_nodes: Iterable["Node"]) -> bool:
    """
    Name says it all.
    """
    seen = set()
    for node in all_nodes:
        if id(node) in seen:
            return False
        seen.add(id(node))
    return True


def all_nodes_have_root_complexity_levels(all_nodes: Iterable["Node"]) -> bool:
    for node in all_nodes:
        if node.get_root_complexity_level() is None:
            return False
    return True


def set_childrens_root_complexity_level(root: Node) -> None:
    assert root.get_complexity_level()
    root_complexity_level = root.get_complexity_level()
    for node in get_all_desc(root):
        node.set_root_complexity_level(root_complexity_level)


###############################################################################
# Data generation functions.
###############################################################################
def generate(
    config,
    previous: list,
    all_previous: list,
    qty_required: Union[int, str],
    complexity_level: int,
    name: str = "",
    filter_lambda: Optional[Callable] = None,
) -> Generator["Node", None, None]:
    """
    Where the data generation actually happens.

    qty_each_op is per direction
    """
    if qty_required == "all":
        for op in config.operators:
            LOGGER.debug(f"({name}): Doing all {len(previous) * len(all_previous)}")
            uniques = set()

            for a in previous:
                for b in all_previous:
                    uniques.add((a, b))
                    uniques.add((b, a))

            for a, b in uniques:
                a = copy.deepcopy(a)
                b = copy.deepcopy(b)
                new_node = Node(
                    op=op,
                    children=[a, b],
                    value=OPMAP[op](a.get_value(), b.get_value()),
                    complexity_level=complexity_level,
                )
                if filter_lambda is not None:
                    if filter_lambda(new_node):
                        yield new_node
                else:
                    yield new_node

        return

    else:
        assert isinstance(qty_required, int)
        qty_each_op = math.ceil(qty_required / (len(config.operators)))
        qty_per_side = math.ceil(qty_each_op / 2)

        for op_no, op in enumerate(config.operators):
            uniques = set()

            for side in ["right", "left"]:
                if side == "right":
                    a_s = np.random.choice(previous, qty_per_side, replace=True)
                    b_s = np.random.choice(all_previous, qty_per_side, replace=True)
                elif side == "left":
                    a_s = np.random.choice(all_previous, qty_per_side, replace=True)
                    b_s = np.random.choice(previous, qty_per_side, replace=True)
                else:
                    raise ValueError(side)

                uniques.update(zip(a_s, b_s))
                rich.print(
                    f"[red]({name}): {op_no + 1}/{len(config.operators)} {side = } {qty_per_side = }, "
                    f"{len(uniques) = }, {len(uniques) / qty_each_op = :0.1%}"
                )

            while len(uniques) < qty_each_op:
                for side in ["right", "left"]:
                    if side == "right":
                        a_s = np.random.choice(previous, qty_per_side, replace=True)
                        b_s = np.random.choice(all_previous, qty_per_side, replace=True)
                    elif side == "left":
                        a_s = np.random.choice(all_previous, qty_per_side, replace=True)
                        b_s = np.random.choice(previous, qty_per_side, replace=True)
                    else:
                        raise ValueError(side)

                    # We want to add just the right quantity of values. Computing the
                    # contains operator twice is a bit awkward, but it's not a big deal
                    # (the whole datagen takes under 15 seconds)
                    good_ones = [(a, b) for a, b in zip(a_s, b_s) if a not in uniques]
                    uniques.update(good_ones[: qty_each_op - len(uniques)])
                    LOGGER.debug(f"[attempt] {qty_each_op} {len(uniques)}")

            uniqe_roots = []
            for sub_node_a, sub_node_b in tqdm(uniques, desc="building nodes"):
                # This compying is necessary because we save the depth of the
                # node inside of the node, and the depth of a node will depend
                # on what tree it belongs to, so we can't just reuse the nodes.
                sub_node_a = copy.deepcopy(sub_node_a)
                sub_node_b = copy.deepcopy(sub_node_b)
                new_node = Node(
                    op=op,
                    children=[sub_node_a, sub_node_b],
                    value=OPMAP[op](sub_node_a.get_value(), sub_node_b.get_value()),
                    complexity_level=complexity_level,
                )
                uniqe_roots.append(new_node)

            if filter_lambda is not None:
                yield from filter(filter_lambda, uniqe_roots)
            else:
                yield from uniqe_roots


@beartype
def filter_length(
    root: Node,
    max_len_answer: Optional[int],
    max_len_total: Optional[int],
    tokenizer: data_tokenizer.Tokenizer,
) -> bool:

    complete_str = root.get_oracle_str()[0]
    complete_tokens = tokenizer(complete_str, return_tensors=None, no_eos=True)

    # Check complete length
    if max_len_total is not None and len(complete_tokens) > max_len_total:
        return False

    # Check answer length
    if max_len_answer is None:
        return True

    for node in get_all_desc(root):
        answer = node.get_value()
        answer_tokens = tokenizer(answer, return_tensors=None, no_eos=True)
        if len(answer_tokens) > max_len_answer:
            return False

    return True


class FilterLengthFunctor:
    def __init__(self, max_len_answer, max_len_total, tokenizer):
        self.max_len_answer = max_len_answer
        self.max_len_total = max_len_total
        self.tokenizer = tokenizer

    def __call__(self, root):
        return filter_length(
            root, self.max_len_answer, self.max_len_total, self.tokenizer
        )


def zeroth_level(config):
    ###########################################################################
    # Prepare the node for the zero-th level
    # Zeroth level nodes are a special case because they're the same
    # for both sets. (& don't have subnodes)
    ###########################################################################
    train = [
        Node(
            op=None,
            children=None,
            value=value,
            complexity_level=0,
        )
        for value in range(10 ** config.max_digits)
    ]
    eval = copy.deepcopy(train)
    return dict(train=train, eval=eval)


def first_level_and_more(
    config: "EquationConfig",
    qty: int,
    complexity_level: int,
    split_previous,
    all_split_previous,
    filter_length_lambda,
):
    ###########################################################################
    # Level 3 nodes.
    # Level 3 nodes are the first nodes where it isn't a special case in any
    # way. For a set, the nodes of the level are generated from the nodes of
    # previous levels, respecting the sets.
    ###########################################################################
    new = {}
    for split_name, split_previous_roots in split_previous.items():
        new[split_name] = list(
            generate(
                config=config,
                previous=split_previous_roots,
                all_previous=all_split_previous[split_name],
                qty_required="all" if complexity_level <= 2 else int(qty * 1.25),
                complexity_level=complexity_level,
                filter_lambda=filter_length_lambda,
                name=f"{split_name} {complexity_level}",
            )
        )

        random.shuffle(new[split_name])
        new[split_name] = new[split_name][:qty]

    return new


def generate_data(
    config: "EquationConfig",
) -> dict[str, dict[int, Any]]:
    tokenizer = data_tokenizer.ArithmeticTokenizer()
    filter_length_lambda = FilterLengthFunctor(
        config.max_answer_length, config.max_total_length, tokenizer
    )

    zeroth_l = zeroth_level(config)

    per_set = {split: {0: zeroth_l[split]} for split in ["train", "eval"]}
    for level in range(1, config.max_depth + 1):
        new = first_level_and_more(
            config=config,
            qty=config.max_qty_per_level,
            complexity_level=level,
            split_previous={
                split: per_set[split][level - 1] for split in ["train", "eval"]
            },
            all_split_previous={
                split: general_utils.concat_lists(per_set[split].values())
                for split in ["train", "eval"]
            },
            filter_length_lambda=filter_length_lambda,
        )

        for split, nodes in new.items():
            per_set[split][level] = nodes

    # We don't save zeroth level nodes. There's nothing to learn from them.
    for per_level in per_set.values():
        del per_level[0]

    ###########################################################################
    # Fix the lengths of the second layer.
    # Needs to be done after level 3 because we don't want to needlessly
    # throw away equations that could be used in level 3.
    ###########################################################################
    for k in per_set:
        per_set[k][2] = per_set[k][2][: config.max_qty_per_level]

    ################################################################################
    # Make some checks on all root nodes, fix root_level_complexity
    ################################################################################
    print("Building a list of all nodes.")
    all_root_nodes = general_utils.concat_lists(
        general_utils.concat_lists(
            [[v for v in tqdm(x.values())] for x in tqdm(per_set.values())]
        )
    )
    all_nodes = list(multiple_get_all_desc(all_root_nodes))

    if DEBUG:
        print("Doing some checks.")
        assert len(set(all_nodes)) == len(all_nodes), "Duplicate nodes"
        assert all_nodes_have_unique_ids(all_nodes)
        assert all_nodes_have_complexity_levels(all_nodes)

    print("Setting root_level_complexity.")
    for top_level_node in tqdm(all_root_nodes):
        set_childrens_root_complexity_level(top_level_node)

    if DEBUG:
        print("Final check")
        assert all_nodes_have_root_complexity_levels(all_nodes)

    ################################################################################
    # Print the lengths
    ################################################################################
    print("Final lengths:")
    lengths = {}
    for cv_set_name, cv_set_per_level in tqdm(per_set.items()):
        lengths[cv_set_name] = {
            level_idx: len(level_nodes)
            for level_idx, level_nodes in cv_set_per_level.items()
        }

    rich.print(lengths)
    return per_set


class PredLogger:
    """
    Logger object for the self learned mode.
    """
    __slots__ = (
        # "root",
        "level_accuracy",
    )

    @dataclasses.dataclass
    class Accuracy:
        count_good: int = 0
        count_total: int = 0

        def compute(self):
            return f"{self.count_good}/{self.count_total}, {self.count_good / self.count_total:.1%}"

    def __init__(
        self,
    ):
        self.level_accuracy = collections.defaultdict(self.Accuracy)

    def log(self):
        for k, v in self.level_accuracy.items():
            rich.print(f"[bright_cyan]Level {k}[/]: {v.compute()}")


class EquationConfig:
    # misc

    @beartype
    def __init__(
        self,
        *,
        max_depth: int,
        max_total_length: int,
        max_answer_length: int,
        max_qty_per_level: int,
    ):
        # Checks
        assert max_total_length
        assert (
            max_total_length < 1024
        ), "This is the length of the whole context of BART, it's the hardest limit."

        # Values coming from args
        self.max_depth: Final[int] = max_depth
        self.max_total_length: Final[int] = max_total_length
        self.max_qty_per_level: Final[int] = max_qty_per_level
        self.max_answer_length: Final[int] = max_answer_length

        # Hardcoded or computed values
        self.seed: Final[int] = 1337
        self.operators: Final[set[str]] = {"+", "*", "-"}
        self.max_digits: Final[int] = 1
        self.output_name: Final[
            str
        ] = f"{self.max_total_length}_{self.max_answer_length}_{self.max_depth}_{self.max_qty_per_level}.json"

    def to_json_dict(self) -> dict[str, Any]:
        base_dict = vars(self)
        base_dict["operators"] = list(self.operators)
        return base_dict

    @classmethod
    def from_json_dict(cls, json_dict) -> "EquationConfig":

        json_dict["operators"] = set(json_dict["operators"])

        config_obj = cls(
            max_depth=json_dict["max_depth"],
            max_total_length=json_dict["max_total_length"],
            max_answer_length=json_dict["max_answer_length"],
            max_qty_per_level=json_dict["max_qty_per_level"],
        )
        obj_dict: Final[dict[str, Any]] = vars(config_obj)

        # Verify that the dicts are the same.
        if not obj_dict == json_dict:
            assert obj_dict.keys() == json_dict.keys(), (
                obj_dict.keys() - json_dict.keys(),
                json_dict.keys() - obj_dict.keys(),
            )
            different_values = {}
            # Here the keys are guaranteed to be the same.
            for k in obj_dict:
                if obj_dict[k] != json_dict[k]:
                    different_values[k] = (obj_dict[k], json_dict[k])
            assert not different_values, different_values

        return config_obj


###############################################################################
# The script's entry point functions, `main` and `test`.
###############################################################################
class EntryPoints:
    @staticmethod
    @beartype
    def main(
        max_depth: int = 6,
        max_total_length: int = 349,
        max_answer_length: int = 6,
        max_qty_per_level: int = 200000,
    ):
        """
        Generate a dataset for the artihmetic dataset.

        Usage: data_generation_arithmetic.py main [kwargs]

        Args:

            --max_depth (default 6): 
                How deep the equations can go. This guarantees that at 
            least one path from root to leaf will be that long.

            --max_total_length (default 349):
                The maximum length of the *label* in number of tokens. 
            If a generated entry is shorter, it will be ignored and a
            new one will be generated.

            --max_answer_length (default 6):
                The maximum length of a value in number of tokens.
            This can be the final value or the intermediate value.

            --max_qty_per_level (default 200000):
                Number of data points to generate per equation depth level.

        """
        # create basic config for logging
        logging.basicConfig(level=logging.DEBUG)

        config = EquationConfig(
            max_depth=max_depth,
            max_total_length=max_total_length,
            max_answer_length=max_answer_length,
            max_qty_per_level=max_qty_per_level,
        )

        per_set = generate_data(config)

        print("Building the dict object that will be saved.")
        dataset: dict[str, dict[str, Any]] = {"data": {}, "config": config.to_json_dict()}
        data = dataset["data"]
        for split, nodes_per_level in per_set.items():
            data[split] = {}
            split_dict = data[split]
            for level_name, nodes_per_level in nodes_per_level.items():
                assert level_name != 0, level_name
                split_dict[level_name] = [
                    sample.to_json_dict()
                    for sample in tqdm(
                        nodes_per_level, desc=f"Split {split} Level {level_name}"
                    )
                ]

        print("Saving the data.")
        start = time.perf_counter()
        with open(SCRIPT_DIR / "data" / f"{config.output_name}.pkl", "bw") as f:
            pickle.dump(dataset, f)
        print(f"Pickled dicts in {time.perf_counter() - start:0.2f} seconds")

        start = time.perf_counter()
        with open(SCRIPT_DIR / "data" / config.output_name, "w") as f:
            f.write(json.dumps(dataset, indent=4))
        print(f"Dumped json in {time.perf_counter() - start:0.2f} seconds")


    @staticmethod
    def test():
        tokenizer = data_tokenizer.ArithmeticTokenizer()
        
        class Entry:
            equation_str: str
            ids: List[int]
            answer: int

            def __init__(self, equation_str, answer, tokenizer):
                self.equation_str = equation_str
                self.answer = answer
                self.ids = tokenizer.encode(equation_str)

        

        entries = [
            Entry(
                "(((1 + 2) * 3) - (3 + (2 - 1)))",
                3,
                tokenizer
            ),
            Entry(
                "((1 + 2 * 3) - (3 + 2 - 100) + (3 * 2) * (1  + ((1 - 2) + -333 )))",
                4,
                tokenizer
            ),
            Entry(
                "((1  + ((1 - (2 + (3 - 10))) + -3 ) * (1 + 2 * 3) - (3 + 2 - 1) + (3 * 2)))",
                6,
                tokenizer
            ),
        ]       

        for entry in entries:
            from_str = tree_depth_from_str(entry.equation_str)
            from_ids = tree_depth_from_ids(entry.ids, tokenizer)
            assert entry.answer == from_ids, (entry.answer, from_ids)
            assert entry.answer == from_str, (entry.answer, from_str)
    
        for _ in range(10):
            shuffled = entries.copy()
            random.shuffle(shuffled)

            ids = [x.ids for x in shuffled]
            ids = tokenizer.pad_array(id)

            answers = np.array([x.answer for x in shuffled], dtype=np.int64)
            from_ids_array = tree_depth_from_ids(
                ids, 
                tokenizer
            )
            
            assert np.all(from_ids_array == answers), (
                f"{from_ids_array = }, {answers = }"
            )


if __name__ == "__main__":
    fire.Fire(EntryPoints)
