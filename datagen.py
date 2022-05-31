#!/usr/bin/env python
# coding: utf-8

import collections
import concurrent.futures as futures
import dataclasses
import itertools
import logging
import math
from pathlib import Path
import pickle
import time
from typing import *

from beartype import beartype
import pretty_traceback
pretty_traceback.install()
import numpy as np
import random
import rich
from tqdm import tqdm
from typing import *
import ujson as json

import our_tokenizer

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).absolute().parent

opmap = {
    "+": lambda x, y: x + y,
    "*": lambda x, y: x * y,
    "-": lambda x, y: x - y,
    "/": lambda x, y: x // y,
}


@beartype
def prep_input_data(
    tokenizer: our_tokenizer.Tokenizer,
    input_str: str,
    pseudo_without_head: str,
) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = tokenizer.encode(input_str, "np", no_eos=False)
    decoder_input_ids = tokenizer.encode(pseudo_without_head, "np", no_eos=False)
    return input_ids, decoder_input_ids


def load_dataset(json_path: str, pkl_path: str):
    LOGGER.debug(f"Reading and parsing dataset from {json_path} or from {pkl_path}")

    if pkl_path:
        with open(pkl_path, "rb") as f:
            dicts = pickle.load(f)
    else:
        assert json_path
        with open(json_path, "r") as f:
            dicts = json.load(f)

    LOGGER.debug(f"Parsing structures from the dicts.")
    top_progress = tqdm(dicts.items())

    per_split = collections.defaultdict(dict)
    for level_name, splits in top_progress:
        for split_name, split in splits.items():
            top_progress.set_description(f"Parsing {level_name}")
            per_split[split_name][level_name] = []
            for node_dict in tqdm(split, desc=level_name):
                per_split[split_name][level_name].append(Node.from_json_dict(node_dict))

    LOGGER.debug(f"Done loading dataset.")
    return per_split


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
    )

    def __init__(
        self,
        op: Optional[str],
        children: Optional[List["Node"]],
        value: int,
        complexity_level: int,
    ):
        self._op: str = op
        self._children: List["Node"] = children
        self._value = value
        self._input_str: Optional[str] = None
        self._oracle_str: Optional[str] = None
        self._oracle_without_top_val: Optional[str] = None
        self._pseudo_value: Optional[int] = None
        self._complexity_level: Optional[int] = complexity_level

    def get_complexity_level(self) -> int:
        return self._complexity_level

    def set_complexity_level(self, value: int) -> None:
        assert self._complexity_level is None
        self._complexity_level = value

    def reset_pseudo_values(self) -> None:
        self._pseudo_value = None
        if self.get_children():
            for child in self.get_children():
                child.reset_pseudo_values()

    def to_json_dict(self) -> Dict[str, Any]:
        oracle_str, oracle_without_top_val = self.get_oracle_str()
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
        }

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
        node._input_str = json_dict["input_str"]
        node._oracle_str = json_dict["oracle_str"]
        node._oracle_without_top_val = json_dict["oracle_without_top_val"]
        return node

    def get_op(self) -> str:
        return self._op

    def get_children(self) -> "Node":
        return self._children

    def get_value(self):
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

    def get_oracle_str(self) -> str:
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
        assert False
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
            _pseudo_without_top_val = ""

        return _pseudo_without_top_val

    def get_pseudo_topsort(self):
        """
        The situation is that if we do generation with the scratch pad fed in,
        we need to be given the left part of the final top most equation without
        the right value.
        """
        assert False
        assert self.get_pseudo_value()
        query = self.get_pseudo_topsort_query()
        if self.get_children():
            return f"{query}{self.get_pseudo_value()} )"
        else:
            return self.get_value()

    def set_pseudo_value(self, value) -> None:
        assert self.get_children()
        self._pseudo_value = value

    def get_pseudo_value(self) -> str:
        if self.get_children() is None:
            return self.get_value()
        return self._pseudo_value

    def get_pseudo(
        self,
        head_type: str,
        conc_mode: str,
        logging_info: "PredLogger",
        tokenizer: our_tokenizer.Tokenizer,
    ):
        """

        head_types are eaither "pred" or "oracle".
        The `head` being the right side of the top most equation.
        We want a "pred" head if we are composing the pseudo label tree.
        We want an "oracle" head at the very top, during training.
        We return "pseudo_without_head" for generation when the scratch pad is fed in.

        """
        assert head_type in ["pred", "oracle"]
        if self.get_children() is not None:
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
                maybe_head_tokens = potential_head + tokenizer(")", None, no_eos=True)
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

            if head_type == "oracle":
                equal_token_ids = tokenizer("=", None, no_eos=True)
            else:
                equal_token_ids = len(tokenizer("=", None, no_eos=True)) * [-100]

            masked_pseudo = (
                len(tokenizer("(", None, no_eos=True)) * [-100]
                + masked_pseudo_a
                + len(tokenizer(self.get_op(), None, no_eos=True)) * [-100]
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
            masked_pseudo = [-100] * len(tokenizer(pseudo_str, None, no_eos=True))

        assert isinstance(pseudo_str, str), type(pseudo_str)
        assert isinstance(pseudo_without_head, str), type(pseudo_without_head)

        return pseudo_str, pseudo_without_head, masked_pseudo

    def __repr__(self) -> str:
        return f"Node({self.get_oracle_str()})"


def generate(
    config,
    previous: list,
    all_previous: list,
    qty_required: int,
    name: str = "",
    filter_lambda: Optional[Callable] = None,
) -> "Node":
    """
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
                yield Node(
                    op=op,
                    children=[a, b],
                    value=opmap[op](a.get_value(), b.get_value()),
                    complexity_level=None,
                )
        return

    else:
        assert isinstance(qty_required, int)
        qty_each_op = math.ceil(qty_required / (len(config.operators)))
        qty_per_side = math.ceil(qty_each_op / 2)

        for op in config.operators:
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
                    f"[red]({name}): {side = } {qty_per_side = }, "
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
                    good_ones = [x for x in zip(a_s, b_s) if x not in uniques]
                    if filter_lambda:
                        good_ones = list(filter(filter_lambda, good_ones))
                    uniques.update(good_ones[: qty_each_op - len(uniques)])
                    LOGGER.debug(f"[attempt] {qty_each_op} {len(uniques)}")

            for a, b in uniques:
                yield Node(
                    op=op,
                    children=[a, b],
                    value=opmap[op](a.get_value(), b.get_value()),
                    complexity_level=None,
                )


def filter_length(
    root: Node,
    max_len_answer: int,
    max_len_total: int,
    tokenizer: our_tokenizer.Tokenizer,
) -> bool:
    complete_str = root.get_oracle_str()[0]
    complete_tokens = tokenizer(complete_str, None, no_eos=True)
    if len(complete_tokens) > max_len_total:
        return False

    work_stack = [root]
    while work_stack:
        node = work_stack.pop()

        # Process current node
        answer = str(node.get_value())
        answer_tokens = tokenizer(answer, None, no_eos=True)
        if len(answer_tokens) > max_len_answer:
            return False

        # Process children
        maybe_children = node.get_children()
        if maybe_children:
            work_stack.extend(maybe_children)

    return True


def generate_data(
    config: "NestedClacConfig",
) -> Tuple[List[Node], List[Node], List[Node]]:
    zeroth_l = [
        Node(op=None, children=None, value=value, complexity_level=0)
        for value in range(10 ** config.max_digits)
    ]

    ###########################################################################
    # Prepare the nodes of the first level
    ###########################################################################
    first_l: List[Node] = []
    # We want all the entries of level 1
    for op in config.operators:
        for a in zeroth_l:
            for b in zeroth_l:
                first_l.append(
                    Node(
                        op=op,
                        children=[a, b],
                        value=opmap[op](a.get_value(), b.get_value()),
                        complexity_level=1,
                    )
                )
    
    # There are way too few level 1 equations, we use them on both 
    # validation splits. (There are just 300 total).
    # We may want to oversample these, too.
    first_all = first_l
    first_l = dict(train=first_l, eval=first_l)

    ###########################################################################
    # Prepare, filter and split all the level 2 eqns
    ###########################################################################    
    all_second_l = list(
        generate(
            config,
            first_all,
            first_all + zeroth_l,
            "all",
            "ALL SECONDS",
        )
    )
    for node in all_second_l:
        node.set_complexity_level(2)

    tokenizer = our_tokenizer.Tokenizer(512, True)
    filter_length_lambda = lambda node: filter_length(
        node, config.max_answer_length, config.max_total_length, tokenizer
    )
    random.shuffle(all_second_l)
    print("l2 all pre-filter", len(all_second_l))
    all_second_l = list(filter(filter_length_lambda, all_second_l))
    print("l2 all post-filter", len(all_second_l))
    second_l = dict(
        train=all_second_l[: len(all_second_l) // 2],
        eval=all_second_l[len(all_second_l) // 2:]
    )

    ###########################################################################
    # Prepare, filter and split all the level 3 eqns
    ###########################################################################
    third_l = {}
    for split_name, split_l2_roots in second_l.items():
        third_l[split_name] = list(
            generate(
                config,
                split_l2_roots,
                split_l2_roots + first_all + zeroth_l,
                config.qty_third_layer * 2,
                "THIRD",
                filter_lambda=filter_length_lambda,
            )
        )
        for node in third_l[split_name]:
            node.set_complexity_level(3)
        print(f"l3 Pre  filter: {len(third_l[split_name])}")
        third_l[split_name] = list(filter(filter_length_lambda, third_l[split_name]))
        random.shuffle(third_l[split_name])
        third_l[split_name] = third_l[split_name][: config.qty_third_layer]
        print(f"l3 Post filter: {len(third_l[split_name])}")

    ###########################################################################
    # Fix the lengths of the second layer.
    # Needs to be done after level 3 because we don't want to needlessly
    # throw away level two equations that could be used in level 3.
    ###########################################################################
    for k in second_l:
        second_l[k] = second_l[k][: config.qty_second_layer]

    ################################################################################
    # Make sure each node has a complexity level
    # Stack based depth first traversal
    ################################################################################
    work_stack: List[Node] = list(
        itertools.chain(
            itertools.chain(*first_l .values()), 
            itertools.chain(*second_l.values()),
            itertools.chain(*third_l .values()),
        ),
    )

    while work_stack:
        current_node = work_stack.pop()
        if current_node.get_children():
            work_stack.extend(current_node.get_children())
        assert current_node.get_complexity_level() is not None


    ################################################################################
    # Print the lengths
    ################################################################################
    lengths_l1 = {k: len(v) for k, v in first_l .items()}
    lengths_l2 = {k: len(v) for k, v in second_l.items()}
    lengths_l3 = {k: len(v) for k, v in third_l .items()}
    print("Final lengths:")
    print(f"\tfirst_l:  {lengths_l1}")
    print(f"\tsecond_l: {lengths_l2}")
    print(f"\tthird_l:  {lengths_l3}")

    return first_l, second_l, third_l


class PredLogger:
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


@dataclasses.dataclass
class NestedClacConfig:
    # misc
    seed: int = 1337
    output_name: str = "dataset.json"

    # dataset
    operators: Set[str] = dataclasses.field(default_factory=lambda: {"+", "*", "-"})
    max_depth = 3
    max_digits = 1
    qty_second_layer = 200000
    qty_third_layer = 200000

    max_answer_length = 4
    max_total_length = 88


if __name__ == "__main__":
    # create basic config for logging
    logging.basicConfig(level=logging.DEBUG)

    config = NestedClacConfig()
    first, second, third = generate_data(config)
    dataset_native = {
        "first": first,
        "second": second,
        "third": third,
    }

    dataset = dict(
        first ={split: [sample.to_json_dict() for sample in tqdm(samples)] for split, samples in first .items()},
        second={split: [sample.to_json_dict() for sample in tqdm(samples)] for split, samples in second.items()},
        third ={split: [sample.to_json_dict() for sample in tqdm(samples)] for split, samples in third .items()},
    )

    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / "dicts.pkl", "bw") as f:
        pickle.dump(dataset, f)
    print(f"Pickled dicts in {time.perf_counter() - start:0.2f} seconds")

    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / config.output_name, "w") as f:
        json.dump(dataset, f)
    print(f"Dumped json in {time.perf_counter() - start:0.2f} seconds")
