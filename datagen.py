#!/usr/bin/env python
# coding: utf-8
import collections
import copy
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

import utils
import our_tokenizer

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).absolute().parent

opmap = {
    "+": lambda x, y: str(int(x) + int(y)),
    "*": lambda x, y: str(int(x) * int(y)),
    "-": lambda x, y: str(int(x) - int(y)),
}

def get_all_desc(root: "Node") -> Generator["Node", None, None]:
    work_stack = [root]
    while work_stack:
        node = work_stack.pop()
        yield node
        
        if node.get_children():
            work_stack.extend(node.get_children())


def multiple_get_all_desc(iterable: Iterable["Node"]) -> Generator["Node", None, None]:
    for node in iterable:
        yield from get_all_desc(node)


def nodes_and_desc_have_complexity_levels(nodes: Iterable["Node"]) -> bool:
    for node in multiple_get_all_desc(nodes):
        if node.get_complexity_level() is None:
            return False
    return True
    

def nodes_and_desc_have_unique_ids(lst: Iterable["Node"]) -> bool:
    seen = set()
    for node in multiple_get_all_desc(lst):
        if id(node) in seen:
            return False
        seen.add(id(node))
    return True

@beartype
def prep_input_data(
    tokenizer: our_tokenizer.Tokenizer,
    input_str: str,
    pseudo_without_head: str,
) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = tokenizer.encode(input_str, "np", no_eos=False)
    decoder_input_ids = tokenizer.encode(pseudo_without_head, "np", no_eos=False)
    return input_ids, decoder_input_ids


def load_dataset(json_path: str, pkl_path: str) -> Tuple[Dict[str, Dict[int, "Node"]], "NestedClacConfig"]:
    LOGGER.debug(f"Reading and parsing dataset from {json_path} or from {pkl_path}")

    rich.print("[blue]Loading data file.")
    if pkl_path:
        rich.print(f"[blue]Loading data file {pkl_path}")
        with open(pkl_path, "rb") as f:
            dicts = pickle.load(f)
    else:
        rich.print(f"[blue]Loading data file {json_path}")
        assert json_path
        with open(json_path, "r") as f:
            dicts = json.load(f)
    rich.print("[blue]Done loading file.")

    LOGGER.debug(f"Parsing structures from the dicts.")
    config = dicts["config"]
    data = dicts["data"]
    
    assert isinstance(data, list)
    num_levels = len(data)
    assert isinstance(data[0], dict)
    assert isinstance(data[0]["train"], list)
    assert isinstance(data[0]["train"][0], dict)
    assert isinstance(data[0]["eval"], list)
    assert isinstance(data[0]["eval"][0], dict)
    assert "op" in data[0]["train"][0]
    assert "op" in data[0]["eval"][0]

    print("Inverting dict")
    per_split = utils.dict_unzip(data)
    print("Dict inverted")

    # Cheap sanity checks
    assert isinstance(per_split, dict)
    assert "train" in per_split
    assert "eval" in per_split
    assert isinstance(per_split["train"], list)
    assert isinstance(per_split["eval"], list)
    assert len(per_split["train"]) == num_levels
    assert len(per_split["eval"]) == num_levels
    assert "op" in per_split["train"][0][0]
    assert "op" in per_split["eval"][0][0]

    print("Building nodes")
    for split, split_data in per_split.items():
        for level_idx, level_list in enumerate(tqdm(split_data, desc=f"Building nodes for {split}")):
            for node_idx, node_dict in enumerate(level_list):
                per_split[split][level_idx][node_idx] = Node.from_json_dict(
                    node_dict
                )

    LOGGER.debug(f"Done loading dataset.")
    return per_split, NestedClacConfig.from_json_dict(config)


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
        assert self._pseudo_value is None, (
            "self._pseudo_value is not None, this is suspiscious."
        )

        return Node(
            op=self._op, 
            children=[copy.deepcopy(c, memo) for c in self._children] if self._children else None,
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
        self._op: str = op
        self._children: List["Node"] = children
        self._value = str(value)
        self._input_str: Optional[str] = None
        self._oracle_str: Optional[str] = None
        self._oracle_without_top_val: Optional[str] = None
        self._pseudo_value: Optional[int] = None
        self._complexity_level: Optional[int] = complexity_level
        self._root_complexity_level: Optional[int] = None

    def get_complexity_level(self) -> int:
        return self._complexity_level
    
    def get_root_complexity_level(self) -> int:
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
        return self._op

    def get_children(self) -> "Node":
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
    complexity_level: int,
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
                a = copy.deepcopy(a)
                b = copy.deepcopy(b)
                new_node = Node(
                    op=op,
                    children=[a, b],
                    value=opmap[op](a.get_value(), b.get_value()),
                    complexity_level=complexity_level,
                )

                if filter_lambda(new_node):
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
                    value=opmap[op](sub_node_a.get_value(), sub_node_b.get_value()), 
                    complexity_level=complexity_level
                )
                uniqe_roots.append(new_node)

            yield from filter(filter_lambda, uniqe_roots)


@beartype
def filter_length(
    root: Node,
    max_len_answer: Optional[int],
    max_len_total: Optional[int],
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
        answer = node.get_value()
        answer_tokens = tokenizer(answer, None, no_eos=True)
        if len(answer_tokens) > max_len_answer:
            return False

        # Process children
        maybe_children = node.get_children()
        if maybe_children:
            work_stack.extend(maybe_children)

    return True


def set_childrens_root_complexity_level(root: Node) -> None:
    assert root.get_complexity_level()
    root_complexity_level = root.get_complexity_level()
    for node in get_all_desc(root):
        node.set_complexity_level(root_complexity_level)


class FilterLengthFunctor:
    def __init__(self, max_len_answer, max_len_total, tokenizer):
        self.max_len_answer = max_len_answer
        self.max_len_total = max_len_total
        self.tokenizer = tokenizer

    def __call__(self, root):
        return filter_length(
            root, 
            self.max_len_answer, 
            self.max_len_total, 
            self.tokenizer
        )

def zeroth_level(config):
    ###########################################################################
    # Prepare the node for the zero-th level
    # Zeroth level nodes are a special case because they're the same
    # for both sets. (& don't have subnodes)
    ###########################################################################
    return [
        Node(op=None, children=None, value=value, complexity_level=0,)
        for value in range(10 ** config.max_digits)
    ]

def first_level(zeroth_l):
    ###########################################################################
    # Level 1 nodes.
    # Level one nodes are a special case because both the train and the eval
    # sets are the same, and because the eval and the train set of the zeroth
    # level are as well.
    ###########################################################################
    first_all: List[Node] = []
    # We want all the entries of level 1
    for op in config.operators:
        for a in zeroth_l:
            for b in zeroth_l:
                # The nodes will be told their depth as an optimization of the
                # sorting procedures, so they need to be distinct.
                sub_node_a = copy.deepcopy(a)
                sub_node_b = copy.deepcopy(b)
                first_all.append(
                    Node(
                        op=op,
                        children=[sub_node_a, sub_node_b],
                        value=opmap[op](a.get_value(), b.get_value()),
                        complexity_level=1,
                    )
                )

    # There are way too few level 1 equations, we use them on both
    # validation splits. (There are just 300 total).
    # We may want to oversample these, too.
    #
    # This is the only level where we copy the nodes. We can't just reuse
    # the same ones, because we have a bunch of checks to make sure no
    # node is used twice, that are useful in general. So we deep copy the nodes.
    first_l = dict(train=first_all, eval=copy.deepcopy(first_all))
    return first_l, first_all

def second_level(config, zeroth_l, first_all, filter_length_lambda):
    ###########################################################################
    # Level 2 nodes.
    # Level 2 is a special case because level 1 nodes are identical, so we
    # can't just call generate on the two sets of level 1 nodes.
    ###########################################################################
    # Here we generate from all the level 1 equations, because both sets 
    # are the same.
    
    all_second_l = list(
        generate(
            config=config,
            previous=first_all,
            all_previous=first_all + zeroth_l,
            qty_required="all",
            complexity_level=2,
            name="ALL SECONDS",
            filter_lambda=filter_length_lambda,
        )
    )

    random.shuffle(all_second_l)
    second_l = dict(
        train=all_second_l[:len(all_second_l) // 2][:config.qty_second_layer],
        eval=all_second_l [len(all_second_l) // 2:][:config.qty_second_layer],
    )
    return second_l

def third_level_and_more(qty, complexity_level, split_previous, all_split_previous, filter_length_lambda):
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
                qty_required=int(qty * 1.25),
                complexity_level=complexity_level,
                filter_lambda=filter_length_lambda,
                name=f"{split_name} {complexity_level}",
            )
        )
        
        random.shuffle(new[split_name])
        new[split_name] = new[split_name][: qty]

    return new


def generate_data(
    config: "NestedClacConfig",
) -> Tuple[List[Node], List[Node], List[Node]]:
    tokenizer = our_tokenizer.Tokenizer(512, True)
    filter_length_lambda = FilterLengthFunctor(
        config.max_answer_length, config.max_total_length, tokenizer
    )

    zeroth_l = zeroth_level(config)
    first_l, first_all = first_level(zeroth_l) 
    second_l = second_level(
        config=config, 
        zeroth_l=zeroth_l, 
        first_all=first_all, 
        filter_length_lambda=filter_length_lambda)
    n_th = [zeroth_l, first_l, second_l]
    del zeroth_l
    del first_l
    del first_all
    del second_l

    for level in range(3, config.max_depth + 1):
        new = third_level_and_more(
            config.qty_third_layer, level, n_th[-1], dict(
                train=sum((level_["train"] for level_ in n_th[1:]), []) + n_th[0],
                eval=sum((level_["eval"] for level_ in n_th[1:]), []) + n_th[0],
            ), filter_length_lambda
        )
        n_th.append(new)


    ###########################################################################
    # Fix the lengths of the second layer.
    # Needs to be done after level 3 because we don't want to needlessly
    # throw away level two equations that could be used in level 3.
    ###########################################################################
    for k in n_th[2]:
        n_th[2][k] = n_th[2][k][: config.qty_second_layer]

    ################################################################################
    # Make some checks on all root nodes, fix root_level_complexity
    ################################################################################
    all_root_nodes: List[Node] = []
    for nodes_splits in n_th[1:]:
        for nodes in nodes_splits.values():
            all_root_nodes.extend(nodes)

    assert len(set(all_root_nodes)) == len(all_root_nodes), "Duplicate nodes"
    assert nodes_and_desc_have_unique_ids(all_root_nodes)
    assert nodes_and_desc_have_complexity_levels(all_root_nodes)

    for top_level_node in all_root_nodes:
        set_childrens_root_complexity_level(top_level_node)      

    ################################################################################
    # Print the lengths
    ################################################################################
    lengths = {i: {k: len(v) for k, v in n_th[i].items()} for i in range(1, len(n_th))}
    print("Final lengths:")
    rich.print(lengths)

    return n_th


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


class NestedClacConfig:
    # misc
    
    def __init__(
        self, *, 
        max_depth: int, 
        max_total_length: int,
        max_answer_length: int,
    ):
        self.max_depth = max_depth
        self.max_total_length = max_total_length

        self.seed: int = 1337
        # dataset
        self.operators = {"+", "*", "-"}
        self.max_digits = 1
        self.qty_second_layer = 200000
        self.qty_third_layer = 200000
        self.max_answer_length = max_answer_length
        
        assert self.max_depth >= 3
        self.output_name = f"{self.max_total_length}_{self.max_answer_length}_{self.max_depth}.json"
        

    def to_json_dict(self):
        base_dict = vars(self)
        base_dict["operators"] = list(self.operators)
        return base_dict

    @classmethod
    def from_json_dict(cls, json_dict):
        
        obj = cls(
            max_depth=json_dict["max_depth"], 
            max_total_length=json_dict["max_total_length"],
            max_answer_length=json_dict["max_answer_length"],
        )
        json_dict["operators"] = set(json_dict["operators"])

        assert json_dict.keys() == vars(obj).keys(), (
            json_dict.keys() - vars(obj).keys(),  vars(obj).keys() - json_dict.keys()
        )
                
        for k in json_dict.keys():
            assert getattr(obj, k) == json_dict[k], (
                k, json_dict[k], getattr(obj, k)
            )

        return obj


if __name__ == "__main__":
    # create basic config for logging
    logging.basicConfig(level=logging.DEBUG)

    config = NestedClacConfig(max_depth=8, max_total_length=None, max_answer_length=None)
    n_th = generate_data(config)
    
    dataset = dict(
        data=[{
            split: [sample.to_json_dict() for sample in tqdm(samples)]
            for split, samples in entry.items()
        } for entry in n_th[1:]],
        config=config.to_json_dict(),
    )

    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / f"{config.output_name}.pkl", "bw") as f:
        pickle.dump(dataset, f)
    print(f"Pickled dicts in {time.perf_counter() - start:0.2f} seconds")

    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / config.output_name, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dumped json in {time.perf_counter() - start:0.2f} seconds")
