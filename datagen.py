#!/usr/bin/env python
# coding: utf-8

import concurrent.futures as futures
import dataclasses
import logging
import math
from pathlib import Path
import pickle
import time
from typing import *

from beartype import beartype
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
    pseudo_without_head: str
) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = tokenizer(input_str)
    decoder_input_ids = tokenizer(pseudo_without_head)
    return input_ids, decoder_input_ids


def load_dataset(json_path, pkl_path):

    LOGGER.debug(f"Reading and parsing dataset from {json_path} or from {pkl_path}")

    if pkl_path:
        with open(pkl_path, "rb") as f:
            dicts = pickle.load(f)
    else:
        assert json_path
        with open(json_path) as f:
            dicts = json.load(f)
    
    LOGGER.debug(f"Parsing structures from the dicts.")
    dataset = {}
    top_progress = tqdm(dicts.items())
    for level_name, node_dict_list in top_progress:
        top_progress.set_description(f"Parsing {level_name}")
        dataset[level_name] = []
        for node_dict in tqdm(node_dict_list, desc=level_name):
            dataset[level_name].append(Node.from_json_dict(node_dict))
    
    LOGGER.debug(f"Done loading dataset.")
    return dataset

class Node:
    __slots__ = (
        "_op", 
        "_children", 
        "_value",
        "_input_str", 
        "_oracle_str", 
        "_oracle_without_top_val",
    )

    def __init__(
        self, 
        op: Optional[str], 
        children: Optional[List["Node"]], 
        value: int
    ):
        self._op = op
        self._children = children
        self._value = value
        self._input_str:  Optional[str] = None
        self._oracle_str: Optional[str] = None
        self._oracle_without_top_val: Optional[str] = None

    def to_json_dict(self):
        oracle_str, oracle_without_top_val = self.get_oracle_str()
        return {
            "op": self.get_op(),
            "children": [child.to_json_dict() for child in self.get_children()] if self.get_children() else None,
            "value": self.get_value(),
            "input_str": self.get_input_str(),
            "oracle_str": oracle_str,
            "oracle_without_top_val": oracle_without_top_val,
        }

    @classmethod
    def from_json_dict(cls, json_dict):
        node = cls(
            op=json_dict["op"],
            children=(
                [Node.from_json_dict(child_json) for child_json in json_dict["children"]] 
                if json_dict["children"] else None
            ),
            value=json_dict["value"],
        )
        node._input_str = json_dict["input_str"]
        node._oracle_str = json_dict["oracle_str"]
        node._oracle_without_top_val = json_dict["oracle_without_top_val"]
        return node

    def get_op(self):
        return self._op

    def get_children(self):
        return self._children

    def get_value(self):
        return self._value

    def get_input_str(self):
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

    def get_oracle_str(self):
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

    def get_pseudo(self, prediction_function: Optional[Callable], head_type: str, conc_mode: str, logging_info: "PredLogger"):
        """ 

        head_types are eaither "pred" or "oracle".
        The `head` being the right side of the top most equation.
        We want a "pred" head if we are composing the pseudo label tree.
        We want an "oracle" head at the very top, during training.
        We return "pseudo_without_head" for generation when the scratch pad is fed in.

        """
        # Multiple calls will have DIFFERENT RESULTS
        if conc_mode == "yield":
            assert prediction_function is None, (
                "Cannot use prediction function with yield mode"
            )


        assert head_type in ["pred", "oracle"]

        if self.get_children() is not None:
            if conc_mode == "pool":
                assert False
                with futures.ThreadPoolExecutor(max_workers=2) as pool:
                    a_prom = pool.submit(
                        self.get_children()[0].get_pseudo, 
                        prediction_function, 
                        head_type="pred", 
                        conc_mode=conc_mode, 
                        logging_info=logging_info
                    )
                    b_prom = pool.submit(
                        self.get_children()[1].get_pseudo, 
                        prediction_function, 
                        head_type="pred", 
                        conc_mode=conc_mode, 
                        logging_info=logging_info,
                    )
                    
                    a_str, _ = a_prom.result()
                    b_str, _ = b_prom.result()
            elif conc_mode == "yield":
                a_str, _ = yield from self.get_children()[0].get_pseudo(
                    None, 
                    head_type="pred", 
                    conc_mode=conc_mode, 
                    logging_info=logging_info
                )
                b_str, _ = yield from self.get_children()[1].get_pseudo(
                    None, 
                    head_type="pred", 
                    conc_mode=conc_mode, 
                    logging_info=logging_info
                )
            elif conc_mode is None:
                assert False
                a_str, _ = self.get_children()[0].get_pseudo(
                    prediction_function, 
                    head_type="pred", 
                    conc_mode=conc_mode, 
                    logging_info=logging_info
                )
                b_str, _ = self.get_children()[1].get_pseudo(
                    prediction_function, 
                    head_type="pred", 
                    conc_mode=conc_mode, 
                    logging_info=logging_info
                )
            else:
                raise ValueError(f"Unknown conc_mode: {conc_mode}")

            if head_type == "oracle" or head_type == "pred":
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
                        head = prediction_function(
                            self.get_input_str(), 
                            pseudo_without_head,
                            logging_info,    
                        )

                pseudo_str = f"({a_str} {self.get_op()} {b_str} = {head})"   
            else:
                raise ValueError(f"Unknown head_type: {head_type}")
        else:
            pseudo_str = f"{self.get_value()}"
            pseudo_without_head = f""
        
        assert isinstance(pseudo_str, str), type(pseudo_str)
        assert isinstance(pseudo_without_head, str), type(pseudo_without_head)
        return pseudo_str, pseudo_without_head


    def __repr__(self):
        return f"Node({self.get_oracle_str()})"

def generate(config, previous, all_previous, qty_required, name=""):
    """
    qty_each_op is per direction
    """

    if qty_required != "all":
        assert isinstance(qty_required, int) 
        qty_each_op = math.ceil(qty_required / (len(config.operators)))
        qty_per_side = math.ceil(qty_each_op / 2)


    if qty_required == "all":
        for op in config.operators:
            LOGGER.debug(f"({name}): Doing all {len(previous) * len(all_previous)}")
            uniques = set()

            for a in previous:
                for b in all_previous:
                    uniques.add((a, b))
                    uniques.add((b, a))  
            
            for a, b in uniques:
                yield Node(op=op, children=[a, b], value=opmap[op](a.get_value(), b.get_value()))
        return

    else: 
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
                    uniques.update(good_ones[:qty_each_op - len(uniques)])
                    LOGGER.debug(f"[attempt] {qty_each_op} {len(uniques)}")

            for a, b in uniques:
                yield Node(op=op, children=[a, b], value=opmap[op](a.get_value(), b.get_value()))


def generate_data(config: "NestedClacConfig") -> Tuple[List[Node], List[Node], List[Node]]:
    zeroth_l = [Node(op=None, children=None, value=value) for value in range(10 ** config.max_digits)]

    first_l: List[Node] = []
    # We want all the entries of level 1
    for op in config.operators:
        for a in zeroth_l:
            for b in zeroth_l:
                first_l.append(
                    Node(op=op, children=[a, b], value=opmap[op](a.get_value(), b.get_value()))
                )    

    # This is kind of dumb, we should likely just use a random subset of seconds_for_third_l
    all_second_l: List[Node] = list(generate(config, first_l, first_l + zeroth_l, "all", "ALL SECONDS"))
    seconds_for_third_l = all_second_l
    
    third_l = list(generate(config, seconds_for_third_l, seconds_for_third_l + first_l + zeroth_l, config.qty_third_layer, "THIRD"))
    
    second_l = list(all_second_l)
    random.shuffle(second_l)
    second_l = second_l[:config.qty_second_layer]
    random.shuffle(third_l)
    third_l = third_l[:config.qty_third_layer]

    print("Final lengths:")
    print(f"\tfirst_l: {len(first_l)}")
    print(f"\tsecond_l: {len(second_l)}")
    print(f"\tthird_l: {len(third_l)}")

    return first_l, second_l, third_l


@dataclasses.dataclass(frozen=True)
class PredLogger:
    __slots__ = ("root",)
    root: "Node"

    def log_doing(self, input_str: str, current_results: str):
        root_str = self.root.get_input_str()
        current_str = input_str
        highlighted = root_str.replace(current_str, f"[green bold]{current_str}[/green bold]")
        return f"{highlighted}"


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

if __name__ == '__main__':
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
        first=[x.to_json_dict() for x in tqdm(first)], 
        second=[x.to_json_dict() for x in tqdm(second)], 
        third=[x.to_json_dict() for x in tqdm(third)],
    )
    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / "dicts.pkl", "bw") as f:
        pickle.dump(dataset, f)
    print(f"Pickled dicts in {time.perf_counter() - start:0.2f} seconds")

    start = time.perf_counter()
    with open(SCRIPT_DIR / "data" / config.output_name, "w") as f:
        json.dump(dataset, f)
    print(f"Dumped json in {time.perf_counter() - start:0.2f} seconds")