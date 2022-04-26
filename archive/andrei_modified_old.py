#!/usr/bin/env python
# coding: utf-8
# cython: linetrace=True
# cython: profile=True

print("loading modules")
from dataclasses import dataclass, field
import time

import math
import numpy as np
import pytorch_lightning as pl
import random
import rich
from tqdm import tqdm
from typing import *
print("Done loading modules")

@dataclass
class NestedClacConfig:
    # misc
    seed: int = 1337

    # dataset
    operators: Set[str] = field(default_factory=lambda: {"+", "*", "-"})
    max_depth: int = 3
    max_digits: int = 1

opmap = {
    "+": lambda x, y: x + y,
    "*": lambda x, y: x * y,
    "-": lambda x, y: x - y,
    "/": lambda x, y: x // y,
}


class Node:
    __slots__ = (
        "_op", 
        "_children", 
        "_value",
        "_input_str", 
        "_oracle_str", 
        "_pseudo_str",
    )

    def __init__(self, op: Optional[str], children: Optional[List["Node"]], value: int):
        self._op = op
        self._children = children
        self._value = value
        self._input_str:  Optional[str] = None
        self._oracle_str: Optional[str] = None
        self._pseudo_str: Optional[str] = None

    def get_op(self):
        return self._op

    def get_children(self):
        return self._children

    def get_value(self):
        return self._value

    def get_input_str(self):
        # Multiple calls should always return the same thing
        if self._input_str is None:
            if self.children:
                a = self.children[0].get_input_str()
                b = self.children[1].get_input_str()
                assert len(self.children) == 2, len(self.children)
                self._input_str = f"({a} {self.get_op()} {b})"
            else:
                self._input_str = f"{self.get_value()}"
        return self._input_str

    def get_oracle_str(self):
        # Multiple calls should always return the same thing
        if self._oracle_str is None:
            if self.get_children():
                a = self.get_children()[0].get_oracle_str()
                b = self.get_children()[1].get_oracle_str()
                self._oracle_str = f"({a} {self._op} {b} = {self.get_value()})"
            else:
                self._oracle_str = f"{self.get_value()}"
            assert len(self.get_children()) == 2, len(self.get_children())
        return self._oracle_str

    def get_pseudo(self, prediction_function):
        # Multiple calls will have DIFFERENT RESULTS

        if self.children is not None:
            a_str, a_pred = self.children[0].get_pseudo(prediction_function)
            b_str, b_pred = self.children[1].get_pseudo(prediction_function)
            prediction = prediction_function(a_pred, b_pred)
            self._pseudo_str = f"({a_str} {self.get_op()} {b_str} = {prediction})"
        else:
            self._pseudo_str = f"{self.get_value()}"
            prediction = self.get_value()
        return self._pseudo_str, prediction

    def __repr__(self):
        return f"Node({self.get_oracle_str()})"


def inner(out, op, a_tree: Node, b_tree: Node):
    new_val = opmap[op](a_tree.get_value(), b_tree.get_value())
    new_node = Node(op=op, children=[a_tree, b_tree], value=new_val)
    out.append(new_node)
    

def create_dataset(depth, depth_datasets, operators, do_drop, keep_prob):
    rich.print(f"[red]depth = {depth}, do_drop = {do_drop}, keep_prob = {keep_prob}")

    for i in range(depth):
        assert i in depth_datasets

    out = []
    previous_depth_ds = depth_datasets[depth - 1]
    counter_qty = len(operators) * len(previous_depth_ds) * 2 * sum(len(x) for x in depth_datasets.values())
    expected_qty = keep_prob * counter_qty

    print("generating kept numbers")
    kept_numbers = [i for i in tqdm(range(counter_qty)) if random.random() > keep_prob]
    print("transforming to set")
    keep_set = set(kept_numbers)
    print(f"starting the work len(keep_set) = {len(keep_set)}")
    
    rich.print(f"[red]{counter_qty = }, {do_drop = }, {expected_qty = }")
    counter = tqdm(total = counter_qty)
    index = 0

    for op in operators:
        # For each operator, for each entry of the dataset of the previous depth,
        # combine each value of the previous depth with all of the previous values
        for x_tree in previous_depth_ds:

            # For each of the previous datasets
            for a_previous_depth_ds in depth_datasets.values():
                # if do_drop and random.random() > keep_prob:
                #     counter.update(2 * len(a_previous_depth_ds))
                #     continue

                for y_tree in a_previous_depth_ds:
                    
                    if not do_drop or index in keep_set:
                        inner(out, op, x_tree, y_tree)
                    index += 1

                counter.update(2 * len(a_previous_depth_ds))

    return out


class MemNestedClacDataModule(pl.LightningDataModule):
    def __init__(self, config: NestedClacConfig):
        print("MemNestedClacDataModule::__init__")
        super().__init__()
        self.config = config

    def estimate_num_expressions(self) -> int:
        print("MemNestedClacDataModule::estimate_num_expressions")
        c = self.config
        num_terms = 10 ** c.max_digits
        total = num_terms  # total number of expressions
        new = num_terms  # number of new expressions added in recursion step
        for _ in range(c.max_depth):
            # consider all pairwise combinations between (1) new expressions
            # from previous recursion step and (2) all expressions so far.
            new = total * new * 2 * len(c.operators)
            total = total + new
        return total

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        print("MemNestedClacDataModule::setup")

        c = self.config
        depth_datasets = {}        
        do_drop = set([3])
        keep_rate = {
            3: 0.000001
        }

        for depth in range(c.max_depth + 1):
            if depth == 0:
                depth_datasets[depth] = [
                    Node(op=None, children=None, value=number) 
                    for number in range(10 ** c.max_digits)
                ]
                continue

            start = time.perf_counter()
            depth_datasets[depth] = create_dataset(
                depth, 
                depth_datasets, 
                c.operators, 
                depth in do_drop, 
                keep_rate.get(depth, 1),
            )
            rich.print(f"[red]Time: {time.perf_counter() - start}")

        self.depth_datasets = depth_datasets
        print("MemNestedClacDataModule::setup::done")
