#!/usr/bin/env python
# coding: utf-8

print("loading modules")
import dataclasses
import math
from pathlib import Path


import json
import numpy as np
import random
import rich
import tqdm
from typing import *
print("Done loading modules")


SCRIPT_DIR = Path(__file__).absolute().parent

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

    def to_json_dict(self):
        return {
            "op": self.get_op(),
            "children": [child.to_json_dict() for child in self.get_children()] if self.get_children() else None,
            "value": self.get_value(),
            "input_str": self.get_input_str(),
            "oracle_str": self.get_oracle_str(),
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
        # Multiple calls should always return the same thing
        if self._oracle_str is None:
            if self.get_children():
                assert len(self.get_children()) == 2, len(self.get_children())
                a = self.get_children()[0].get_oracle_str()
                b = self.get_children()[1].get_oracle_str()
                self._oracle_str = f"({a} {self._op} {b} = {self.get_value()})"
            else:
                self._oracle_str = f"{self.get_value()}"
            
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

def generate(config, previous, all_previous, qty_each_op, name=""):
    for op in config.operators:
        if qty_each_op == "all" or qty_each_op == len(previous) * len(all_previous):
            rich.print(f"[red]({name}): Doing all {len(previous) * len(all_previous)}")
            uniques = set()
            for a in previous:
                for b in all_previous:
                    uniques.add((a, b))

        else:
            upper_bound = len(previous) * len(all_previous)
            assert qty_each_op <= upper_bound, f"{qty_each_op = } > {upper_bound = }"
            a_s = np.random.choice(previous, qty_each_op, replace=True)
            b_s = np.random.choice(all_previous, qty_each_op, replace=True)
            
            pre_unique = qty_each_op
            uniques = set(zip(a_s, b_s))
            rich.print(f"[red]({name}): {pre_unique = }, {len(uniques) = }, {len(uniques) / pre_unique = :0.1%}")

            while len(uniques) != qty_each_op:
                a_s = np.random.choice(previous, qty_each_op, replace=True)
                b_s = np.random.choice(all_previous, qty_each_op, replace=True)
                good_ones = [x for x in zip(a_s, b_s) if x not in uniques]
                uniques.update(good_ones[:qty_each_op - len(uniques)])
                print("[attempt]")

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
    
    qty_each_op = math.ceil(config.qty_third_layer / len(config.operators))
    third_l = list(generate(config, seconds_for_third_l, seconds_for_third_l + first_l + zeroth_l, qty_each_op, "THIRD"))
    
    second_l = list(all_second_l)
    random.shuffle(second_l)
    second_l = second_l[:config.qty_second_layer]
    random.shuffle(third_l)
    third_l = third_l[:config.qty_third_layer]

    return first_l, second_l, third_l
    

@dataclasses.dataclass
class NestedClacConfig:
    # misc
    seed: int = 1337
    output_name: str = "dataset.json"

    # dataset
    operators: Set[str] = dataclasses.field(default_factory=lambda: {"+", "*", "-"})
    max_depth: int = 3
    max_digits: int = 1
    qty_third_layer = 100000
    qty_second_layer = 100000

if __name__ == '__main__':
    config = NestedClacConfig()
    first, second, third = generate_data(config)
    dataset = dict(
        first=[x.to_json_dict() for x in tqdm.tqdm(first)], 
        second=[x.to_json_dict() for x in tqdm.tqdm(second)], 
        third=[x.to_json_dict() for x in tqdm.tqdm(third)],
    )
    with open(SCRIPT_DIR / "data" / config.output_name, "w") as f:
        json.dump(dataset, f)
