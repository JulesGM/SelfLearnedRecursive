#!/usr/bin/env python
# coding: utf-8

# In[3]:


from dataclasses import dataclass, field
import itertools
from pathlib import Path
import random

import datasets as ds
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *


EXP_COL = "expression"
VAL_COL = "value"

@dataclass
class NestedClacConfig:
    # misc
    seed: int = 1337
    num_proc: int = 8

    # dataset
    operators: Set[str] = field(default_factory=lambda: {"+", "*", "-"})
    max_depth: int = 2
    max_digits: int = 1

opmap = {
    "+": lambda x, y: x + y,
    "*": lambda x, y: x * y,
    "-": lambda x, y: x - y,
    "/": lambda x, y: x // y,
}

class MemNestedClacDataModule(pl.LightningDataModule):
    def __init__(self, config: NestedClacConfig):
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

    def setup(self, stage: Optional[str] = None) -> None:
        print("MemNestedClacDataModule::setup")
        c = self.config
        depth_datasets = {}

        def create_dataset(depth: int):
            print("MemNestedClacDataModule::setup::create_dataset")

            for i in range(depth):
                assert i in depth_datasets

            out = {EXP_COL: [], VAL_COL: []}
            previous_depth_ds = depth_datasets[depth - 1]

            for op in c.operators:
                # For each operator, for each entry of the dataset of the previous depth,
                # combine each value of the previous depth with all of the previous values
                for value, expression in zip(
                    tqdm(previous_depth_ds[VAL_COL]), previous_depth_ds[EXP_COL]
                ):

                    # For each of the previous datasets
                    for a_previous_depth_ds in depth_datasets.values():
                        new_values = (
                            [opmap[op](y, value) for y in a_previous_depth_ds[VAL_COL]] + 
                            [opmap[op](value, y) for y in a_previous_depth_ds[VAL_COL]]
                        )

                        new_expressions = (
                            [f"({entry}{op}{expression})" for entry in a_previous_depth_ds[EXP_COL]] +
                            [f"({expression}{op}{entry})" for entry in a_previous_depth_ds[EXP_COL]]
                        )

                        out[VAL_COL].extend(new_values)
                        out[EXP_COL].extend(new_expressions)

            return out

        for depth in range(c.max_depth + 1):
            if depth == 0:
                values = list(range(10 ** c.max_digits))
                expressions = [f"{x}" for x in values]
                depth_datasets[depth] = {EXP_COL: expressions, VAL_COL: values}
                continue
            depth_datasets[depth] = create_dataset(depth)
        self.depth_datasets = depth_datasets

        print("MemNestedClacDataModule::setup::done")


if __name__ == '__main__':
    config = NestedClacConfig()
    dm = MemNestedClacDataModule(config)
    dm.prepare_data()
    dm.setup()
    print(dm)


# In[ ]:




