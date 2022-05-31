from dataclasses import dataclass
import math
import random
from typing import *
import time

import numpy as np
import rich
import torch

import datagen

import our_tokenizer

class MostBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False
    has_decoder_input_ids_for_gen = False

    def __init__(
        self, 
        dataset: Dict[str, List[datagen.Node]], 
        tokenizer: our_tokenizer.Tokenizer,
    ):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        input_ = self.dataset[idx].get_input_str()
        label = str(self.dataset[idx].get_value())
        return dict(
            input_ids=self.tokenizer(input_), 
            labels=self.tokenizer(label),
        )


class OracleBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False
    has_decoder_input_ids_for_gen = True

    def __init__(
        self, 
        dataset: Dict[str, List[datagen.Node]], 
        tokenizer: our_tokenizer.Tokenizer, 
    ):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoder_input = self.dataset[idx].get_input_str()
        label, decoder_input_for_gen = self.dataset[idx].get_oracle_str()

        return dict(
            input_ids=self.tokenizer(encoder_input), 
            labels=self.tokenizer(label),
            decoder_input_ids_for_gen=self.tokenizer(decoder_input_for_gen),
        )


class SelfLearnedBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False
    has_decoder_input_ids_for_gen = True
    
    def __init__(
        self, 
        dataset: Dict[str, List[datagen.Node]], 
        tokenizer: our_tokenizer.Tokenizer, 
    ):
        self._dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self._dataset)
        
        self._tokenizer = tokenizer
        self._conc_mode = None
        self._mask_intermediate_labels = None
    
    def has_len(self) -> bool:
        return hasattr(self._dataset, "__len__")

    def __len__(self) -> int:
        return len(self._dataset)
        
    def set_mask_intermediate_labels(
        self, 
        mask_intermediate_labels: bool,
    ):
        self._mask_intermediate_labels = mask_intermediate_labels

    def get(self, idx: int, pred_logger: datagen.PredLogger) -> Dict[str, torch.Tensor]:
        input_ = self._dataset[idx].get_input_str()
        if self._conc_mode == "top_sort":
            self._dataset[idx].reset_pseudo_values()
            return self._dataset[idx]
        elif self._conc_mode == "yield":
            output = yield from self._dataset[idx].get_pseudo(
                head_type="oracle", 
                conc_mode=self._conc_mode, 
                logging_info=pred_logger,
                tokenizer=self._tokenizer,
            )
        else:
            raise NotImplementedError(f"Conc mode {self._conc_mode} unknown")

        pseudo_str, pseudo_without_head, masked_intermediate_solutions = output
        masked_intermediate_solutions = masked_intermediate_solutions + [self._tokenizer.eos_token_id]
        assert isinstance(pseudo_str, str), type(pseudo_str)
        assert isinstance(pseudo_without_head, str), type(pseudo_without_head)

        tokenized_pseudo_str = self._tokenizer(pseudo_str)
        ouptut = dict(
            input_ids=self._tokenizer(input_), 
            labels=(
                masked_intermediate_solutions 
                if self._mask_intermediate_labels 
                else tokenized_pseudo_str
            ),
            decoder_input_ids=tokenized_pseudo_str,
            decoder_input_ids_for_gen=self._tokenizer(pseudo_without_head),
        )
        return ouptut

    def set_conc_mode(self, conc_mode: str) -> None:
        self._conc_mode = conc_mode
        assert conc_mode in ["top_sort", "yield"]

    def set_send_pull(self, send_pull):
        self._pred_functor.set_send_pull(send_pull)


class CurriculumSelfLearned(SelfLearnedBasicDataset):
    def __init__(self, 
        dataset: Dict[str, List[datagen.Node]], 
        tokenizer: our_tokenizer.Tokenizer, 
    ):
        super().__init__(dataset, tokenizer)
        self._split_datasets = [dataset["first"], dataset["second"], dataset["third"]]
        self._dataset = None

    def mix(self, mix: Dict[int, float]) -> None:
        mixed = []
        for split_idx, ratio in mix.items():
            split_dataset = self._split_datasets[split_idx]
            random.shuffle(split_dataset)
            mixed += split_dataset[:int(len(split_dataset) * ratio)]
        self._dataset = mixed
        random.shuffle(self._dataset)

        
DATASET_TYPES = dict(
    most_basic_dataset=MostBasicDataset,
    oracle_basic_dataset=OracleBasicDataset,
    self_learned_basic_dataset=CurriculumSelfLearned,
)