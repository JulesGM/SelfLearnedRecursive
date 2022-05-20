from dataclasses import dataclass
import random
from typing import *
import time

import rich
import torch

import datagen



class MostBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False

    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ = self.dataset[idx].get_input_str()
        label = str(self.dataset[idx].get_value())
        return dict(
            input_ids=self.tokenizer(input_), 
            labels=self.tokenizer(label),
        )


class OracleBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False

    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        encoder_input = self.dataset[idx].get_input_str()
        label, decoder_input_for_gen = self.dataset[idx].get_oracle_str()

        return dict(
            input_ids=self.tokenizer(encoder_input), 
            labels=self.tokenizer(label),
            decoder_input_ids_for_gen=self.tokenizer(decoder_input_for_gen),
        )


class SelfLearnedBasicDataset(torch.utils.data.Dataset):
    has_curriculum = False
    
    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer):
        self._dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self._dataset)
        self.tokenizer = tokenizer
        self._pred_functor = None
        self._conc_mode = None
        
    def __len__(self):
        return len(self._dataset)

    def get(self, idx):
        input_ = self._dataset[idx].get_input_str()
        if self._conc_mode == "yield":
            assert self._pred_functor is None, "Pred functor not used in yield mode"
            output = yield from self._dataset[idx].get_pseudo(
                None, 
                head_type="oracle", 
                conc_mode=self._conc_mode, 
                logging_info=datagen.PredLogger(self._dataset[idx])
            )
        else:
            assert self._pred_functor
            output = self._dataset[idx].get_pseudo(
                self._pred_functor, 
                head_type="oracle", 
                conc_mode=self._conc_mode, 
                logging_info=datagen.PredLogger(self._dataset[idx])
            )

        pseudo_str, pseudo_without_head = output

        assert isinstance(pseudo_str, str), type(pseudo_str)
        assert isinstance(pseudo_without_head, str), type(pseudo_without_head)

        ouptut = dict(
            input_ids=self.tokenizer(input_), 
            labels=self.tokenizer(pseudo_str),
            decoder_input_ids_for_gen=self.tokenizer(pseudo_without_head),
        )
        end = time.perf_counter()
        # rich.print(f"[green]Took {end - start} seconds to get item {idx}")
        return ouptut

    def set_pred_functor(self, pred_functor):
        self._pred_functor = pred_functor
        if self._conc_mode == "yield":
            assert self._pred_functor is None, (
                "Pred functor not used in yield mode"
            )

    def set_conc_mode(self, conc_mode):
        self._conc_mode = conc_mode
        if self._conc_mode == "yield":
            assert self._pred_functor is None, (
                "Pred functor not used in yield mode"
            )

    def set_send_pull(self, send_pull):
        self._pred_functor.set_send_pull(send_pull)


        
DATASET_TYPES = dict(
    most_basic_dataset=MostBasicDataset,
    oracle_basic_dataset=OracleBasicDataset,
    self_learned_basic_dataset=SelfLearnedBasicDataset,
)