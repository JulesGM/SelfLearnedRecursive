print("Importing modules")
import abc
import logging
import itertools
import math
from pathlib import Path
import pickle
import random
import time
from unittest.util import _MAX_LENGTH

from beartype import beartype
import datasets
import fire
import numpy as np
import pytorch_lightning as pl
import re
from regex import P
import rich
import torch
from torchmetrics import F1
import transformers
from tqdm import tqdm
from typing import *
import ujson as json

import pretty_traceback
pretty_traceback.install()

import datagen
print("Done loading modules")

SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)

class ANSWER_MODES:
    WHOLE_OUTPUT = "whole_output"
    FINAL_TERM_ONLY = "final_term_only"
    _VALUES = set([WHOLE_OUTPUT, FINAL_TERM_ONLY])

    def check(cls, mode):
        assert mode in cls._VALUES, f"Unknown mode '{mode}', must be one of {cls._VALUES}"


def zip_dicts(*dicts):
    d = {}
    for d_ in dicts:
        for k in d_.keys():
            assert k not in d, f"Duplicate key {k} in dicts. {d.keys()}"
        d.update(d_)

    keys = d.keys()
    length = None
    for k, v in d.items():
        if length is None:
            length = len(v)
        assert len(v) == length, f"{k} has length {len(v)} != {length}"

    iter_d = {k: iter(v) for k, v in d.items()}
    while True:
        try:
            yield {k: next(iter_d[k]) for k in keys}
        except StopIteration:
            break

class OurMetric(abc.ABC):
    @classmethod
    def prepare(cls, tokenizer, pred, label, do_print):
        things_to_ignore = {
            -100, 
            tokenizer.pad_token_id, 
            tokenizer.bos_token_id, 
            tokenizer.eos_token_id
        }
        
        assert 0 in things_to_ignore, things_to_ignore
        assert 1 in things_to_ignore, things_to_ignore
        assert 2 in things_to_ignore, things_to_ignore

        cleaned_preds = [x for x in  pred.cpu().numpy().tolist() if x not in things_to_ignore]
        cleaned_labels = [x for x in label.cpu().numpy().tolist() if x not in things_to_ignore]

        return dict(
            cleaned_preds=cleaned_preds, 
            cleaned_labels=cleaned_labels
        )

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")


class EM(OurMetric):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, pred: list, label: list, do_print=False, descr=""):
        prepped_decoded = list(pred)
        prepped_label =   list(label)

        is_match = prepped_decoded == prepped_label
        is_match_np = np.all(np.array(prepped_decoded) == np.array(prepped_label)) 
        assert is_match == is_match_np, (is_match, is_match_np)

        if prepped_decoded == prepped_label:
            self.correct += 1
        else:
            pass
        
        if do_print:            
            rich.print(f"(EM) Answer - {descr}: " + ", ".join(
                [f"[green]{a}" if a == b else f"[red]{a}" for a, b 
                in itertools.zip_longest(prepped_decoded, prepped_label, fillvalue="")]
            ))
            rich.print(f"(EM) Label - {descr}:   " + ", ".join(
                [f"[green]{b}" if a == b else f"[red]{b}" for a, b 
                in itertools.zip_longest(prepped_decoded, prepped_label, fillvalue="")]
            ))

        self.total += 1 
    
    def compute(self):
        return self.correct / self.total


class RecallAcc:
    def __init__(self):
        self.recall_accuracies = []

    @beartype
    def add(self, pred: list, label: list, do_print: bool = False, descr=""):
        recall_acc_decoded = list(pred)
        recall_acc_label = list(label)

        if len(recall_acc_decoded) < len(recall_acc_label):
            recall_acc_decoded += [0] * (len(recall_acc_label) - len(recall_acc_decoded))
        elif len(recall_acc_decoded) > len(recall_acc_label):
            recall_acc_decoded = recall_acc_decoded[:len(recall_acc_label)]

        recall_acc_label_np =   np.array(recall_acc_label,   dtype=np.int64)
        recall_acc_decoded_np = np.array(recall_acc_decoded, dtype=np.int64)
        recall_acc =            np.mean(recall_acc_decoded_np == recall_acc_label_np)

        self.recall_accuracies.append(recall_acc)

    def compute(self):
        return np.mean(self.recall_accuracies)


class PrecisionAcc:
    def __init__(self): 
        self.precision_accuracies = []

    @beartype
    def add(self, pred: list, label: list, do_print: bool = False, descr=""):
        precision_acc_decoded = list(pred)
        precision_acc_label = list(label)

        if len(precision_acc_decoded) > len(precision_acc_label):
            precision_acc_label += [0] * (len(precision_acc_decoded) - len(precision_acc_label))
        elif len(precision_acc_decoded) < len(precision_acc_label):
            precision_acc_label = precision_acc_label[:len(precision_acc_decoded)]

        precision_acc_label_np =   np.array(precision_acc_label,   dtype=np.int64)
        precision_acc_decoded_np = np.array(precision_acc_decoded, dtype=np.int64)
        precision_acc =            np.mean(precision_acc_decoded_np == precision_acc_label_np) 

        self.precision_accuracies.append(precision_acc) 

    def compute(self): 
        return np.mean(self.precision_accuracies)


class MostBasicDataset(torch.utils.data.Dataset):
    answer_mode = ANSWER_MODES.WHOLE_OUTPUT
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
        return dict(input_ids=self.tokenizer(input_), labels=self.tokenizer(label))


class OracleBasicDataset(torch.utils.data.Dataset):
    answer_mode = ANSWER_MODES.WHOLE_OUTPUT
    has_curriculum = False

    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ = self.dataset[idx].get_input_str()
        label = self.dataset[idx].get_oracle_str()
        return dict(input_ids=self.tokenizer(input_), labels=self.tokenizer(label))


class SelfLearnedBasicDataset(torch.utils.data.Dataset):
    answer_mode = ANSWER_MODES.WHOLE_OUTPUT
    has_curriculum = False

    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer, pred_fn):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        self.pred_fn = pred_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ = self.dataset[idx].get_input_str()
        str_with_predictions, _ = self.dataset[idx].get_pseudo(self.pred_fn, True)
        return dict(
            input_ids=self.tokenizer(input_), 
            labels=self.tokenizer(str_with_predictions)
        )


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
            dataset[level_name].append(datagen.Node.from_json_dict(node_dict))
    
    LOGGER.debug(f"Done loading dataset.")
    return dataset


class Tokenizer:
    def __init__(self, max_length, use_equal_symbol):
        self.vocab = [
            "<pad>", 
            "<bos>", 
            "<eos>", 
            "0", 
            "1", 
            "2", 
            "3", 
            "4", 
            "5", 
            "6", 
            "7", 
            "8", 
            "9", 
            "+", 
            "-", 
            "*", 
            "(", 
            ")",
        ]

        if use_equal_symbol:
            self.vocab.append("=")

        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = self.vocab
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = self.token_to_idx["<bos>"]
        self.eos_token_id = self.token_to_idx["<eos>"]
        self.pad_token_id = self.token_to_idx["<pad>"]
        self.padding_side = "left"
        self.max_length = max_length

    def tokenize(self, input_str):
        assert type(input_str) == str, type(input_str)
        output = []
        for char in input_str:
            if char in self.token_to_idx:
                output.append(self.token_to_idx[char])
            else:
                if char == " ":
                    continue
                else:
                    raise ValueError(f"Unknown token '{char}'")
        
        output = np.array(output + [self.token_to_idx["<eos>"]], dtype=np.int64)
        return output
    
    def decode(self, input_tokens):
        assert type(input_tokens) == list, type(input_tokens)
        output = []
        for token_index in input_tokens:
            if isinstance(token_index, int) and token_index >= 0 and token_index < len(self.idx_to_token):
                output.append(self.idx_to_token[token_index])
            else:
                raise ValueError(f"Unknown token index '{token_index}'")
        
        return " ".join(output)

    def pad(self, features, padding, max_length, pad_to_multiple_of, return_tensors):
        assert not pad_to_multiple_of, "Not implemented"
        assert padding is True or padding == "longuest", "Other values are not implemented"
        

        max_read = max(len(x["input_ids"]) for x in features)

        if max_length is not None:
            max_read = min(max_read, max_length)
        
        padded_sequences = []
        for seq in features:
            if len(seq["input_ids"]) < max_read:
                seq["input_ids"] = seq["input_ids"].tolist() + [self.token_to_idx["<pad>"]] * (max_read - len(seq["input_ids"]))
                seq["input_ids"] = seq["input_ids"][:max_read]
            seq["attention_mask"] = [int(x != self.token_to_idx["<pad>"]) for x in seq["input_ids"]]
            padded_sequences.append(seq)
        
        
        if return_tensors == "np":
            return {
                "input_ids": np.array([x["input_ids"] for x in padded_sequences], dtype=np.int64),
                "attention_mask": np.array([x["attention_mask"] for x in padded_sequences], dtype=np.int64),
                "labels": np.array([x["labels"] for x in padded_sequences], dtype=np.int64)
            }
        elif return_tensors == "pt":
            return {
                "input_ids": torch.tensor([x["input_ids"] for x in padded_sequences], dtype=torch.int64),
                "attention_mask": torch.tensor([x["attention_mask"] for x in padded_sequences], dtype=torch.int64),
                "labels": torch.tensor([x["labels"] for x in padded_sequences], dtype=torch.int64)
            }
        else:
            raise ValueError(f"Unknown return_tensors value '{return_tensors}'")

    def __call__(self, input_str):
        return self.tokenize(input_str)


class PLBart(pl.LightningModule):
    def __init__(self, *, 
            model, 
            tokenizer, 
            train_ds, 
            eval_ds, 
            train_batch_size, 
            eval_batch_size, 
            num_workers_dl, 
            generation_kwargs,
            learning_rate,
            is_adamw,
            weight_decay,
            scheduler_type,
            scheduler_kwargs,
            do_allen_nlp_predictions: bool,
            answer_mode: str,
            curriculum_mode: str,
            curriculum_instance,
        ):
        super().__init__()
        self.answer_mode =              answer_mode
        self.tokenizer =                tokenizer
        self.model =                    model 
        self.batch_size =               train_batch_size
        self.eval_batch_size =          eval_batch_size
        self.generation_kwargs =        generation_kwargs
        self.logging_conf =             dict(
                                            prog_bar= True, 
                                            on_step=  True, 
                                            on_epoch= True, 
                                            logger=   True
                                        )
        # self.do_allen_nlp_predictions = do_allen_nlp_predictions

        # Related to datasets
        self.train_ds =                 train_ds
        self.eval_ds =                  eval_ds
        self.num_workers_dl =           num_workers_dl

        # Specific to the optimizer:
        self.learning_rate =            learning_rate
        self.is_adamw =                 is_adamw
        self.weight_decay =             weight_decay

        # # Related to the scheduler:
        # self.scheduler_type =         scheduler_type
        # self.scheduler_kwargs =       scheduler_kwargs

        # # Experimental
        # self.allen_ai_bart =          allen_ai_bart.BartReuser(
        #                                   model=self.model, 
        #                                   model_name=self.tokenizer.name_or_path, 
        #                                   vocab=allennlp.data.Vocabulary.from_pretrained_transformer(
        #                                       self.tokenizer.name_or_path), 
        #                                   max_decoding_steps=generation_kwargs["max_length"],
        #                                   beam_size=generation_kwargs["num_beams"],
        #                               )

        # Curriculum
        self.curriculum_mode =          curriculum_mode
        self.cluster_picker =           curriculum_instance
        self.already_started =          False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):        
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self.logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        # self.allen_ai_bart.training = False

        loss: torch.Tensor = self(**batch).loss
        self.log(f"loss",           loss,               **self.logging_conf)

        if batch_idx % 100 == 0:        

            preds: torch.Tensor = self.model.generate(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                **self.generation_kwargs, 
            )

            # if self.do_allen_nlp_predictions:
            #     allen_ai_inputs = dict(
            #         tokens=dict(
            #             token_ids=batch["input_ids"], 
            #             mask=batch["attention_mask"],
            #         )
            #     )

            #     allen_ai_preds = self.allen_ai_bart.forward(
            #         source_tokens=allen_ai_inputs,
            #     )["predictions"]
            #     types_of_preds.append(("allen_nlp", allen_ai_preds))

            em = EM()
            recall_accuracy = RecallAcc()
            precision_accuracy = PrecisionAcc()


            for i, (sample, pred) in enumerate(zip(zip_dicts(batch), preds)):
                do_print = batch_idx == 0 and i < 5
                cleaned = OurMetric.prepare(
                    self.tokenizer, pred, sample["labels"], do_print=False
                )

                clean_pred  = cleaned["cleaned_preds"]
                clean_label = cleaned["cleaned_labels"]

                assert isinstance(clean_pred,  list), type(clean_pred ).mro()
                assert isinstance(clean_label, list), type(clean_label).mro()

                if self.answer_mode == ANSWER_MODES.FINAL_TERM_ONLY:
                    assert "=" in ref
                    if "=" not in answer:
                        answer = ""
                    answer = clean_pred.strip().rsplit("=", 1).strip()
                    ref = clean_label.strip().rsplit("=", 1).strip()
                elif self.answer_mode == ANSWER_MODES.WHOLE_OUTPUT:
                    answer = clean_pred
                    ref = clean_label
                else:
                    raise ValueError(f"Unknown answer_mode '{self.answer_mode}'")

                header = "{whole_output}" if self.answer_mode == ANSWER_MODES.WHOLE_OUTPUT else "{{final_term}}"
                
                if do_print:
                    rich.print(f"[red]{header} - Answer: {answer}")
                    rich.print(f"[red]{header} - Ref:    {ref}")
                    rich.print(f"[orange]{header} - Inputs: \"{self.tokenizer.decode(sample['input_ids'].detach().cpu().numpy().tolist())}\"")
                    rich.print(f"[orange]{header} - Answer: \"{self.tokenizer.decode(answer)}\"")
                    rich.print(f"[orange]{header} - Ref:    \"{self.tokenizer.decode(ref)}\"")


                em                .add(answer, ref, do_print=do_print)
                recall_accuracy   .add(answer, ref, do_print=do_print)
                precision_accuracy.add(answer, ref, do_print=do_print)

                em_acc_val =         em.compute()
                recall_acc_val =     recall_accuracy.compute()
                precision_acc_val =  precision_accuracy.compute()
                f1_ACC =             2 * precision_acc_val * recall_acc_val / (precision_acc_val + recall_acc_val)
                
            rich.print(f"[orange]GOT {em.correct}/{em.total} = {em.correct / em.total:.2%}")
            self.log(f"EM",             em_acc_val,         **self.logging_conf)
            self.log(f"recall_ACC",     recall_acc_val,     **self.logging_conf)
            self.log(f"precision_ACC",  precision_acc_val,  **self.logging_conf)
            self.log(f"f1_ACC",         f1_ACC,             **self.logging_conf)

            # self.allen_ai_bart.training = True


    def configure_optimizers(self):
        """
        See ref 
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        if self.is_adamw: 
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adam

        optimizer = optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )

        output = dict(optimizer=optimizer)

        # if SCHEDULER_TYPES[self.scheduler_type]:
        #     output["lr_scheduler"] = {}
        #     output["lr_scheduler"]["scheduler"] = SCHEDULER_TYPES[self.scheduler_type](
        #         optimizer=optimizer, **self.scheduler_kwargs
        #     )
        #     output["lr_scheduler"]["interval"] = "epoch"
        #     output["frequency"] = 1

        return output


    def _make_dataloader(self, ds, batch_size, shuffle):
        return torch.utils.data.DataLoader(
            ds, 
            collate_fn=transformers.data.data_collator.DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True
            ), 
            batch_size=batch_size, 
            num_workers=self.num_workers_dl,
            shuffle=shuffle,
        ) 

    def train_dataloader(self):
        return self._make_dataloader(
            self.train_ds, 
            self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        return self._make_dataloader(
            self.eval_ds,
            self.eval_batch_size,
            shuffle=False
        )
        
DATASET_TYPES = dict(
    most_basic_dataset=MostBasicDataset,
    oracle_basic_dataset=OracleBasicDataset,
    self_learned_basic_dataset=SelfLearnedBasicDataset,
)

def main(dataset_path = None, dicts_path = SCRIPT_DIR / "data" / "dicts.pkl"):
    
    LEARNING_RATE = 0.0001
    NUM_GPUS = 1
    DATASET_TYPE = "oracle_basic_dataset"
    WANDB_RUN_NAME = "early runs"
    
    WANDB_ENTITY = "julesgm"
    WANDB_PROJECT = "self_learned_explanations"
    TRAIN_BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 512
    GENERATION_MAX_LENGTH = 64
    PRECISION = 16
    TRAIN_VALID_SPLIT = .5

    dataset = load_dataset(dataset_path, dicts_path)
    train_ds = {k: v[:len(v)//2] for k, v in dataset.items()}
    valid_ds = {k: v[len(v)//2:] for k, v in dataset.items()}
    tokenizer = Tokenizer(max_length=1024, use_equal_symbol=True)
    train_torch_dataset = DATASET_TYPES[DATASET_TYPE](train_ds, tokenizer)
    valid_torch_dataset = DATASET_TYPES[DATASET_TYPE](valid_ds, tokenizer)

    config = transformers.BartConfig.from_pretrained("facebook/bart-base")
    config.no_repeat_ngram_size = 0
    config.pad_token_id = tokenizer.pad_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.forced_bos_token_id = tokenizer.bos_token_id
    config.forced_eos_token_id = tokenizer.eos_token_id
    config.vocab_size = len(tokenizer.vocab)
    config.task_specific_params = {}

    assert tokenizer.max_length == config.max_position_embeddings, (
        f"max_length={tokenizer.max_length} != "
        f"config.max_position_embeddings={config.max_position_embeddings}"
    )

    model = transformers.BartForConditionalGeneration(
        config
    )

    pl_object = PLBart(
        model=model, 
        tokenizer=tokenizer, 
        train_ds=train_torch_dataset, 
        eval_ds=valid_torch_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers_dl=0,
        generation_kwargs=dict(
            max_length=GENERATION_MAX_LENGTH,
            min_length=0,
        ),
        learning_rate=LEARNING_RATE,
        is_adamw=True,
        weight_decay=0,
        scheduler_type="WarmupLinear",
        scheduler_kwargs=dict(),
        do_allen_nlp_predictions=False,
        curriculum_mode=None,
        curriculum_instance=None,
        answer_mode=train_torch_dataset.answer_mode,
    )

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(
            project=WANDB_PROJECT, 
            name=WANDB_RUN_NAME, 
            entity=WANDB_ENTITY,
            log_model=False, 
            config=dict(
                **vars(config),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                num_gpus=NUM_GPUS,
                generation_max_length=GENERATION_MAX_LENGTH,
                approach_type=train_torch_dataset.__class__.__name__,
                answer_mode=train_torch_dataset.answer_mode,
                precision=PRECISION
            )
        ),
        precision=PRECISION,
        max_epochs=1000, 
        gpus=NUM_GPUS,
        # val_check_interval=1000,
        check_val_every_n_epoch=1,
        limit_val_batches=300,
    )
    trainer.fit(pl_object)
    

if __name__ == "__main__":
    fire.Fire(main)