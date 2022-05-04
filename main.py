print("Importing modules")
import copy
import logging
import math
from pathlib import Path
import pickle
import random
import time

from beartype import beartype
import fire
import numpy as np
import pytorch_lightning as pl
import rich
import torch
import transformers
from tqdm import tqdm
from typing import *
import ujson as json

import pretty_traceback
pretty_traceback.install()

import datagen
import our_metrics
import our_tokenizer
import our_data_collator
print("Done loading modules")

OUR_DEBUG = True
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

    def __init__(self, dataset: Dict[str, List[datagen.Node]], tokenizer, pred_fn):
        self.dataset = dataset["first"] + dataset["second"] + dataset["third"]
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        self.pred_fn = pred_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ = self.dataset[idx].get_input_str()
        pseudo_str, pseudo_without_head, _ = self.dataset[idx].get_pseudo(
            self.pred_fn, head_type="oracle"
        )

        return dict(
            input_ids=self.tokenizer(input_), 
            labels=self.tokenizer(pseudo_str),
            decoder_input_ids_for_gen=self.tokenizer(pseudo_without_head),
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
            curriculum_mode: str,
            curriculum_instance,
            max_generation_quantity_valid: int,
        ):
        super().__init__()
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
        self.max_generation_quantity_valid = max_generation_quantity_valid
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
        if "decoder_input_ids_for_gen" in kwargs:
            del kwargs["decoder_input_ids_for_gen"]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self.logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        # self.allen_ai_bart.training = False
        
        loss: torch.Tensor = self(**batch).loss

        def clean_fn(tokens):
            tokens_list = tokens.cpu().numpy().tolist()
            tokens_list = [x for x in tokens_list if x != -100]
            as_string = self.tokenizer.decode(tokens_list)

            return as_string.replace("<pad>", "[bright_cyan]<p>[/bright_cyan]")

        if batch_idx % 100 == 0:
            MODE = "per_unit"
            if MODE == "per_batch":
                assert False
                preds: torch.Tensor = self.model.generate(
                    input_ids=              batch["input_ids"], 
                    attention_mask=         batch["attention_mask"], 
                    decoder_input_ids=      batch["decoder_input_ids_for_gen"],
                    decoder_attention_mask= batch["decoder_input_ids_for_gen"] != self.tokenizer.pad_token_id,
                    **self.generation_kwargs, 
                )
                loop_package = zip(zip_dicts(batch), preds)
            elif MODE == "per_unit":
                loop_package = zip_dicts(batch)
            else:
                raise ValueError(f"Unknown mode {MODE}")

            freeform_options = [False]
            if "decoder_input_ids_for_gen" in batch:
                freeform_options.append(True)

            for is_freeform in freeform_options:
                freeform_maybe = "_freeform" if is_freeform else ""

                em = our_metrics.EM()

                for i, pack in enumerate(tqdm(loop_package)):
                    if i >= self.max_generation_quantity_valid:
                        break
                    
                    do_print = batch_idx == 0 and i < 5

                    if MODE == "per_batch":
                        sample, pred = pack

                    if MODE == "per_unit":
                        sample = pack
                        
                        generation_kwargs = dict(**self.generation_kwargs)
                        if "decoder_input_ids_for_gen" in pack:
                            filtered = [
                                x for x in sample["decoder_input_ids_for_gen"] 
                                if x != self.tokenizer.pad_token_id
                            ]
                            if is_freeform:
                                generation_kwargs["decoder_input_ids"] = (
                                    torch.stack(filtered).reshape(1, -1)
                                )

                        assert sample["input_ids"] is not None
                        assert sample["attention_mask"] is not None

                        pred = self.model.generate(
                            input_ids=sample["input_ids"].reshape(1, -1), 
                            attention_mask=sample["attention_mask"].reshape(1, -1), 
                            **self.generation_kwargs, 
                        )
                        
                        assert len(pred) == 1, pred.shape
                        assert len(pred.shape) == 2, pred.shape
                        pred = pred[0]

                    cleaned = our_metrics.OurMetric.prepare(
                        self.tokenizer, pred, sample["labels"], do_print=False
                    )

                    clean_pred  = cleaned["cleaned_preds"]
                    clean_label = cleaned["cleaned_labels"]
                    
                    if do_print:
                        rich.print(f"[blue]Inputs{freeform_maybe}:         \"{clean_fn(sample['input_ids'])}\"")
                        if "decoder_input_ids_for_gen" in sample:
                            rich.print(f"[blue]Decoder Inputs{freeform_maybe}: \"{clean_fn(sample['decoder_input_ids_for_gen'])}\"")
                        rich.print(f"[blue]Gen{freeform_maybe}:            \"{clean_fn(pred)}\"")
                        rich.print(f"[blue]Label{freeform_maybe}:          \"{clean_fn(sample['labels'])}")

                    em                .add(clean_pred, clean_label, do_print=do_print, descr=freeform_maybe)
                
                ###############################################################
                # Compute and print metrics
                ###############################################################
                em_acc_val =         em.compute()
                rich.print(f"[orange]GOT{freeform_maybe} {em.correct}/{em.total} = {em.correct / em.total:.2%}")
                self.log(f"EM{freeform_maybe}",             em_acc_val,         **self.logging_conf)
                self.log(f"eval_loss{freeform_maybe}",      loss,               **self.logging_conf)

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
            collate_fn=our_data_collator.DataCollatorWithDecoderInputIds(
                self.tokenizer, model=self.model, 
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

def main(
    run_name="oracle",
    dataset_type="oracle_basic_dataset", 
    dataset_path=None, 
    dicts_path=SCRIPT_DIR / "data" / "dicts.pkl",
):
    
    LEARNING_RATE = 0.001
    NUM_GPUS = 1
    
    EVAL_EVERY_N_EPOCHS = 2
    WANDB_ENTITY = "julesgm"
    WANDB_PROJECT = "self_learned_explanations"
    TRAIN_BATCH_SIZE = 1024
    MAX_GENERATION_QUANTITY_VALID = 128
    EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE
    GENERATION_MAX_LENGTH = 64
    PRECISION = 16
    GRADIENT_CLIP_VAL = 0.1

    dataset = load_dataset(dataset_path, dicts_path)
    train_ds = {k: v[:len(v) // 2] for k, v in dataset.items()}
    valid_ds = {k: v[len(v) // 2:] for k, v in dataset.items()}
    tokenizer = our_tokenizer.Tokenizer(max_length=1024, use_equal_symbol=True)
    train_torch_dataset = DATASET_TYPES[dataset_type](train_ds, tokenizer)
    valid_torch_dataset = DATASET_TYPES[dataset_type](valid_ds, tokenizer)

    ###############################################################
    # Should never change
    ###############################################################
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

    ###############################################################
    # Can change
    ###############################################################
    config.num_hidden_layers = 2 # Default is 6
    config.hidden_size = 128 # Default is 768
    config.encoder_attention_heads = 4 # Default is 16
    config.decoder_attention_heads = 4 # Default is 16
    config.encoder_ffn_dim = config.hidden_size * 4 # Default is 4096
    config.decoder_ffn_dim = config.hidden_size * 4

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
        max_generation_quantity_valid=MAX_GENERATION_QUANTITY_VALID,
    )

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(
            project=WANDB_PROJECT, 
            name=run_name, 
            entity=WANDB_ENTITY,
            log_model=False, 
            config=dict(
                **vars(config),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                num_gpus=NUM_GPUS,
                generation_max_length=GENERATION_MAX_LENGTH,
                approach_type=train_torch_dataset.__class__.__name__,
                precision=PRECISION,
                bart_config=vars(config)
            )
        ),
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision=PRECISION,
        max_epochs=1000, 
        gpus=NUM_GPUS,
        # val_check_interval=1000,
        check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
        limit_val_batches=300,
    )
    trainer.fit(pl_object)
    

if __name__ == "__main__":
    fire.Fire(main)