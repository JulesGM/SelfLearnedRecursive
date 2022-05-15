print("Importing modules")
from asyncio.subprocess import PIPE
import concurrent.futures as futures
import copy
import dataclasses
import logging
import math
import multiprocessing.dummy as dummy
import multiprocessing

import wandb
from pathlib import Path
import pickle
import queue
import random
import threading
import time

from beartype import beartype
import fire
import numpy as np
import pytorch_lightning as pl
import rich
import torch
from tqdm import tqdm
import transformers
from typing import *
import ujson as json

import pretty_traceback
pretty_traceback.install()

import datagen
import our_data_collator
import our_datasets
import our_metrics
import our_tokenizer
import modded_bart
import multi_threaded
print("Done loading modules")

OUR_DEBUG = True
SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)

#########################################################################################################
MODE = "per_batch"
USE_CACHE = False  # Currently doesn't work, because of the way the positional embeddings are calculated.
FREEFORM_OPTIONS = [False, True]
NUM_BATCHES_VALID = 30
#########################################################################################################
PIPE_TYPE = multi_threaded.make_thread_safe_pipes
POOL_CONSTRUCTOR = lambda num_workers: dummy.Pool(processes=num_workers)
CONC_MODE = "yield"
LOOP_WAIT_SEC = 0
BATCH_SIZE = 1024
MULTIPROCESSING_METHOD = None
#########################################################################################################


def clean_sample_for_logging(tokens, tokenizer):
    tokens_list = tokens.cpu().numpy().tolist()
    tokens_list = [x for x in tokens_list if x != -100]

    as_string = tokenizer.decode(tokens_list, ignore_special_symbols=False)

    return as_string.replace("<pad>", "[bright_cyan]<p>[/bright_cyan]")


def zip_dicts(*dicts):
    """
    Zips the iterables in the values of the dicts by returning a dict with 
    the same keys and a set of value at each iteration.
    """
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


def prep_return_data(output, decoder_input_ids, tokenizer):
    assert len(output.shape) == 1, output.shape
    list_output = output.tolist()
    list_output_filtered = list_output[len(decoder_input_ids):]
    good_output = tokenizer.decode(list_output_filtered, ignore_special_symbols=True)
    if good_output and good_output[-1] == ")":
        good_output = good_output[:-1]
    return good_output.replace("<eos>", "").strip()


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
            kwargs = {
                k: w for k, w in kwargs.items() 
                if k != "decoder_input_ids_for_gen" and k != "decoder_attention_mask_for_gen"
            }
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self.logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        # self.allen_ai_bart.training = False
        loss: torch.Tensor = self(**batch).loss
        things_to_log = dict(eval_loss=loss)

        if batch_idx % 100 == 0:
            if MODE == "per_batch":
                preds = {}
                for is_freeform in FREEFORM_OPTIONS:
                    generation_kwargs = dict(**self.generation_kwargs)
                    if not is_freeform:
                        generation_kwargs["decoder_input_ids"] =      batch["decoder_input_ids_for_gen"]
                        # generation_kwargs["decoder_attention_mask"] = batch["decoder_input_ids_for_gen"] != self.tokenizer.pad_token_id
                        
                    preds[is_freeform] = self.model.generate(
                        input_ids=              batch["input_ids"], 
                        attention_mask=         batch["attention_mask"], 
                        **generation_kwargs, 
                    )

                    loop_package = zip_dicts(batch)

            elif MODE == "per_unit":
                loop_package = zip_dicts(batch)
            else:
                raise ValueError(f"Unknown mode {MODE}")

            em = {k: our_metrics.EM() for k in FREEFORM_OPTIONS}
            rich.print("[red bold]FIX THE LENGTH SHIT")
            for i, pack in enumerate(tqdm(loop_package, desc="Validating", total=len(batch["input_ids"]))):
                do_print = batch_idx == 0 and i < 5
                
                if do_print:
                    rich.print(f"[blue]" + "#" * 80)

                for is_freeform in FREEFORM_OPTIONS:
                    freeform_maybe = "_freeform" if is_freeform else ""

                    if i >= self.max_generation_quantity_valid:
                        break
                    
                    if MODE == "per_batch":
                        sample = pack
                        pred = preds[is_freeform][i]

                    if MODE == "per_unit":
                        sample = pack
                        generation_kwargs = dict(**self.generation_kwargs)

                        if "decoder_input_ids_for_gen" in pack and not is_freeform :
                            filtered = [
                                x for x in sample["decoder_input_ids_for_gen"] 
                                if x != self.tokenizer.pad_token_id
                            ]

                            for x in filtered:
                                assert x < len(self.tokenizer.vocab), x

                            
                            generation_kwargs["decoder_input_ids"] = (
                                torch.stack(filtered).reshape(1, -1)[:, :generation_kwargs["max_length"] - 1]
                            )
                        
                        assert sample["input_ids"] is not None
                        assert sample["attention_mask"] is not None
                        
                        pred = self.model.generate(
                            input_ids=sample["input_ids"].reshape(1, -1), 
                            attention_mask=sample["attention_mask"].reshape(1, -1), 
                            **generation_kwargs, 
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
                        rich.print(f"[blue]Inputs{freeform_maybe}:         \"{clean_sample_for_logging(sample['input_ids'], self.tokenizer)}\"")
                        if "decoder_input_ids_for_gen" in generation_kwargs:
                            assert len(generation_kwargs['decoder_input_ids']) == 1, generation_kwargs['decoder_input_ids'].shape
                            rich.print(f"[blue]Decoder Inputs{freeform_maybe}: \"{clean_sample_for_logging(generation_kwargs['decoder_input_ids'][0], self.tokenizer)}\"")
                        rich.print(f"[blue]Gen{freeform_maybe}:            \"{clean_sample_for_logging(pred, self.tokenizer)}\"")
                        rich.print(f"[blue]Label{freeform_maybe}:          \"{clean_sample_for_logging(sample['labels'], self.tokenizer)}")

                    em[is_freeform].add(clean_pred, clean_label, do_print=do_print, descr=freeform_maybe)
                
            ###############################################################
            # Compute and print metrics
            ###############################################################
            for is_freeform in FREEFORM_OPTIONS:
                freeform_maybe = "_freeform" if is_freeform else ""
                em_acc_val = em[is_freeform].compute()
                things_to_log[f"EM{freeform_maybe}"] = em_acc_val
                ratio = em[is_freeform].correct / em[is_freeform].total
                rich.print(f"[orange]GOT{freeform_maybe} {em[is_freeform].correct}/{em[is_freeform].total} = {ratio:.2%}\n")
            
        self.log_dict(things_to_log, **self.logging_conf)

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

    def _make_dataloader(self, ds, batch_size, shuffle, dl_name):
        collator = our_data_collator.DataCollatorWithDecoderInputIds(
            self.tokenizer, 
            model=self.model, 
        )

        if shuffle:
            indices = np.random.permutation(len(ds))
        else:
            indices = np.arange(len(ds))

        if CONC_MODE != "yield":             
            yield from multi_threaded.start(
                indices, 
                batch_size, 
                PIPE_TYPE, 
                dl_name, 
                ds, 
                collator, 
                self.model, 
                self.tokenizer, 
                self.generation_kwargs, 
                LOOP_WAIT_SEC, 
                POOL_CONSTRUCTOR,
            )
        else:
            for i in tqdm(range(0, len(ds), batch_size), desc=f"progress in {dl_name}"):
                batch_indices = indices[i:i + batch_size]
                
                iterators = [iter(ds.get(i)) for i in batch_indices]
                send_values = [None] * len(iterators)
                final_values = [None] * len(iterators)
                prediction_batch = []
                prediction_batch_iterator_idx = []

                at_least_one = True
                iteration_no = 0
                while at_least_one:
                    iteration_no += 1
                    at_least_one = False
                    for idx, iterator in enumerate(iterators):
                        # print(f"{iteration_no = } - working on iterator #{idx}")
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Ignore this iterator if it is done
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        if final_values[idx] is not None:
                            continue
                        
                        try:
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Get the next value from the iterator, no state if first iteration.
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            if send_values[idx] is None:
                                query = next(iterator)
                            else:
                                query = iterator.send(send_values[idx])
                            
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Add to the batch pile, with meta info
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            prediction_batch.append(
                                dict(
                                    input_data=datagen.prep_input_data(
                                        tokenizer=self.tokenizer, 
                                        input_str=query["input_str"], 
                                        pseudo_without_head=query["pseudo_without_head"]
                                    ), 
                                    logging_info=query["logging_info"],
                                )
                            )
                            # Meta info
                            prediction_batch_iterator_idx.append(idx)
                            at_least_one = True

                        except StopIteration as err:
                            final_value = err.value
                            final_values[idx] = final_value

                    if prediction_batch:
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep the inputs
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        without_logging_info = [
                            dict(input_ids=x["input_data"][0], decoder_input_ids_for_gen=x["input_data"][1]) 
                            for x in prediction_batch
                        ]
                        assert isinstance(without_logging_info[0], dict), type(without_logging_info[0])

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Do the prediction
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        rich.print(f"[redbold]{len(without_logging_info) = } / {batch_size}")
                        outputs = multi_threaded.actual_prediction(
                            batch=without_logging_info,
                            collator=collator, 
                            model=self.model, 
                            tokenizer=self.tokenizer, 
                            generation_kwargs=self.generation_kwargs,
                        )

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep the outputs
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        for idx, output, input_data in zip(
                            prediction_batch_iterator_idx, 
                            outputs, 
                            without_logging_info
                        ):
                            send_values[idx] = prep_return_data(
                                output, 
                                decoder_input_ids=input_data["decoder_input_ids_for_gen"], 
                                tokenizer=self.tokenizer
                            )

                        prediction_batch = []
                        prediction_batch_iterator_idx = []

                num_nones = sum(x is None for x in final_values)
                assert num_nones == 0, f"{num_nones}/{len(final_values)} were None"
                
                yield collator(final_values)

    @beartype
    def _make_regular_dataloader(
        self, 
        ds: Union[our_datasets.MostBasicDataset, our_datasets.OracleBasicDataset, our_datasets.SelfLearnedBasicDataset], 
        batch_size: int, 
        shuffle: bool,
    ):
        return torch.utils.data.DataLoader(
                ds, 
                collate_fn=our_data_collator.DataCollatorWithDecoderInputIds(
                    self.tokenizer, 
                    model=self.model, 
                ), 
                batch_size=batch_size, 
                num_workers=self.num_workers_dl,
                shuffle=shuffle,
            ) 

    def train_dataloader(self):
        if isinstance(self.train_ds, our_datasets.SelfLearnedBasicDataset):
            return self._make_dataloader(
                self.train_ds, 
                self.batch_size,
                shuffle=True,
                dl_name="train_dl"
            )
        else:
            return self._make_regular_dataloader(
                ds=self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
            )

    def val_dataloader(self):
        if isinstance(self.eval_ds, our_datasets.SelfLearnedBasicDataset):
            return self._make_dataloader(
                self.eval_ds, 
                self.batch_size,
                shuffle=True,
                dl_name="val_dl"
            )
        else:
            return self._make_regular_dataloader(
                ds=self.eval_ds,
                batch_size=self.batch_size,
                shuffle=True,
            )

def main(
    run_name="oracle",
    dataset_type="oracle_basic_dataset", 
    dataset_path=None, 
    dicts_path=SCRIPT_DIR / "data" / "dicts.pkl",
):
    
    if MULTIPROCESSING_METHOD:
        multiprocessing.set_start_method(MULTIPROCESSING_METHOD)
    
    LEARNING_RATE = 0.001
    NUM_GPUS = 1
    
    EVAL_EVERY_N_EPOCHS = 1
    WANDB_ENTITY = "julesgm"
    WANDB_PROJECT = "self_learned_explanations"
    TRAIN_BATCH_SIZE = BATCH_SIZE
    MAX_GENERATION_QUANTITY_VALID = min(128, TRAIN_BATCH_SIZE)
    EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE
    GENERATION_MAX_LENGTH = 90
    PRECISION = 16
    GRADIENT_CLIP_VAL = 0.1

    dataset = datagen.load_dataset(dataset_path, dicts_path)
    train_ds = {k: v[:len(v) // 2] for k, v in dataset.items()}
    valid_ds = {k: v[len(v) // 2:] for k, v in dataset.items()}
    tokenizer = our_tokenizer.Tokenizer(max_length=1024, use_equal_symbol=True)
    train_torch_dataset = our_datasets.DATASET_TYPES[dataset_type](train_ds, tokenizer)
    valid_torch_dataset = our_datasets.DATASET_TYPES[dataset_type](valid_ds, tokenizer)

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

    # The only difference is that we take decoder_attention_masks into account
    # for positional embeddings
    model = modded_bart.ModifiedBartForConditionalGeneration(
        config
    )
    
    functor = multi_threaded.SendPullPredFunctor(tokenizer)
    if isinstance(train_torch_dataset, our_datasets.SelfLearnedBasicDataset):
        train_torch_dataset.set_pred_functor(functor, CONC_MODE)
    if isinstance(valid_torch_dataset, our_datasets.SelfLearnedBasicDataset):
        valid_torch_dataset.set_pred_functor(functor, CONC_MODE)

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
            use_cache=USE_CACHE
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
    try:
        logger = pl.loggers.WandbLogger(
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
        )
    except ConnectionRefusedError:
        logger = None

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision=PRECISION,
        max_epochs=1000, 
        gpus=NUM_GPUS,
        # val_check_interval=1000,
        check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
        limit_val_batches=NUM_BATCHES_VALID,
    )
    trainer.fit(pl_object)
    

if __name__ == "__main__":
    fire.Fire(main)