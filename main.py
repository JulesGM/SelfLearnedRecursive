print("Importing modules")
import dataclasses
import graphlib
import itertools
import logging
import math
from pathlib import Path
import random
import re
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
import wandb

import pretty_traceback
pretty_traceback.install()

import datagen
import our_data_collator
import our_datasets
import our_metrics
import our_tokenizer
import modded_bart
import concurrent_parts
print("Done loading modules")

SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)


class ValidModes:
    per_batch = "per_batch"
    per_sample = "per_sample"
    valid_modes = {per_batch, per_sample}


#########################################################################################################
RUN_NAME_DEFAULT  = "experimentation"
ACTIVE_MODES      = {ValidModes.per_batch}
FREEFORM_OPTIONS  = {False}
NUM_BATCHES_VALID = 5
EVAL_EVERY_N_EPOCHS = 2

N_LAYERS = 2
N_HEADS = 4
H_SIZE = 64

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
MAX_ANSWER_GEN = 5
MAX_TOTAL_GEN_LENGTH = 88
GRADIENT_CLIP_VAL = 0.1

GENERATION_KWARGS = dict(
    num_beams=1,
    use_cache=True,

    # Should never hange:
    min_length=0,
    constraints=None,
    do_sample=False,
)

def generate(model: modded_bart.ModifiedBartForConditionalGeneration, **kwargs):
    for k in GENERATION_KWARGS.keys():
        assert GENERATION_KWARGS[k] == kwargs[k], f"{k} mismatch"
    
    assert "max_length" in kwargs, "max_length not in kwargs"

    return model.generate(**kwargs)
GEN_FUNCTION = generate


#########################################################################################################
CONC_MODE = "yield"
#########################################################################################################


def clean_sample_for_logging(tokens, tokenizer):
    """
    Only removes -100 tokens values.
    """
    tokens_list = tokens.cpu().numpy().tolist()

    as_string = tokenizer.decode(tokens_list, ignore_special_symbols=False)

    
    pale_map = {
        "<-100>": "[bright_cyan]#[/bright_cyan]",
        "<pad>": "[bright_cyan]^[/bright_cyan]",
    }

    for token, new_v in pale_map.items():
        as_string = re.sub(
            r"(?<=" + re.escape(token) + r")" + 
            r"\s+"  + 
            r"(?="  + re.escape(token) + r")", 
            "", 
            as_string
        )
        as_string = as_string.replace(token, new_v)

    return as_string

def find_last(seq, item):
    return len(seq) - seq[::-1].index(item) - 1

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


def color_matching(seq_a, seq_b):
    output_a = [
        f"[green]{a}" if a == b else f"[red]{a}" for a, b 
        in itertools.zip_longest(seq_a, seq_b, fillvalue="") if a
    ]
    output_b = [
        f"[green]{b}" if a == b else f"[red]{b}" for a, b 
        in itertools.zip_longest(seq_a, seq_b, fillvalue="") if b
    ]

    return output_a, output_b


class RenewableGenerator:
    def __init__(self, fn, len):
        self.fn = fn
        self.iter = None
        self.len = len

    def __len__(self):
        return self.len

    def __iter__(self):
        self.iter = self.fn()
        return self
    
    def __next__(self):
        features = next(self.iter)
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.pin_memory()
        return features

def top_sort_build_tree(dep_dict: dict, visited_datagen: datagen.Node):
    if visited_datagen.get_children():
        dep_dict[visited_datagen] = []
        for child in visited_datagen.get_children():
            if child.get_children():
                dep_dict[visited_datagen].append(child)
                top_sort_build_tree(dep_dict, child)

def top_sort_build_tree(dep_dict: dict, visited_datagen: datagen.Node):
    if visited_datagen.get_children():
        dep_dict[visited_datagen] = []
        for child in visited_datagen.get_children():
            if child.get_children():
                dep_dict[visited_datagen].append(child)
                top_sort_build_tree(dep_dict, child)


class PLBart(pl.LightningModule):
    def __init__(self, *, 
            model, 
            tokenizer: our_tokenizer.Tokenizer, 
            train_ds, 
            eval_ds, 
            train_batch_size, 
            eval_batch_size, 
            num_workers_dl, 
            generation_kwargs,
            learning_rate,
            is_adamw,
            weight_decay,
            # scheduler_type,
            # scheduler_kwargs,
            # do_allen_nlp_predictions: bool,
            curriculum_mode: str,
            max_generation_quantity_valid: int,
        ):
        super().__init__()
        self.mask_intermediate_labels = isinstance(train_ds, our_datasets.SelfLearnedBasicDataset)
        self.shuffle_train =            False
        self.shuffle_val =              False
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

        # Curriculum
        self.curriculum_mode =          curriculum_mode
        assert "max_length" not in self.generation_kwargs, "the max length is computed dynamically"

    def on_train_epoch_start(self):
        if isinstance(self.train_ds, our_datasets.CurriculumSelfLearned):
            self.train_ds.mix(
                {
                    0: 1., 
                    1: 1., 
                    2: 1.,
                }
            )

    def on_validation_start(self) -> None:
        if isinstance(self.eval_ds, our_datasets.CurriculumSelfLearned):
            self.eval_ds.mix(
                {
                    0: 1., 
                    1: 1., 
                    2: 1.,
                }
            )

    def forward(self, **kwargs):
        if "decoder_input_ids_for_gen" in kwargs:
            kwargs = {
                k: w for k, w in kwargs.items() 
                if k != "decoder_input_ids_for_gen" and k != "decoder_attention_mask_for_gen"
            }

        assert self.model.config.bos_token_id not in kwargs["labels"]
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self.logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss: torch.Tensor = self(**batch).loss
        things_to_log = dict(eval_loss=loss)
        per_batch_preds = None
        self.model = self.model.eval()

        #######################################################################
        # Print every N batches
        #######################################################################
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Batch mode inference
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ValidModes.per_batch in ACTIVE_MODES:
            per_batch_preds = {}
            for is_freeform in FREEFORM_OPTIONS:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Prep the argumetns.
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # We copy the dict because we will modify it.
                generation_kwargs = dict(**self.generation_kwargs)
                # Freeform mode doesn't need `decode_input_ids_for_gen`, & basic mode 
                # only has a freeform mode.
                if not is_freeform:
                    assert (
                        "decoder_input_ids_for_gen" in batch or 
                        not self.eval_ds.has_decoder_input_ids_for_gen
                    )
                    
                    if self.eval_ds.has_decoder_input_ids_for_gen:
                        # We truncate the decoder_input_ids to the max length.
                        generation_kwargs["decoder_input_ids"] = (
                            batch["decoder_input_ids_for_gen"][:, :MAX_TOTAL_GEN_LENGTH])
                    
                # Freeform always potentially generates all the way. 
                # In othercases, we either generate MAX_ANSWER_GEN more tokens or 
                # stop at the end of the max total length.
                if is_freeform or not self.eval_ds.has_decoder_input_ids_for_gen:
                    max_length = MAX_TOTAL_GEN_LENGTH
                else:
                    max_length = min(
                        generation_kwargs["decoder_input_ids"].shape[1] + MAX_ANSWER_GEN, 
                        MAX_TOTAL_GEN_LENGTH
                )

                # Run inference.
                per_batch_preds[is_freeform] = GEN_FUNCTION(
                    self.model,
                    input_ids      =batch["input_ids"], 
                    attention_mask =batch["attention_mask"],
                    max_length     =max_length,
                    **generation_kwargs, 
                )

        # Initialize metrics
        em = {
            mode: {
                k: our_metrics.EM() for k in FREEFORM_OPTIONS 
                if not (mode == ValidModes.per_sample and k)
            } for mode in ACTIVE_MODES
        }

        ###################################################################
        # Examine the samples and predictions one by one.
        # In per_sample mode, also do predictions.
        ###################################################################
        for i, pack in enumerate(tqdm(
            zip_dicts(batch), desc="Validating", total=len(batch["input_ids"])
        )):
            do_print = i < 5

            if do_print:
                print("#" * 80)

            for is_freeform in FREEFORM_OPTIONS:
                if i >= self.max_generation_quantity_valid:
                    break
                
                # TODO: wtf is this
                per_batch_pred = None
                if ValidModes.per_batch in ACTIVE_MODES:
                    sample = pack
                    per_batch_pred = per_batch_preds[is_freeform][i]
                per_sample_pred = None

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Freeform mode doesn't neeed to be done in per sample mode
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                not_necessary = is_freeform and ValidModes.per_batch in ACTIVE_MODES
                if ValidModes.per_sample in ACTIVE_MODES and not not_necessary:
                    sample = pack
                    generation_kwargs = dict(**self.generation_kwargs)

                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Prep decoder input ids
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    if self.eval_ds.has_decoder_input_ids_for_gen and not is_freeform:
                        took_out_pad_tokens = [
                            x for x in sample["decoder_input_ids_for_gen"] 
                            if x != self.tokenizer.pad_token_id
                        ]
                        generation_kwargs["decoder_input_ids"] = torch.stack(
                            took_out_pad_tokens).reshape(1, -1)[:, :MAX_TOTAL_GEN_LENGTH]

                    assert sample["input_ids"] is not None
                    assert sample["attention_mask"] is not None

                    if is_freeform or not self.eval_ds.has_decoder_input_ids_for_gen:
                        max_length = MAX_TOTAL_GEN_LENGTH
                    else:
                        max_length = min(
                            generation_kwargs["decoder_input_ids"].shape[1] + MAX_ANSWER_GEN, 
                            MAX_TOTAL_GEN_LENGTH
                        )
                    

                    per_sample_pred = GEN_FUNCTION(
                        self.model,
                        input_ids      =sample["input_ids"].reshape(1, -1), 
                        attention_mask =sample["attention_mask"].reshape(1, -1), 
                        max_length     =max_length,
                        **generation_kwargs, 
                    )
                    
                    assert len(per_sample_pred) == 1, per_sample_pred.shape
                    assert len(per_sample_pred.shape) == 2, per_sample_pred.shape
                    per_sample_pred = per_sample_pred[0]

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # File the predictions per mode 
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                pred_per_mode = {}
                if ValidModes.per_sample in ACTIVE_MODES and not not_necessary:
                    pred_per_mode[ValidModes.per_sample] = per_sample_pred
                if ValidModes.per_batch in ACTIVE_MODES:
                    pred_per_mode[ValidModes.per_batch ] = per_batch_pred
                
                for mode, pred in pred_per_mode.items():
                    cleaned = our_metrics.OurMetric.prepare(
                        self.tokenizer, pred, sample["labels"], do_print=False
                    )
                    
                    clean_pred  = cleaned["cleaned_preds" ]
                    clean_label = cleaned["cleaned_labels"]

                    if self.eval_ds.has_decoder_input_ids_for_gen and not is_freeform:
                        clean_pred  = clean_pred [find_last(clean_pred , self.tokenizer.token_to_idx["="]):]
                        clean_label = clean_label[find_last(clean_label, self.tokenizer.token_to_idx["="]):]
                    
                    if do_print:
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep per-sample outputs
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        rich.print("Raw `input_ids`:\n", sample["input_ids"])
                        if not is_freeform and self.eval_ds.has_decoder_input_ids_for_gen:
                            rich.print("Raw `decoder_input_ids_for_gen`:\n", sample["decoder_input_ids_for_gen"])
                        rich.print("Raw Pred:\n", pred)
                        rich.print("Raw `labels`:\n", sample["labels"])

                        cleaned_inputs = clean_sample_for_logging(sample['input_ids'], self.tokenizer)
                        cleaned_labels = clean_sample_for_logging(sample['labels']   , self.tokenizer)
                        cleaned_gen    = clean_sample_for_logging(pred               , self.tokenizer)
                        cleaned_decoder_input_ids = None
                        
                        if "decoder_input_ids" in generation_kwargs:
                            if mode == ValidModes.per_sample:
                                assert len(generation_kwargs["decoder_input_ids"].shape) == 2, (
                                    generation_kwargs['decoder_input_ids'].shape)
                                assert generation_kwargs["decoder_input_ids"].shape[0] == 1, (
                                    generation_kwargs['decoder_input_ids'].shape)
                                cleaned_decoder_input_ids = clean_sample_for_logging(
                                    generation_kwargs["decoder_input_ids"][0], self.tokenizer)
                                ids = generation_kwargs["decoder_input_ids"][0]
                            elif ValidModes.per_batch:
                                cleaned_decoder_input_ids = clean_sample_for_logging(
                                    sample["decoder_input_ids_for_gen"], self.tokenizer)
                                ids = sample["decoder_input_ids_for_gen"]
                            else:
                                raise ValueError(f"Unknown mode {mode}")

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Print them
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # rich.print(f"[black bold]{mode} {'Freeform' if is_freeform else 'Not-Freeform'}:")
                        rich.print(f"Inputs:\n[blue]\"{cleaned_inputs}\"")

                        if cleaned_decoder_input_ids:
                            rich.print(f"Decoder Inputs:\n[blue]\"{cleaned_decoder_input_ids}\"")

                        rich.print(f"Gen:\n[blue]\"{cleaned_gen}\"")
                        rich.print(f"Label:\n[blue]     \"{cleaned_labels}\"")

                        idx_colored_a, idx_colored_b = color_matching(clean_pred, clean_label)
                        rich.print(f"clean_pred:  ", clean_pred)
                        rich.print(f"clean_label: ", clean_label)
                        rich.print(f"(EM) Answer: " + ", ".join(idx_colored_a))
                        rich.print(f"(EM) Label:  "  + ", ".join(idx_colored_b))
                        print("~" * 80)

                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update Metrics
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if not (mode == ValidModes.per_sample and is_freeform):
                        em[mode][is_freeform].add(
                            clean_pred, clean_label, do_print=False, descr=None,
                        )
            
        ###############################################################
        # Compute and print per batch metrics
        ###############################################################
        for is_freeform in FREEFORM_OPTIONS:
            for mode in ACTIVE_MODES:
                if is_freeform and mode == ValidModes.per_sample:
                    continue
                
                em_obj = em[mode][is_freeform]


                freeform_maybe = (
                    "_freeform" if (is_freeform and not mode == ValidModes.per_sample) else ""
                )
                em_acc_val = em_obj.compute()
                things_to_log[f"EM{freeform_maybe}"] = em_acc_val
                ratio = em_obj.correct / em_obj.total
                rich.print(f"[orange]GOT{freeform_maybe}_{mode} {em_obj.correct}/{em_obj.total} = {ratio:.2%}\n")
    
        self.model = self.model.train()
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
            max_length=self.tokenizer.max_length,
            mask_intermediate_labels=self.mask_intermediate_labels,
        )

        if shuffle:
            indices = np.random.permutation(len(ds))
        else:
            indices = np.arange(len(ds))

        
        if CONC_MODE == "yield":
            for i in tqdm(range(0, len(ds), batch_size), desc=f"progress in {dl_name}"):
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # 
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                        # rich.print(f"[redbold]{len(without_logging_info) = } / {batch_size}")
                        outputs = concurrent_parts.actual_prediction(
                            batch=without_logging_info,
                            collator=collator, 
                            model=self.model, 
                            generation_kwargs=self.generation_kwargs,
                            gen_function=GEN_FUNCTION,
                            max_answer_gen=MAX_ANSWER_GEN,
                            max_total_gen_length=MAX_TOTAL_GEN_LENGTH,
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

        if CONC_MODE == "top_sort":
            assert False
            for i in tqdm(range(0, len(ds), batch_size), desc=f"progress in {dl_name}"):
                batch_indices = indices[i:i + batch_size]
                root_nodes = [ds.get(i) for i in batch_indices]
                sorters = [] # 1 to 1 with root nodes, ok

                for root_node in root_nodes:
                    tree = {}
                    top_sort_build_tree(tree, root_node)
                    sorter = graphlib.TopologicalSorter(tree)
                    sorter.prepare()
                    sorters.append(sorter)
                
                at_least_one = True
                while at_least_one:
                    at_least_one = False
                    node_batch: List[datagen.Node] = []
                    nodes_to_sorters = {}
                    for sorter in sorters:
                        if sorter.is_active():
                            nodes = sorter.get_ready()
                            for node in nodes:
                                nodes_to_sorters[node] = sorter
                            node_batch.extend()

                    input_ids = [self.tokenizer(node.get_input_str()) for node in node_batch]
                    decoder_input_ids = [self.tokenizer(node.get_pseudo_topsort_query()) for node in node_batch]
                    batch = dict(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
                    outputs = concurrent_parts.actual_prediction(
                            batch=all_queries,
                            collator=collator, 
                            model=self.model, 
                            generation_kwargs=self.generation_kwargs,
                            gen_function=GEN_FUNCTION,
                            max_answer_gen=MAX_ANSWER_GEN,
                            max_total_gen_length=MAX_TOTAL_GEN_LENGTH,
                        )

                    for node, output in zip(sorters, node_batch, outputs):
                        text = self.tokenizer.decode(output.detach().cpu().numpy().tolist())
                        node.set_pseudo_value(text.split("=")[-1])
                        node_to_sorter[node].done(node)
                
                    for sorter in sorters:
                        if sorter.is_active():
                            at_least_one = True
                            break

                # TODO: Build final values.
                # TODO: This means, do the -100 stuff.
                yield 

    @beartype
    def _make_regular_dataloader(
        self, 
        ds: Union[
            our_datasets.MostBasicDataset, 
            our_datasets.OracleBasicDataset, 
            our_datasets.SelfLearnedBasicDataset
        ], 
        batch_size: int, 
        shuffle: bool,
    ):
        return torch.utils.data.DataLoader(
                ds, 
                collate_fn=our_data_collator.DataCollatorWithDecoderInputIds(
                    self.tokenizer, 
                    model=self.model, 
                    max_length=self.tokenizer.max_length,
                    mask_intermediate_labels=self.mask_intermediate_labels,
                ), 
                batch_size=batch_size, 
                num_workers=self.num_workers_dl,
                shuffle=shuffle,
            ) 

    def train_dataloader(self):
        if isinstance(self.train_ds, our_datasets.SelfLearnedBasicDataset):
            if self.train_ds.has_len():
                num_batches = math.ceil(len(self.train_ds) / self.batch_size)
            else:
                num_batches = None

            return RenewableGenerator(
                lambda: self._make_dataloader(
                    self.train_ds, 
                    self.batch_size,
                    shuffle=self.shuffle_train,
                    dl_name="train_dl"
                ),
                len=num_batches,
            )
        else:
            return self._make_regular_dataloader(
                ds=self.train_ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
            )

    def val_dataloader(self):
        if isinstance(self.eval_ds, our_datasets.SelfLearnedBasicDataset):
            if self.eval_ds.has_len():
                num_batches = math.ceil(len(self.eval_ds) / self.batch_size)
            else:
                num_batches = None

            return RenewableGenerator(
                lambda: self._make_dataloader(
                    self.eval_ds, 
                    self.batch_size,
                    shuffle=self.shuffle_val,
                    dl_name="val_dl"
                ),
                len=num_batches,
            )
        else:
            return self._make_regular_dataloader(
                ds=self.eval_ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle_val,
            )

def main(
    run_name=RUN_NAME_DEFAULT,
    dataset_type="oracle_basic_dataset", 
    dataset_path=None, 
    dicts_path=SCRIPT_DIR / "data" / "dicts.pkl",
): 
    NUM_GPUS = 1
    WANDB_ENTITY = "julesgm"
    WANDB_PROJECT = "self_learned_explanations"
    TRAIN_BATCH_SIZE = BATCH_SIZE
    MAX_GENERATION_QUANTITY_VALID = min(128, TRAIN_BATCH_SIZE)
    EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE
    PRECISION = 16

    dataset = datagen.load_dataset(dataset_path, dicts_path)
    
    for v in dataset.values():
        random.shuffle(v)

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
    assert "pad_token_id" in config.__dict__
    assert "eos_token_id" in config.__dict__
    assert "bos_token_id" in config.__dict__
    assert "forced_bos_token_id" in config.__dict__
    assert "forced_eos_token_id" in config.__dict__
    assert "decoder_start_token_id" in config.__dict__


    config.pad_token_id = tokenizer.pad_token_id
    config.decoder_start_token_id = tokenizer.decoder_start_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    assert config.eos_token_id != config.pad_token_id, (
        f"eos_token_id and pad_token_id should not be the same. eos_token_id:"
        f" {config.eos_token_id}, pad_token_id: {config.pad_token_id}"
    )
    config.forced_bos_token_id = None
    config.forced_eos_token_id = None
    config.vocab_size = len(tokenizer.vocab)
    config.task_specific_params = {}

    ###############################################################
    # Can change
    ###############################################################
    config.num_hidden_layers = N_LAYERS # Default is 6
    config.hidden_size = H_SIZE # Default is 768
    config.encoder_attention_heads = N_HEADS # Default is 16
    config.decoder_attention_heads = N_HEADS # Default is 16
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
    
    assert type(train_torch_dataset) == type(valid_torch_dataset), (
        type(train_torch_dataset), type(valid_torch_dataset)
    )
    assert CONC_MODE == "yield"
    
    if (
        isinstance(train_torch_dataset, our_datasets.SelfLearnedBasicDataset) and 
        isinstance(valid_torch_dataset, our_datasets.SelfLearnedBasicDataset)
    ):
        train_torch_dataset.set_conc_mode(CONC_MODE)
        valid_torch_dataset.set_conc_mode(CONC_MODE)
        train_torch_dataset.set_mask_intermediate_labels(True)
        valid_torch_dataset.set_mask_intermediate_labels(True)

        assert CONC_MODE == "yield"
        

    pl_object = PLBart(
        model=model, 
        tokenizer=tokenizer, 
        train_ds=train_torch_dataset, 
        eval_ds=valid_torch_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers_dl=0,
        generation_kwargs=GENERATION_KWARGS,
        learning_rate=LEARNING_RATE,
        is_adamw=True,
        weight_decay=WEIGHT_DECAY,
        # scheduler_type="WarmupLinear",
        # scheduler_kwargs=dict(),
        # do_allen_nlp_predictions=False,
        curriculum_mode=None,
        max_generation_quantity_valid=MAX_GENERATION_QUANTITY_VALID,
    )

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
            approach_type=train_torch_dataset.__class__.__name__, 
            precision=PRECISION, bart_config=vars(config), 
            generation_kwargs=GENERATION_KWARGS,
        )
    )
    wandb.run.log_code(SCRIPT_DIR)

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision=PRECISION,
        max_epochs=1000, 
        gpus=NUM_GPUS,
        check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
        limit_val_batches=NUM_BATCHES_VALID,
    )
    trainer.fit(pl_object)
    

if __name__ == "__main__":
    fire.Fire(main)