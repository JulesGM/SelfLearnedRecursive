print("Importing modules.")

from beartype.typing import *

import collections
import dataclasses
import graphlib
import inspect  # Only used for tests
import itertools
import logging
import math
from pathlib import Path
import random
import re
import time

from beartype import beartype
import fire  # type: ignore
import jsonlines as jsonl  # type: ignore
import numpy as np
import os
import pytorch_lightning as pl
import rich
import torch
from tqdm import tqdm  # type: ignore
import transformers

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]
    
import wandb

import pretty_traceback  # type: ignore
pretty_traceback.install()

import bart_relative_attention
import data_generation_arithmetic
import our_data_collator
import our_datasets
import our_metrics
import our_tokenizer
import bart_modified
import utils
print("Done loading modules.\n")


SCRIPT_DIR = Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)
DEBUG = os.environ.get("DEBUG", "False") == "True"


class ValidModes:
    per_batch = "per_batch"
    per_sample = "per_sample"
    valid_modes = {per_batch, per_sample}


#########################################################################################################
ACTIVE_MODES = {ValidModes.per_batch}
NUM_SAMPLES_VALID = 20000
EVAL_EVERY_N_EPOCHS = 1
DETERMINISTIC = True

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0

GRADIENT_CLIP_VAL = 0.1

GENERATION_KWARGS = dict(
    num_beams=1,
    use_cache=True,

    # Should never hange:
    min_length=0,
    constraints=None,
    do_sample=False,
)

# Stuff that should never change
NUM_GPUS = 1
WANDB_ENTITY = "julesgm"
WANDB_PROJECT = "self_learned_explanations"
PRECISION = 16
DATA_PATH = SCRIPT_DIR / "data"



def generate(model: transformers.PreTrainedModel, **kwargs):
    assert isinstance(model, bart_modified.ModifiedBartForConditionalGeneration), "only type currently supported"

    for k in GENERATION_KWARGS.keys():
        assert GENERATION_KWARGS[k] == kwargs[k], f"{k} mismatch"

    assert "max_length" in kwargs, "max_length not in kwargs"

    return model.generate(**kwargs)


GEN_FUNCTION = generate


#########################################################################################################
CONC_MODE = "top_sort"
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
            r"(?<="
            + re.escape(token)
            + r")"
            + r"\s+"
            + r"(?="
            + re.escape(token)
            + r")",
            "",
            as_string,
        )
        as_string = as_string.replace(token, new_v)

    return as_string


def prep_return_data(output, decoder_input_ids, tokenizer):
    assert len(output.shape) == 1, output.shape
    list_output = output.tolist()
    list_output_filtered = list_output[len(decoder_input_ids) :]
    good_output = tokenizer.decode(list_output_filtered, ignore_special_symbols=True)
    if good_output and good_output[-1] == ")":
        good_output = good_output[:-1]
    return good_output.replace("<eos>", "").strip()


def color_matching(seq_a, seq_b):
    output_a = [
        f"[green]{a}" if a == b else f"[red]{a}"
        for a, b in itertools.zip_longest(seq_a, seq_b, fillvalue="")
        if a
    ]
    output_b = [
        f"[green]{b}" if a == b else f"[red]{b}"
        for a, b in itertools.zip_longest(seq_a, seq_b, fillvalue="")
        if b
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


def top_sort_build_tree(
    dep_dict: dict, 
    visited_datagen: data_generation_arithmetic.Node
):
    if visited_datagen.get_children():
        dep_dict[visited_datagen] = []
        for child in visited_datagen.get_children():
            if child.get_children():
                dep_dict[visited_datagen].append(child)
                top_sort_build_tree(dep_dict, child)

def actual_prediction(
    batch: dict[str, torch.Tensor], 
    collator: transformers.data.data_collator.DataCollatorMixin, 
    model: transformers.PreTrainedModel, 
    generation_kwargs: dict[str, Any], 
    gen_function: Callable,
):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Use the Data Collator on the data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    batch = collator(batch)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cudify everything
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for k, v in batch.items():
        batch[k] = v.cuda()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create the decoder_attention_mask
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if "decoder_attention_mask" in batch:
        batch["decoder_attention_mask"] = batch["decoder_attention_mask_for_gen"]
        del batch["decoder_attention_mask_for_gen"]

    bound_length = min(
        model.config.max_length - 6, batch["decoder_input_ids_for_gen"].shape[1] - 1
    )
    batch["decoder_input_ids"] = batch["decoder_input_ids_for_gen"][:, :bound_length]
    rich.print("[red bold]FIX DECODER LENGTH STUFF AND DECODER HEAD")
    del batch["decoder_input_ids_for_gen"]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start = time.perf_counter()
    output = gen_function(
        model=model,
        **batch,
        **generation_kwargs,
    )

    delta = time.perf_counter() - start

    print(f"Generation took {delta:.5f} seconds, {delta / output.shape[0]}s per item.")
    return output


def json_dumper_default(obj: Any) -> list[Union[int, float]]:
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    raise TypeError("Type not serializable")


class _PLBart(pl.LightningModule):
    def __init__(
        self,
        *,
        model: transformers.PreTrainedModel,
        tokenizer: our_tokenizer.Tokenizer,
        train_ds,
        eval_ds,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers_dl: int,
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        is_adamw: bool,
        weight_decay: Optional[float],
        # scheduler_type,
        # scheduler_kwargs,
        # do_allen_nlp_predictions: bool,
        freeform_options: set[bool],
        max_total_length_gen: int,
        max_answer_gen: int,
        max_depth: int,
        do_log_results: bool,
        path_log_results: Path,
        extra_info_file: Path,
    ):
        super().__init__()
        self._model: transformers.PreTrainedModel = model
        self._tokenizer: Final = tokenizer
        self._batch_size: Final[int] = train_batch_size
        self._eval_batch_size: Final[int] = eval_batch_size
        self._generation_kwargs: Final[dict[str, Any]] = generation_kwargs
        self._logging_conf: Final[dict[str, bool]] = dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        self._freeform_options: Final[set[bool]] = freeform_options
        self._extra_info_file: Final[Path] = extra_info_file

        ################################################################################
        # Related to datasets
        ################################################################################
        # These things are defined in the dataset
        self._mask_intermediate_labels = isinstance(
            train_ds, our_datasets.SelfLearnedBasicDataset
        )
        self._max_depth: Final[int] = max_depth
        self._max_answer_gen: Final[int] = max_answer_gen
        self._max_total_length_gen: Final[int] = max_total_length_gen
        self._shuffle_train: Final[bool] = True
        self._shuffle_val: Final[bool] = False
        assert train_ds is not eval_ds, "train_ds and eval_ds must be different objects"
        self._train_ds: Final = train_ds
        self._eval_ds: Final = eval_ds
        self._num_workers_dl: Final[int] = num_workers_dl
        
        ################################################################################
        # Rel. to logging results for answer overlap estim.
        ################################################################################
        self._do_log_results: Final[bool] = do_log_results
        self._path_log_results: Final[Optional[Path]] = path_log_results
        self._results_to_log: Optional[dict[str, dict[bool, dict[str, torch.Tensor]]]] = {}

        ################################################################################
        # Specific to the optimizer, its scheduler
        ################################################################################
        self._learning_rate: Final[float] = learning_rate
        self._is_adamw: Final[bool] = is_adamw
        self._weight_decay: Final[Optional[float]] = weight_decay

        # Related to the scheduler:
        # self.scheduler_type =         scheduler_type
        # self.scheduler_kwargs =       scheduler_kwargs

        assert (
            "max_length" not in self._generation_kwargs
        ), "the max length is computed dynamically"

    def on_train_epoch_start(self):
        if isinstance(self._train_ds, our_datasets.CurriculumSelfLearned):
            self._train_ds.mix(
                {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                }
            )

    def on_validation_start(self) -> None:
        if isinstance(self._eval_ds, our_datasets.CurriculumSelfLearned):
            self._eval_ds.mix(
                {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                }
            )

    def on_train_epoch_end(self) -> None:
        with self._extra_info_file.open("w") as f:
            json.dump(dict(  # type: ignore[call-arg]
                torch_rng_state=torch.random.get_rng_state(),
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
                wandb_run_id=wandb.run.id,
            ), f, default=json_dumper_default)

    def forward(self, **kwargs):
        if "decoder_input_ids_for_gen" in kwargs:
            kwargs_filtered = {
                k: w
                for k, w in kwargs.items()
                if k != "decoder_input_ids_for_gen"
                and k != "decoder_attention_mask_for_gen"
            }
        else:
            kwargs_filtered = kwargs

        assert "decoder_input_ids" in kwargs_filtered, "'decoder_input_ids' is required"
        assert self._model.config.bos_token_id not in kwargs_filtered["labels"]

        assert torch.all(
            kwargs_filtered["decoder_input_ids"][:, 0]
            == self._model.config.decoder_start_token_id
        )
        return self._model(**kwargs_filtered)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self._logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, list[str]]], batch_idx):  # type: ignore[override]
        idents = cast(list[str], batch.pop("idents"))
        
        loss: torch.Tensor = self(**batch).loss
        things_to_log = dict(eval_loss=loss)
        per_batch_preds = None
        self._model: transformers.PreTrainedModel = self._model.eval()  # type: ignore[no-redef]

        #######################################################################
        # Print every N batches
        #######################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Batch mode inference
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ValidModes.per_batch in ACTIVE_MODES:
            per_batch_preds = {}
            for is_freeform in self._freeform_options:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Prep the argumetns.
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # We copy the dict because we will modify it.
                generation_kwargs = dict(**self._generation_kwargs)
                # Freeform mode doesn't need `decode_input_ids_for_gen`, & basic mode
                # only has a freeform mode.
                if not is_freeform:
                    assert (
                        "decoder_input_ids_for_gen" in batch
                        or not self._eval_ds.has_decoder_input_ids_for_gen
                    )

                    if self._eval_ds.has_decoder_input_ids_for_gen:
                        # We truncate the decoder_input_ids to the max length.
                        generation_kwargs["decoder_input_ids"] = cast(torch.Tensor, batch[
                            "decoder_input_ids_for_gen"
                        ])[:, :self._max_total_length_gen]

                # Freeform always potentially generates all the way.
                # In othercases, we either generate MAX_ANSWER_GEN more tokens or
                # stop at the end of the max total length.
                if is_freeform or not self._eval_ds.has_decoder_input_ids_for_gen:
                    max_length = self._max_total_length_gen
                else:
                    max_length = min(
                        generation_kwargs["decoder_input_ids"].shape[1]
                        + self._max_answer_gen,
                        self._max_total_length_gen,
                    )

                # Run inference.
                per_batch_preds[is_freeform] = GEN_FUNCTION(
                    self._model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_length,
                    **generation_kwargs,
                )

        # Initialize metrics
        em = {
            mode: {
                k: {"all": our_metrics.EM()} | {level: our_metrics.EM() for level in range(1, self._max_depth + 1)}
                for k in self._freeform_options
                if not (mode == ValidModes.per_sample and k)
            } 
            for mode in ACTIVE_MODES
        }
        
        ###################################################################
        # Examine the samples and predictions one by one.
        # In per_sample mode, also do predictions.
        ###################################################################

        big_counter = []
        preds: DefaultDict[str, dict[bool, dict[str, torch.Tensor]]] = collections.defaultdict(dict)
        for i, pack in enumerate(
            tqdm(utils.zip_dicts(batch), desc="Validating", total=len(batch["input_ids"]))
        ):

            do_print = i < 5
            if do_print:
                print("#" * 80)

            for is_freeform in self._freeform_options:
                # TODO: wtf is this
                per_batch_pred = None
                if ValidModes.per_batch in ACTIVE_MODES:
                    sample = pack
                    per_batch_pred = per_batch_preds[is_freeform][i]
                per_sample_pred = None

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Freeform mode doesn't neeed to be done in per sample mode
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                not_necessary = is_freeform and ValidModes.per_batch in ACTIVE_MODES
                if ValidModes.per_sample in ACTIVE_MODES and not not_necessary:
                    sample = pack
                    generation_kwargs = dict(**self._generation_kwargs)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Prep decoder input ids
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    if self._eval_ds.has_decoder_input_ids_for_gen and not is_freeform:
                        took_out_pad_tokens = [
                            x
                            for x in sample["decoder_input_ids_for_gen"]
                            if x != self._tokenizer.pad_token_id
                        ]
                        generation_kwargs["decoder_input_ids"] = torch.stack(
                            took_out_pad_tokens
                        ).reshape(1, -1)[:, :self._max_total_length_gen]

                    assert sample["input_ids"] is not None
                    assert sample["attention_mask"] is not None

                    if is_freeform or not self._eval_ds.has_decoder_input_ids_for_gen:
                        max_length = self._max_total_length_gen
                    else:
                        max_length = min(
                            generation_kwargs["decoder_input_ids"].shape[1]
                            + self._max_answer_gen,
                            self._max_total_length_gen,
                        )

                    per_sample_pred = GEN_FUNCTION(
                        self._model,
                        input_ids=sample["input_ids"].reshape(1, -1),
                        attention_mask=sample["attention_mask"].reshape(1, -1),
                        max_length=max_length,
                        **generation_kwargs,
                    )

                    assert len(per_sample_pred) == 1, per_sample_pred.shape
                    assert len(per_sample_pred.shape) == 2, per_sample_pred.shape
                    per_sample_pred = per_sample_pred[0]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # File the predictions per mode
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                pred_per_mode = {}
                if ValidModes.per_sample in ACTIVE_MODES and not not_necessary:
                    pred_per_mode[ValidModes.per_sample] = per_sample_pred
                if ValidModes.per_batch in ACTIVE_MODES:
                    pred_per_mode[ValidModes.per_batch] = per_batch_pred

                preds[idents[i]][is_freeform] = pred_per_mode

                for mode, pred in pred_per_mode.items():
                    cleaned = our_metrics.OurMetric.prepare(
                        self._tokenizer, pred, sample["labels"], do_print=False
                    )

                    clean_pred = cleaned["cleaned_preds"]
                    clean_label = cleaned["cleaned_labels"]

                    if self._eval_ds.has_decoder_input_ids_for_gen and not is_freeform:
                        clean_pred = clean_pred[
                            utils.find_last(clean_pred, self._tokenizer.token_to_idx["="]) :
                        ]
                        clean_label = clean_label[
                            utils.find_last(clean_label, self._tokenizer.token_to_idx["="]) :
                        ]

                    if do_print:
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep per-sample outputs
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # rich.print("Raw `input_ids`:\n", sample["input_ids"])
                        # if (
                        #     not is_freeform
                        #     and self.eval_ds.has_decoder_input_ids_for_gen
                        # ):
                        #     rich.print(
                        #         "Raw `decoder_input_ids_for_gen`:\n",
                        #         sample["decoder_input_ids_for_gen"],
                        #     )
                        # rich.print("Raw Pred:\n", pred)
                        # rich.print("Raw `labels`:\n", sample["labels"])

                        print_cleaned_inputs = clean_sample_for_logging(
                            sample["input_ids"], self._tokenizer
                        )
                        print_cleaned_labels = clean_sample_for_logging(
                            sample["labels"], self._tokenizer
                        )
                        print_cleaned_gen = clean_sample_for_logging(pred, self._tokenizer)
                        print_cleaned_decoder_input_ids = None

                        if "decoder_input_ids" in generation_kwargs:
                            if mode == ValidModes.per_sample:
                                assert (
                                    len(generation_kwargs["decoder_input_ids"].shape)
                                    == 2
                                ), generation_kwargs["decoder_input_ids"].shape
                                assert (
                                    generation_kwargs["decoder_input_ids"].shape[0] == 1
                                ), generation_kwargs["decoder_input_ids"].shape
                                print_cleaned_decoder_input_ids = clean_sample_for_logging(
                                    generation_kwargs["decoder_input_ids"][0],
                                    self._tokenizer,
                                )
                            elif ValidModes.per_batch:
                                print_cleaned_decoder_input_ids = clean_sample_for_logging(
                                    sample["decoder_input_ids_for_gen"], self._tokenizer
                                )
                            else:
                                raise ValueError(f"Unknown mode {mode}")

                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Print them
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # rich.print(f"[black bold]{mode} {'Freeform' if is_freeform else 'Not-Freeform'}:")
                        # rich.print(f'Inputs:\n[bright_cyan]"{print_cleaned_inputs}"')

                        # if cleaned_decoder_input_ids:
                        #     rich.print(
                        #         f'Decoder Inputs:\n[bright_cyan]"{cleaned_decoder_input_ids}"'
                        #     )

                        # rich.print(f'Gen:\n[bright_cyan]"{print_cleaned_gen}"')
                        # rich.print(f'Label:\n[bright_cyan]     "{print_cleaned_labels}"')

                        # idx_colored_a, idx_colored_b = color_matching(
                        #     clean_pred, clean_label
                        # )
                        # rich.print(f"clean_pred:  ", clean_pred)
                        # rich.print(f"clean_label: ", clean_label)
                        # rich.print(f"(EM) Answer: " + ", ".join(idx_colored_a))
                        # rich.print(f"(EM) Label:  " + ", ".join(idx_colored_b))
                        # print("~" * 80)


                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update Metrics
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if not (mode == ValidModes.per_sample and is_freeform):
                        # TODO: This is slow but it is precise.
                        
                        level = data_generation_arithmetic.tree_depth_from_ids(sample["input_ids"], self._tokenizer)
                        big_counter.append(level)

                        assert level > 0, level
                        assert level <= self._max_depth, level
                        em[mode][is_freeform][level].add(
                            clean_pred,
                            clean_label,
                            do_print=False,
                            descr=None,
                        )
                        em[mode][is_freeform]["all"].add(
                            clean_pred,
                            clean_label,
                            do_print=False,
                            descr=None,
                        )

        ###############################################################
        # Compute and print per batch metrics
        ###############################################################
        for is_freeform in self._freeform_options:
            for mode in ACTIVE_MODES:
                if is_freeform and mode == ValidModes.per_sample:
                    continue
                
                # Make sure that the totals make sense
                if DEBUG:
                    sum_sub_totals = sum([
                        level_metric.total for name, level_metric in 
                        em[mode][is_freeform].items() if isinstance(name, int)
                    ])
                    assert em[mode][is_freeform]["all"].total == sum_sub_totals, (
                        em[mode][is_freeform]["all"].total, sum_sub_totals
                    ) 

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Per level or all ...
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                freeform_str_maybe = (
                    "freeform_"
                    if (is_freeform and not mode == ValidModes.per_sample)
                    else ""
                )
                for em_level in em[mode][is_freeform]:
                    em_obj = em[mode][is_freeform][em_level]
                    header = f"{freeform_str_maybe}{em_level}_{mode}"
                    
                    if em_obj.total:   
                        em_acc_val = em_obj.compute()
                        ratio = em_obj.correct / em_obj.total
                        text_ratio = f"{em_obj.correct}/{em_obj.total}"
                        
                        things_to_log_key = f"EM_{header}"
                        assert things_to_log_key not in things_to_log
                        things_to_log[things_to_log_key] = em_acc_val
                        rich.print(
                            f"GOT {header} {text_ratio} = {ratio:.2%}\n"
                        )
                    else:
                        rich.print(
                            f"GOT {header} -/0 = -\n"
                        )

        if self._do_log_results:
            # Add the results of the batch to the accumulated results of the epoch
            if DEBUG:
                joint_idents = preds.keys() & self._results_to_log.keys()
                assert not joint_idents, joint_idents
            self._results_to_log.update(preds)

        self.log_dict(things_to_log, **self._logging_conf)  # type: ignore[arg-type]
        self._model = self._model.train()  # type: ignore[no-redef]

    def on_validation_epoch_end(self) -> None:
        with jsonl.open(self._path_log_results, "a", dumps=lambda x: json.dumps(x, allow_nan=False, default=json_dumper_default)) as f:  # type: ignore[call-arg]
            f.write(dict(
                epoch=self.current_epoch,
                results=self._results_to_log,
            ))

        self._results_to_log = {}

    def configure_optimizers(self):
        """
        See ref
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        if self._is_adamw:
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adam

        optimizer = optimizer_class(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
            capturable=True,
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
        assert False, "Not tested in a while"
        collator = our_data_collator.DataCollatorWithDecoderInputIds(
            self._tokenizer,
            model=self._model,
            max_length=self._tokenizer.max_length,
            mask_intermediate_labels=self._mask_intermediate_labels,
        )

        if shuffle:
            indices = np.random.permutation(len(ds))
        else:
            indices = np.arange(len(ds))

        if CONC_MODE == "yield":
            for i in tqdm(range(0, len(ds), batch_size), desc=f"progress in {dl_name}"):
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                batch_indices = indices[i : i + batch_size]
                pred_logger = datagen.PredLogger()
                iterators = [iter(ds.get(i, pred_logger)) for i in batch_indices]
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
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Ignore this iterator if it is done
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        if final_values[idx] is not None:
                            continue

                        try:
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Get the next value from the iterator, no state if first iteration.
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            if send_values[idx] is None:
                                query = next(iterator)
                            else:
                                query = iterator.send(send_values[idx])

                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Add to the batch pile, with meta info
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            prediction_batch.append(
                                dict(
                                    input_data=datagen.prep_input_data(
                                        tokenizer=self._tokenizer,
                                        input_str=query["input_str"],
                                        pseudo_without_head=query[
                                            "pseudo_without_head"
                                        ],
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
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep the inputs
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        without_logging_info = [
                            dict(
                                input_ids=x["input_data"][0],
                                decoder_input_ids_for_gen=x["input_data"][1],
                            )
                            for x in prediction_batch
                        ]
                        assert isinstance(without_logging_info[0], dict), type(
                            without_logging_info[0]
                        )

                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Do the prediction
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # rich.print(f"[redbold]{len(without_logging_info) = } / {batch_size}")
                        outputs = actual_prediction(
                            batch_arg=without_logging_info,
                            collator=collator,
                            model=self._model,
                            generation_kwargs=self._generation_kwargs,
                            gen_function=GEN_FUNCTION,
                            max_answer_gen=self._max_answer_gen,
                            max_total_gen_length=self._max_total_length_gen,
                        )

                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Prep the outputs
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        for idx, output, input_data in zip(
                            prediction_batch_iterator_idx, outputs, without_logging_info
                        ):
                            send_values[idx] = prep_return_data(
                                output,
                                decoder_input_ids=input_data[
                                    "decoder_input_ids_for_gen"
                                ],
                                tokenizer=self._tokenizer,
                            )

                        prediction_batch = []
                        prediction_batch_iterator_idx = []

                num_nones = sum(x is None for x in final_values)
                assert num_nones == 0, f"{num_nones}/{len(final_values)} were None"

                if ds is self._eval_ds:
                    pred_logger.log()
                yield collator(final_values)

        if CONC_MODE == "top_sort":
            assert self._train_ds._mask_intermediate_labels

            for i in tqdm(range(0, len(ds), batch_size), desc=f"progress in {dl_name}"):
                batch_indices = indices[i : i + batch_size]
                root_nodes = [ds.get_top_sort(i) for i in batch_indices]

                sorters = []  # 1 to 1 with root nodes, ok
                node_to_sorter = {}

                for root_node in root_nodes:
                    tree = {}
                    top_sort_build_tree(tree, root_node)
                    sorter = graphlib.TopologicalSorter(tree)
                    sorter.prepare()
                    sorters.append(sorter)
                    node_to_sorter[root_node] = sorter

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
                            node_batch.extend(nodes)

                    # Heuristic: do the deepest nodes first, they are likely the ones blocking
                    # the most nodes.
                    node_batch.sort(
                        key=lambda x: x.get_root_complexity_level() - x.get_complexity_level(), 
                        reverse=True
                    )
                    node_batch = node_batch[:batch_size]

                    input_ids = [
                        self._tokenizer(node.get_input_str()) for node in node_batch
                    ]
                    decoder_input_ids = [
                        self._tokenizer(
                            node.get_pseudo_topsort_query(),
                            return_tensors=None,
                            no_eos=False,
                            strip_special_symbols=True,
                        )
                        for node in node_batch
                    ]
                    labels = [
                        len(decoder_ii) * [-100] + self._tokenizer(node.get_value(), None, no_eos=False)
                        for decoder_ii, node in zip(decoder_input_ids, node_batch)
                    ]
                    batch = dict(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels,
                    )
                    outputs = actual_prediction(
                        batch_arg=batch,
                        collator=collator,
                        model=self._model,
                        generation_kwargs=self._generation_kwargs,
                        gen_function=GEN_FUNCTION,
                        max_answer_gen=self._max_answer_gen,
                        max_total_gen_length=self._max_total_length_gen,
                    )

                    for node, output in zip(sorters, node_batch, outputs):
                        text = self._tokenizer.decode(
                            output.tolist()
                        )
                        node.set_pseudo_value(text.split("=")[-1])
                        node_to_sorter[node].done(node)

                    for sorter in sorters:
                        if sorter.is_active():
                            at_least_one = True
                            break

                pre_collate_batch = []
                for node in nodes:
                    encoder_ii = self._tokenizer(
                        node.get_input_str(), return_tensors="pt", no_eos=False
                    )
                    # Like this the special sybol stripping is OK
                    decoder_body = self._tokenizer(
                        node.get_pseudo_topsort_query(),
                        return_tensors=None,
                        no_eos=True,
                        strip_special_symbols=True,
                    )
                    decoder_ii = decoder_body
                    label = (
                        len(decoder_body) * [-100]
                        + self._tokenizer(
                            node.get_value(), return_tensors=None, no_eos=True
                        )
                        + self._tokenizer.encode(")", return_tensors=None, no_eos=False)
                    )
                    pre_collate_batch.append(
                        dict(
                            input_ids=encoder_ii,
                            decoder_input_ids=decoder_ii,
                            labels=label,
                        )
                    )

                batch = collator(pre_collate_batch)
                yield batch

    @beartype
    def _make_regular_dataloader(
        self,
        ds: Union[
            our_datasets.MostBasicDataset,
            our_datasets.OracleBasicDataset,
            our_datasets.SelfLearnedBasicDataset,
        ],
        batch_size: int,
        shuffle: bool,
        return_ids: bool, 
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=our_data_collator.DataCollatorWithDecoderInputIds(
                tokenizer=self._tokenizer,
                model=self._model,
                max_length=self._model.config.max_position_embeddings,
                mask_intermediate_labels=self._mask_intermediate_labels,
                return_idents=return_ids,
            ),
            batch_size=batch_size,
            num_workers=self._num_workers_dl,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        if isinstance(self._train_ds, our_datasets.SelfLearnedBasicDataset):
            if self._train_ds.has_len():
                num_batches = math.ceil(len(self._train_ds) / self._batch_size)
            else:
                num_batches = None

            return RenewableGenerator(
                lambda: self._make_dataloader(
                    ds=self._train_ds,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle_train,
                    dl_name="train_dl",
                ),
                len=num_batches,
            )
        else:
            return self._make_regular_dataloader(
                ds=self._train_ds,
                batch_size=self._batch_size,
                shuffle=self._shuffle_train,
                return_ids=False,
            )

    def val_dataloader(self):
        if isinstance(self._eval_ds, our_datasets.SelfLearnedBasicDataset):
            if self._eval_ds.has_len():
                num_batches = math.ceil(len(self._eval_ds) / self._batch_size)
            else:
                num_batches = None

            return RenewableGenerator(
                lambda: self._make_dataloader(
                    ds=self._eval_ds,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle_val,
                    dl_name="val_dl",
                ),
                len=num_batches,
            )
        else:
            return self._make_regular_dataloader(
                ds=self._eval_ds,
                batch_size=self._batch_size,
                shuffle=self._shuffle_val,
                return_ids=True,
            )

def get_last_checkpoint_path(checkpoints_folder, wandb_run_id):
    rich.print(f"[red bold]{wandb_run_id = }")
    checkpoint_files = list(checkpoints_folder.glob("**/*.ckpt"))
    assert len(checkpoint_files) == 1, checkpoint_files
    checkpoints = list((checkpoints_folder / WANDB_PROJECT / wandb_run_id / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 1, checkpoints
    checkpoint_path = checkpoints[0]
    rich.print(f"[red bold]{checkpoint_path = }")
    return checkpoint_path
    

DEFAULT_SAVE_DIR = SCRIPT_DIR / "log_results/test_bench/"


def main(
    ###########################################################################
    # Should not change
    ###########################################################################
    data_name="349_6_6_200000.json.pkl",    
    inf_num=6,
    abs_pos_embs_mode=bart_modified.AbsPosEmbsModes.learned_pos_embs,
    rel_pos_embs_mode=bart_relative_attention.RelPosEmbsChoices.no_rel_pos_embs,
    num_rel_pos_embs=64,
    max_epochs=100,

    ###########################################################################
    # Things to handle resuming
    ###########################################################################
    checkpoints_folder=DEFAULT_SAVE_DIR / "checkpoints/",
    extra_info_file=DEFAULT_SAVE_DIR / "extra_info.json",

    ###########################################################################
    # Changes often
    ###########################################################################
    seed=453345,
    freeform_options=[True, False],
    dataset_type=our_datasets.DatasetTypesChoices.oracle_basic_dataset,
    max_level_training=6,
    do_log_results=True,
    path_log_results=DEFAULT_SAVE_DIR / "results.json",

    ###########################################################################
    # Changes with model size
    ###########################################################################
    batch_size=256,

    ###########################################################################
    # Model Stuff
    ###########################################################################
    # Small config
    h_size=64,
    n_layers=2,
    n_heads=4,
):
    all_arguments = locals().copy()
    # Inspect is only used for a test
    assert all_arguments.keys() == inspect.signature(main).parameters.keys()

    extra_info_file: Final[Path] = Path(extra_info_file)
    checkpoints_folder: Final[Path] = Path(checkpoints_folder)

    if extra_info_file.exists():
        with extra_info_file.open() as f:
            wandb_run_id = json.load(f)["wandb_run_id"]
        path_last_checkpoint = get_last_checkpoint_path(checkpoints_folder, wandb_run_id)
        assert path_last_checkpoint.exists() == extra_info_file.exists(), (
            f"\n\t- {path_last_checkpoint.exists()}: {path_last_checkpoint}"
            f"\n\t- {extra_info_file.exists()}: {extra_info_file}"
        )
    resuming = extra_info_file.exists()


    ###########################################################################
    # Set the seeds
    ###########################################################################
    torch.use_deterministic_algorithms(mode=DETERMINISTIC)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ###############################################################
    # Handle resuming the Wandb run & setting the seeds
    ###############################################################
    if dataset_type == our_datasets.DatasetTypesChoices.most_basic_dataset:
        ds_str = "most_basic"
    elif dataset_type == our_datasets.DatasetTypesChoices.oracle_basic_dataset:
        ds_str = "oracle"
    else:
        raise ValueError("Unknown dataset type")

    fix_str = f"abs_pe_{abs_pos_embs_mode}"
    rel_str = f"rel_pe_{rel_pos_embs_mode}_{num_rel_pos_embs}_" if rel_pos_embs_mode else ""
    run_name = f"{rel_str}{fix_str}_trained_{max_level_training}_inf_{inf_num}_{ds_str}" # "_h_size_{h_size}_n_layers_{n_layers}_n_heads_{n_heads}"
    
    if resuming:
        with extra_info_file.open() as f:
            extra_info = json.load(f)
            wandb_run_id = extra_info["wandb_run_id"]
            torch.random.set_rng_state(torch.ByteTensor(extra_info["torch_rng_state"]))
            np.random.set_state(extra_info["numpy_rng_state"])

            for i, v in enumerate(extra_info["python_rng_state"]):
                if isinstance(v, list):
                    extra_info["python_rng_state"][i] = tuple(extra_info["python_rng_state"][i])

            random.setstate(tuple(extra_info["python_rng_state"]))

        rich.print("[red bold]Resuming Wandb run:", wandb_run_id)
        wandb.init(project=WANDB_PROJECT, resume="must", id=wandb_run_id)

        assert wandb.run.resumed, wandb.run.resumed
        assert wandb.run.project == WANDB_PROJECT, (wandb.run.project, WANDB_PROJECT)
        assert wandb.run.id == wandb_run_id, (wandb.run.id, wandb_run_id)

    else:
        extra_info = None       
        wandb_run_id = None


    rich.print(f"Run name: [green]'{run_name}'\n")
    utils.print_dict(all_arguments)

    if do_log_results:
        path_log_results = Path(path_log_results)
        if path_log_results.exists():
            assert resuming, f"The log file already exists, but we are not resuming: {path_log_results}"
        
    assert dataset_type in our_datasets.DATASET_TYPES, dataset_type
    assert freeform_options, freeform_options
    assert isinstance(freeform_options, (list, tuple, set)), freeform_options
    assert all(isinstance(x, bool) for x in freeform_options)

    dataset, dataset_config = data_generation_arithmetic.load_dataset(None, DATA_PATH / data_name)
    rich.print(vars(dataset_config))
    
    train_ds : Dict[int, data_generation_arithmetic.Node] = dataset["train"]
    valid_ds : Dict[int, data_generation_arithmetic.Node] = dataset["eval"]
    assert len(train_ds) == inf_num, (len(train_ds), inf_num)

    if max_level_training:
        assert max_level_training <= dataset_config.max_depth
        assert all(isinstance(node_name, int) for node_name in train_ds)
        train_ds = {
            level_no: level_nodes for level_no, level_nodes 
            in train_ds.items() if level_no <= max_level_training
        }

    tokenizer = our_tokenizer.ArithmeticTokenizer()
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
    config.num_hidden_layers = n_layers  # Default is 6
    config.hidden_size = h_size  # Default is 768
    config.encoder_attention_heads = n_heads  # Default is 16
    config.decoder_attention_heads = n_heads  # Default is 16
    config.encoder_ffn_dim = config.hidden_size * 4  # Default is 4096, 4 x hidden_size
    config.decoder_ffn_dim = config.hidden_size * 4  # Default is 4096, 4 x hidden_size

    # The only difference is that we take decoder_attention_masks into account
    # for positional embeddings
    model = bart_modified.ModifiedBartForConditionalGeneration(
        config, 
        abs_pos_embs_mode=abs_pos_embs_mode,
        rel_pos_embs_mode=rel_pos_embs_mode,
        num_rel_pos_embs=num_rel_pos_embs,
    )

    assert type(train_torch_dataset) == type(valid_torch_dataset), (
        type(train_torch_dataset),
        type(valid_torch_dataset),
    )
    assert CONC_MODE in ["yield", "top_sort"]

    if isinstance(
        train_torch_dataset, our_datasets.SelfLearnedBasicDataset
    ) and isinstance(valid_torch_dataset, our_datasets.SelfLearnedBasicDataset):

        train_torch_dataset.set_conc_mode(CONC_MODE)
        valid_torch_dataset.set_conc_mode(CONC_MODE)
        train_torch_dataset.set_mask_intermediate_labels(True)
        valid_torch_dataset.set_mask_intermediate_labels(True)

        assert CONC_MODE in ["yield", "top_sort"], CONC_MODE

    eval_batch_size = batch_size * 4
    pl_object = _PLBart(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_torch_dataset,
        eval_ds=valid_torch_dataset,
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers_dl=0,
        generation_kwargs=GENERATION_KWARGS,
        learning_rate=LEARNING_RATE,
        is_adamw=True,
        weight_decay=WEIGHT_DECAY,
        # scheduler_type="WarmupLinear",
        # scheduler_kwargs=dict(),
        # do_allen_nlp_predictions=False,
        max_total_length_gen=dataset_config.max_total_length,
        max_answer_gen=dataset_config.max_answer_length,
        max_depth=dataset_config.max_depth,
        do_log_results=do_log_results,
        path_log_results=path_log_results,
        extra_info_file=extra_info_file,
        freeform_options=freeform_options,
    )
    logger = pl.loggers.WandbLogger(
        project=WANDB_PROJECT,
        name=run_name,
        entity=WANDB_ENTITY,
        log_model=False,
        config=dict(
            bart_config=vars(config),
            dataset_type=type(train_torch_dataset).__name__,
            eval_batch_size=eval_batch_size,
            generation_kwargs=GENERATION_KWARGS,
            learning_rate=LEARNING_RATE,
            num_gpus=NUM_GPUS,
            precision=PRECISION,
            train_batch_size=batch_size,
            max_level_training=max_level_training,
            arguments=all_arguments,
        ),
    )
    wandb.run.log_code(SCRIPT_DIR)

    trainer = pl.Trainer(
        enable_checkpointing=pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_folder,
            every_n_epochs=1, 
            save_on_train_epoch_end=True, 
            save_last=True
        ),
        deterministic=DETERMINISTIC,
        default_root_dir=checkpoints_folder,
        logger=logger,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision=PRECISION,
        max_epochs=max_epochs,
        gpus=NUM_GPUS,
        check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
        limit_val_batches=NUM_SAMPLES_VALID // eval_batch_size,
    )
    
    if resuming:
        trainer.fit(pl_object, ckpt_path=path_last_checkpoint)
    else:
        trainer.fit(pl_object)


if __name__ == "__main__":
    fire.Fire(main)
