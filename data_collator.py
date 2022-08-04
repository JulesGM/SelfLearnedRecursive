"""
Table of Contents:

    > `_pad_batch_of_sequences_from_dict`:
        - Pad batches in a dict.

    > `DataCollatorWithDecoderInputIds`:
        - Pad everything in a batch
        - Convert everything to tensors
        - Create decoder input ids from labels if we have them.
        - Shift decoder input ids
"""
import collections
from typing import *

import numpy as np
import rich
import torch
import transformers
import transformers.models.bart.modeling_bart as modeling_bart

import data_generation_arithmetic


def _pad_batch_of_sequences_from_dict(
    features: list[dict[str, Union[torch.Tensor, np.ndarray, list]]], 
    key: str, 
    pad_token_id: int, 
    pad_direction: str,
) -> None:
    """
    Pad a batch of sequences, with a certain key value.

    This is used because huggingface's PretrainedTokenizer.pad only pads input_ids.

    Changes "features" in place in a dict.
    """
    max_length = max(len(entry[key]) for entry in features)
    for i, entry in enumerate(features):
        key_entry = entry[key]
        remainder = [pad_token_id] * (max_length - len(key_entry))

        if isinstance(entry, list):
            if pad_direction == "right":
                features[i][key] = key_entry + remainder
            elif pad_direction == "left":
                features[i][key] = remainder + key_entry
            else:
                raise ValueError("pad_direction must be 'right' or 'left'")
        else:
            if pad_direction == "right":
                features[i][key] = np.concatenate([key_entry, remainder]).astype(
                    np.int64
                )
            elif pad_direction == "left":
                features[i][key] = np.concatenate([remainder, key_entry]).astype(
                    np.int64
                )


class DataCollatorWithDecoderInputIds:
    __slots__ = (
        "_tokenizer",
        "_model",
        "_max_length",
        "_mask_intermediate_labels",
        "_label_pad_token_id",
        "_return_idents",
    )

    def __init__(
        self,
        *,
        tokenizer,
        model: transformers.PreTrainedModel,
        max_length: int,
        mask_intermediate_labels: bool,
        return_idents: bool,
        label_pad_token_id: int = -100,
    ):
        self._tokenizer: Final = tokenizer
        self._model: Final[transformers.PretrainedModel] = model  # type: ignore[name-defined]
        self._max_length: Final[int] = max_length
        self._mask_intermediate_labels: Final[bool] = mask_intermediate_labels
        self._label_pad_token_id: Final[int] = label_pad_token_id
        self._return_idents: Final[bool] = return_idents

    def __call__(self, features):
        """
        #######################################################################
        Situation:
        - We need to pad everything.
        - We need regular decoder input ids for training and validation.
        - When we have a scratch pad, we need special decoder input ids to only
        predict the final value.

        #######################################################################
        Table of contents:
        1. Pad labels
        2. Pad gen decoder inputs if they are there
        3. Pad inputs & convert everything to tensors
        4. Build training decoder input ids by shifting labels
        5. If we have gen decoder inputs, we shift them as well.
        #######################################################################

        """
        
        assert (
            self._tokenizer.padding_side == "left"
        ), f"Only left-padded inputs are supported. Got {self.tokenizer.padding_side}."
        assert self._model is not None, "You must provide a model to the data collator"
        assert (
            self._model.config.pad_token_id == self._tokenizer.pad_token_id
        ), "The pad_token_id of the model must be the same as the one used by the tokenizer"
        assert "decoder_attention_mask_for_gen" not in features, features.keys()
        if "decoder_input_ids" in features:
            assert self._mask_intermediate_labels

        keys = set(features[0].keys())
        for feature in features[1:]:
            assert feature.keys() == keys
        
        if self._return_idents:
            assert "idents" in keys, f"We needs ids to be able to return them. {keys = }"
            idents = [x.pop("idents") for x in features]

        assert "input_ids" in keys
        DECODER_PAD_DIRECTION = "left"

        #######################################################################
        # 1. Pad labels
        #######################################################################
        if "labels" in keys:
            _pad_batch_of_sequences_from_dict(
                features,
                "labels",
                self._label_pad_token_id,
                pad_direction=DECODER_PAD_DIRECTION,
            )

        #######################################################################
        # 2. Pad gen decoder inputs if they are there
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            _pad_batch_of_sequences_from_dict(
                features,
                "decoder_input_ids_for_gen",
                self._model.config.pad_token_id,
                pad_direction=DECODER_PAD_DIRECTION,
            )

        if "decoder_input_ids" in keys:
            _pad_batch_of_sequences_from_dict(
                features,
                "decoder_input_ids",
                self._model.config.pad_token_id,
                pad_direction=DECODER_PAD_DIRECTION,
            )

        #######################################################################
        # 3. Pad inputs & convert everything to tensors
        #######################################################################
        out_features = self._tokenizer.pad(
            features,
            padding=True,
            max_length=self._max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
            ignore_keys=["idents"]
        )
        del features
        if self._return_idents:
            out_features["idents"] = idents
        
        #######################################################################
        # 4. Build training decoder input ids by shifting labels
        #######################################################################
        if "decoder_input_ids" in keys:
            assert self._mask_intermediate_labels

            out_features["decoder_input_ids"] = modeling_bart.shift_tokens_right(
                out_features["decoder_input_ids"],
                self._model.config.pad_token_id,
                self._model.config.decoder_start_token_id,
            )

            assert torch.all(
                out_features["decoder_input_ids"][:, 0]
                == self._model.config.decoder_start_token_id
            ), out_features["decoder_input_ids"][:, 0]

        if "labels" in out_features and not self._mask_intermediate_labels:
            # If we're not masking the intermediate labels, we can use the "labels"
            # field to build the `decoder_input_ids`.
            out_features[
                "decoder_input_ids"
            ] = self._model.prepare_decoder_input_ids_from_labels(
                labels=out_features["labels"]
            )

        if self._mask_intermediate_labels:
            # If we're masking the intermediate labels, we can't use the "labels"
            # to build the `decoder_input_ids`, because the intermediate results are
            # masked in the label.
            assert (
                "decoder_input_ids_for_gen" in out_features
                or "decoder_input_ids" in out_features
            )

        #######################################################################
        # 5. If we have gen decoder inputs, we shift them as well.
        # We remove the eos token if it's there, as we generate after it.
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            out_features["decoder_input_ids_for_gen"] = modeling_bart.shift_tokens_right(
                out_features["decoder_input_ids_for_gen"],
                self._model.config.pad_token_id,
                self._model.config.decoder_start_token_id,
            )

            assert torch.all(
                out_features["decoder_input_ids_for_gen"][:, 0]
                == self._model.config.decoder_start_token_id
            ), out_features["decoder_input_ids_for_gen"][:, 0]

        return out_features
