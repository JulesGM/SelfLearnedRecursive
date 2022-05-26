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
import dataclasses
from typing import *

import numpy as np
import torch
import transformers
import transformers.models.bart.modeling_bart as modeling_bart


def _pad_batch_of_sequences_from_dict(features, key, pad_token_id, pad_direction) -> None:
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
                features[i][key] = np.concatenate([key_entry, remainder]).astype(np.int64)
            elif pad_direction == "left":
                features[i][key] = np.concatenate([remainder, key_entry]).astype(np.int64)

class DataCollatorWithDecoderInputIds:
    __slots__ = (
        "_tokenizer", 
        "_model", 
        "_max_length", 
        "_mask_intermediate_labels",
        "_label_pad_token_id",
    )

    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        model: transformers.PreTrainedModel, 
        max_length: int,
        mask_intermediate_labels: bool,
        label_pad_token_id: int = -100,
    ):    
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = max_length
        self._mask_intermediate_labels = mask_intermediate_labels
        self._label_pad_token_id = label_pad_token_id

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
        assert self._tokenizer.padding_side == "left", f"Only left-padded inputs are supported. Got {self.tokenizer.padding_side}."
        assert self._model is not None, "You must provide a model to the data collator"
        assert self._model.config.pad_token_id == self._tokenizer.pad_token_id, (
            "The pad_token_id of the model must be the same as the one used by the tokenizer"
        )
        assert self._model.config.decoder_start_token_id == self._tokenizer.bos_token_id
        assert "decoder_attention_mask_for_gen" not in features, features.keys()

        keys = features[0].keys()
        for feature in features[1:]:
            assert feature.keys() == keys

        assert "input_ids" in keys
        assert not "decoder_input_ids" in keys
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

        #######################################################################
        # 3. Pad inputs & convert everything to tensors
        #######################################################################
        features = self._tokenizer.pad(
            features,
            padding=True,
            max_length=self._max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        #######################################################################
        # 4. Build training decoder input ids by shifting labels
        #######################################################################
        if "labels" in features and not self._mask_intermediate_labels:
            # If we're not masking the intermediate labels, we can use the "labels"
            # field to build the `decoder_input_ids`.
            features["decoder_input_ids"] = self._model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )

        if self._mask_intermediate_labels:
            # If we're masking the intermediate labels, we can't use the "labels"
            # to build the `decoder_input_ids`, because the intermediate results are
            # masked in the label.
            assert "decoder_input_ids_for_gen" in features or "decoder_input_ids" in features
            
        
        #######################################################################
        # 5. If we have gen decoder inputs, we shift them as well.
        # We remove the eos token if it's there, as we generate after it.
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            features["decoder_input_ids_for_gen"] = modeling_bart.shift_tokens_right(
                features["decoder_input_ids_for_gen"],
                self._model.config.pad_token_id,
                self._model.config.bos_token_id,
            ) 

            assert torch.all(
                features["decoder_input_ids_for_gen"][:, 0] == 
                self._model.config.decoder_start_token_id
            ), features["decoder_input_ids_for_gen"][:, 0]

        return features

class DataCollatorDecoderOnly:
    __slots__ = (
        "_tokenizer", 
        "_model", 
        "_max_length", 
        "_mask_intermediate_labels",
        "_label_pad_token_id",
    )

    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        model: transformers.PreTrainedModel, 
        max_length: int,
        mask_intermediate_labels: bool,
        label_pad_token_id: int = -100,
    ):    
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = max_length
        self._mask_intermediate_labels = mask_intermediate_labels
        self._label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        assert self._tokenizer.padding_side == "right", f"Only right-padded inputs are supported. Got {self.tokenizer.padding_side}."
        assert self._model is not None, "You must provide a model to the data collator"
        assert self._model.config.pad_token_id == self._tokenizer.pad_token_id, (
            "The pad_token_id of the model must be the same as the one used by the tokenizer"
        )
        assert self._model.config.decoder_start_token_id == self._tokenizer.bos_token_id
        assert "decoder_attention_mask_for_gen" not in features, features.keys()

        keys = features[0].keys()
        for feature in features[1:]:
            assert feature.keys() == keys

        assert "input_ids" in keys
        assert not "decoder_input_ids" in keys
        DECODER_PAD_DIRECTION = "left"

        #######################################################################
        # 1. Concatenate inputs and labels and pad them
        #######################################################################
        if "labels" in keys:
            _pad_batch_of_sequences_from_dict(
                features,
                "labels", 
                self._label_pad_token_id,
                pad_direction=DECODER_PAD_DIRECTION,
            )

        #######################################################################
        # 2. Concatenate inputs and decoder_inputs_for_gen and pad them
        #    Pad gen decoder inputs if they are there       
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            _pad_batch_of_sequences_from_dict(
                features, 
                "decoder_input_ids_for_gen",
                self._model.config.pad_token_id,
                pad_direction=DECODER_PAD_DIRECTION,
                )

        #######################################################################
        # 3. Pad inputs & convert everything to tensors
        #######################################################################
        features = self._tokenizer.pad(
            features,
            padding=True,
            max_length=self._max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        #######################################################################
        # 4. Build training decoder input ids by shifting labels
        #######################################################################
        if "labels" in features and not self._mask_intermediate_labels:
            # If we're not masking the intermediate labels, we can use the "labels"
            # field to build the `decoder_input_ids`.
            features["decoder_input_ids"] = self._model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )

        if self._mask_intermediate_labels:
            # If we're masking the intermediate labels, we can't use the "labels"
            # to build the `decoder_input_ids`, because the intermediate results are
            # masked in the label.
            assert "decoder_input_ids_for_gen" in features or "decoder_input_ids" in features
            
        
        #######################################################################
        # 5. If we have gen decoder inputs, we shift them as well.
        # We remove the eos token if it's there, as we generate after it.
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            features["decoder_input_ids_for_gen"] = modeling_bart.shift_tokens_right(
                features["decoder_input_ids_for_gen"],
                self._model.config.pad_token_id,
                self._model.config.bos_token_id,
            ) 

            assert torch.all(
                features["decoder_input_ids_for_gen"][:, 0] == 
                self._model.config.decoder_start_token_id
            ), features["decoder_input_ids_for_gen"][:, 0]

        return features