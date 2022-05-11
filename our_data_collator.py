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
from cProfile import label
import dataclasses
from typing import *

import numpy as np
import torch
import transformers
import transformers.models.bart.modeling_bart as modeling_bart


def _pad_batch_of_sequences_from_dict(features, key, pad_token_id) -> None:
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
            features[i][key] = key_entry + remainder
        else:
            features[i][key] = np.concatenate([key_entry, remainder]).astype(np.int64)


class DataCollatorWithDecoderInputIds:
    __slots__ = (
        "tokenizer", 
        "model", 
        "max_length", 
        "label_pad_token_id"
    )

    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        model: Optional[Any] = None, 
        max_length=512,
        label_pad_token_id: int = -100
    ):    
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

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
        assert self.tokenizer.padding_side == "right", f"Only right-padded inputs are supported. Got {self.tokenizer.padding_side}."
        assert self.model is not None, "You must provide a model to the data collator"
        assert self.model.config.pad_token_id == self.tokenizer.pad_token_id, (
            "The pad_token_id of the model must be the same as the one used by the tokenizer"
        )
        assert self.model.config.decoder_start_token_id == self.tokenizer.bos_token_id
        assert "decoder_attention_mask_for_gen" not in features, features.keys()

        keys = features[0].keys()
        for feature in features[1:]:
            assert feature.keys() == keys

        assert "input_ids" in keys
        assert not "decoder_input_ids" in keys

        #######################################################################
        # 1. Pad labels
        #######################################################################
        if "labels" in keys:
            _pad_batch_of_sequences_from_dict(
                features,
                "labels", 
                self.label_pad_token_id,
            )

        #######################################################################
        # 2. Pad gen decoder inputs if they are there       
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            _pad_batch_of_sequences_from_dict(
                features, 
                "decoder_input_ids_for_gen",
                self.model.config.pad_token_id,
                )

        #######################################################################
        # 3. Pad inputs & convert everything to tensors
        #######################################################################
        pre_tok_features = features
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        #######################################################################
        # 4. Build training decoder input ids by shifting labels
        #######################################################################
        if "labels" in features:
            features["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
        
        #######################################################################
        # 5. If we have gen decoder inputs, we shift them as well.
        # We remove the eos token if it's there, as we generate after it.
        #######################################################################
        if "decoder_input_ids_for_gen" in keys:
            features["decoder_input_ids_for_gen"] = modeling_bart.shift_tokens_right(
                features["decoder_input_ids_for_gen"],
                self.model.config.pad_token_id,
                self.model.config.bos_token_id,
            ) 

            # Remove the mask
            features["decoder_input_ids_for_gen"].masked_fill_(
                features["decoder_input_ids_for_gen"] == self.tokenizer.eos_token_id, 
                self.tokenizer.pad_token_id,
            )

            assert torch.all(
                features["decoder_input_ids_for_gen"][:, 0] == 
                self.model.config.decoder_start_token_id
            ), features["decoder_input_ids_for_gen"][:, 0]

            features["decoder_attention_mask_for_gen"] = (
                features["decoder_input_ids_for_gen"] != self.tokenizer.pad_token_id
            )
            
        return features