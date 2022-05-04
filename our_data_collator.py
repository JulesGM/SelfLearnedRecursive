from cProfile import label
import dataclasses
from typing import *

import numpy as np
import torch
import transformers
import transformers.models.bart.modeling_bart as modeling_bart

def _pad_batch_of_sequences_from_dict(features, key, pad_token_id):
    """
    Pad a batch of sequences. 
    This is used because huggingface's PretrainedTokenizer.pad only pads input_ids.
    """
    max_length = max(len(entry[key]) for entry in features)
    for i, entry in enumerate(features):
        key_entry = entry[key]
        remainder = [pad_token_id] * (max_length - len(key_entry))
        if isinstance(entry, list):
            features[i][key] = key_entry + remainder
        else:
            features[i][key] = np.concatenate([key_entry, remainder]).astype(np.int64)


def _shift_tokens_right_no_delete_last(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Move input ids one token to the right, but we don't remove the last token.
    This is needed because huggingface's modeling_bart.shift_tokens_right removes the last token, which
    makes sense for teacher forcing. However, we use this for decoding.
    """
    new_shape = list(input_ids.shape)
    new_shape[1] += 1
    shifted_input_ids = input_ids.new_zeros(new_shape)
    shifted_input_ids[:, 1:] = input_ids.clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclasses.dataclass
class DataCollatorWithDecoderInputIds:
    tokenizer: transformers.PreTrainedTokenizerBase
    model: Optional[Any] = None
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

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

        keys = features[0].keys()
        for feature in features[1:]:
            assert feature.keys() == keys

        assert "input_ids" in keys
        assert "labels" in keys

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
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            labels=features["labels"]
        )
        features["decoder_input_ids"] = decoder_input_ids
        
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
            
        return features