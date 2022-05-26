from typing import *

import numpy as np
import torch

class Tokenizer:
    __slots__ = (
        "vocab",
        "token_to_idx",
        "idx_to_token",
        "bos_token",
        "eos_token",
        "pad_token",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "padding_side",
        "max_length",
    )

    def __init__(self, max_length: int, use_equal_symbol: bool):
        self.vocab = [
            "<pad>",  # 0
            "<bos>",  # 1
            "<eos>",  # 2
            "0",      # 3
            "1",      # 4
            "2",      # 5
            "3",      # 6
            "4",      # 7
            "5",      # 8
            "6",      # 9
            "7",      # 10
            "8",      # 11 
            "9",      # 12
            "+",      # 13
            "-",      # 14
            "*",      # 15
            "(",      # 16
            ")",      # 17
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

    def encode(self, input_str: str, return_tensors: str = "np", no_eos: bool = False) -> Sequence:
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
        if no_eos:
            list_form = output
        else:
            list_form = output + [self.token_to_idx["<eos>"]]
            
        if return_tensors is None:
            output = list_form
        elif return_tensors == "np":
            output = np.array(list_form, dtype=np.int64)
        elif return_tensors == "pt":
            output = torch.tensor(list_form, dtype=torch.int64)
        else:
            raise ValueError(f"Unknown return_tensors value '{return_tensors}'")
            
        return output
    
    def decode(self, input_tokens: List[int], ignore_special_symbols: bool) -> str:
        assert type(input_tokens) == list, type(input_tokens)
        output = []
        ignore_set = set([self.bos_token_id, self.eos_token_id, self.pad_token_id])
        for token_index in input_tokens:
            if ignore_special_symbols and token_index in ignore_set:
                continue
            if token_index == -100:
                output.append("<-100>")
            elif isinstance(token_index, int) and token_index >= 0 and token_index < len(self.idx_to_token):
                output.append(self.idx_to_token[token_index])
            else:
                raise ValueError(f"Unknown token index '{token_index}'")
        
        return " ".join(output)

    def pad(self, features, padding, max_length: int, pad_to_multiple_of: bool, return_tensors: str = "np") -> Sequence:
        """ Pad input_token_ids, create attention_mask, convert everything to tensors.
        Mirrors huggingface tokenizers that way.
        """
        assert not pad_to_multiple_of, "Not implemented"
        assert padding is True or padding == "longuest", "Other values are not implemented"
        

        max_read = max(len(x["input_ids"]) for x in features)

        if max_length is not None:
            max_read = min(max_read, max_length)
        
        padded_sequences = []
        for seq in features:
            if len(seq["input_ids"]) < max_read:
                seq["input_ids"] = (
                    seq["input_ids"].tolist() + 
                    [self.token_to_idx["<pad>"]] * (max_read - len(seq["input_ids"]))
                )
                seq["input_ids"] = seq["input_ids"][:max_read]

            seq["attention_mask"] = [int(x != self.token_to_idx["<pad>"]) for x in seq["input_ids"]]
            padded_sequences.append(seq)
        
        keys = padded_sequences[0].keys()
        for padded_seq in padded_sequences[1:]:
            as_set = set(padded_seq.keys())
            assert as_set == keys, as_set
        
        if return_tensors == "np":
            output = {}
            for k in keys:
                seq = [x[k] for x in padded_sequences]
                output[k] = np.array(seq, dtype=np.int64)
            return output

        elif return_tensors == "pt":
            output = {}
            for k in keys:
                seq = [x[k] for x in padded_sequences]
                output[k] = torch.tensor(seq, dtype=torch.int64)
            return output

        elif return_tensors is None:
            return {
                k: [x[k] for x in padded_sequences] 
                for k in padded_sequences[0].keys()
            }

        else:
            raise ValueError(f"Unknown return_tensors value '{return_tensors}'")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)