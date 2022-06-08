from typing import *

import numpy as np
import torch


class ArithmeticTokenizer:
    __slots__ = (
        "vocab",
        "token_to_idx",
        "idx_to_token",
        "bos_token",
        "bos_token_id",
        "decoder_start_token",
        "decoder_start_token_id",
        "eos_token",
        "eos_token_id",
        "pad_token",
        "pad_token_id",
        "padding_side",
        "special_tokens",
        "special_token_ids",
    )

    def __init__(self):
        self.vocab = [
            "<pad>",  # 0
            "<eos>",  # 1
            "<bos>",  # 3
            "<start>",  # 5
            "0",  # 6
            "1",  # 7
            "2",  # 8
            "3",  # 9
            "4",  # 10
            "5",  # 11
            "6",  # 12
            "7",  # 13
            "8",  # 14
            "9",  # 15
            "+",  # 16
            "-",  # 17
            "*",  # 18
            "(",  # 19
            ")",  # 20
            "=",  # 21
        ]

        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = self.vocab
        self.bos_token = "<bos>"
        self.bos_token_id = self.token_to_idx["<bos>"]
        self.decoder_start_token = "<start>"
        self.decoder_start_token_id = self.token_to_idx["<start>"]
        self.eos_token = "<eos>"
        self.eos_token_id = self.token_to_idx["<eos>"]
        self.pad_token = "<pad>"
        self.pad_token_id = self.token_to_idx["<pad>"]
        self.padding_side = "left"

        self.special_tokens = {"<bos>", "<eos>", "<pad>", "<start>"}
        self.special_token_ids = {
            self.token_to_idx[token] for token in self.special_tokens
        }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Checks:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for token in self.vocab:
            if token in self.special_tokens:
                assert token[0] == "<" and token[-1] == ">", token
            else:
                assert "<" not in token and ">" not in token, token
                assert len(token) == 1, token

    def strip_special_tokens(self, input_text):
        for token in self.special_tokens:
            input_text = input_text.replace(token, "")
        return input_text

    def encode(
        self,
        input_str: str,
        return_tensors: str = "np",
        *,
        no_eos: bool = False,
        strip_special_symbols: bool = False,
    ) -> Sequence:
        assert type(input_str) == str, type(input_str)

        if strip_special_symbols:
            input_str = self.strip_special_tokens(input_str)

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

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.detach().cpu().numpy().tolist()
        elif isinstance(input_tokens, np.ndarray):
            input_tokens = input_tokens.tolist()

        output = []
        ignore_set = set([self.bos_token_id, self.eos_token_id, self.pad_token_id])
        for token_index in input_tokens:
            if ignore_special_symbols and token_index in ignore_set:
                continue
            if token_index == -100:
                output.append("<-100>")
            elif (
                isinstance(token_index, int)
                and token_index >= 0
                and token_index < len(self.idx_to_token)
            ):
                output.append(self.idx_to_token[token_index])
            else:
                raise ValueError(f"Unknown token index '{token_index}'")

        return " ".join(output)

    def pad(
        self,
        features,
        padding,
        max_length: int,
        pad_to_multiple_of: bool,
        return_tensors: str = "np",
    ) -> Sequence:
        """Pad input_token_ids, create attention_mask, convert everything to tensors.
        Mirrors huggingface tokenizers that way.
        """
        assert not pad_to_multiple_of, "Not implemented"
        assert (
            padding is True or padding == "longuest"
        ), "Other values are not implemented"

        max_read = max(len(x["input_ids"]) for x in features)

        if max_length is not None:
            max_read = min(max_read, max_length)

        padded_sequences = []
        for seq in features:
            if len(seq["input_ids"]) < max_read:
                seq["input_ids"] = seq["input_ids"].tolist() + [
                    self.token_to_idx["<pad>"]
                ] * (max_read - len(seq["input_ids"]))
                seq["input_ids"] = seq["input_ids"][:max_read]

            seq["attention_mask"] = [
                int(x != self.token_to_idx["<pad>"]) for x in seq["input_ids"]
            ]
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
                k: [x[k] for x in padded_sequences] for k in padded_sequences[0].keys()
            }

        else:
            raise ValueError(f"Unknown return_tensors value '{return_tensors}'")

    __call__ = encode
