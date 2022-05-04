
import numpy as np
import torch

class Tokenizer:
    def __init__(self, max_length, use_equal_symbol):
        self.vocab = [
            "<pad>", 
            "<bos>", 
            "<eos>", 
            "0", 
            "1", 
            "2", 
            "3", 
            "4", 
            "5", 
            "6", 
            "7", 
            "8", 
            "9", 
            "+", 
            "-", 
            "*", 
            "(", 
            ")",
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
        self.padding_side = "right"
        self.max_length = max_length

    def tokenize(self, input_str):
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
        
        output = np.array(output + [self.token_to_idx["<eos>"]], dtype=np.int64)
        return output
    
    def decode(self, input_tokens):
        assert type(input_tokens) == list, type(input_tokens)
        output = []
        for token_index in input_tokens:
            if isinstance(token_index, int) and token_index >= 0 and token_index < len(self.idx_to_token):
                output.append(self.idx_to_token[token_index])
            else:
                raise ValueError(f"Unknown token index '{token_index}'")
        
        return " ".join(output)

    def pad(self, features, padding, max_length, pad_to_multiple_of, return_tensors):
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
            return {k: np.array    ([x[k] for x in padded_sequences], dtype=np   .int64) for k in padded_sequences[0].keys()}
        elif return_tensors == "pt":
            return {k: torch.tensor([x[k] for x in padded_sequences], dtype=torch.int64) for k in padded_sequences[0].keys()}

        else:
            raise ValueError(f"Unknown return_tensors value '{return_tensors}'")

    def __call__(self, input_str):
        return self.tokenize(input_str)