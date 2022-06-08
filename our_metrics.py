import itertools

import abc
from beartype import beartype
import rich
import torch
import numpy as np

import our_tokenizer


class OurMetric(abc.ABC):
    @classmethod
    def prepare(cls, tokenizer: our_tokenizer.Tokenizer, pred, label, do_print):
        things_to_ignore = {
            -100,
        } | tokenizer.special_token_ids

        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()

        cleaned_preds = [x for x in pred if x not in things_to_ignore]
        cleaned_labels = [x for x in label if x not in things_to_ignore]

        return dict(cleaned_preds=cleaned_preds, cleaned_labels=cleaned_labels)

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")


class EM(OurMetric):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, pred: list, label: list, do_print=False, descr=""):
        prepped_decoded = list(pred)
        prepped_label = list(label)

        if prepped_decoded == prepped_label:
            self.correct += 1
        else:
            pass

        if do_print:
            rich.print(
                f"(EM) Answer - {descr}: "
                + ", ".join(
                    [
                        f"[green]{a}" if a == b else f"[red]{a}"
                        for a, b in itertools.zip_longest(
                            prepped_decoded, prepped_label, fillvalue=""
                        )
                        if a
                    ]
                )
            )
            rich.print(
                f"(EM) Label  - {descr}: "
                + ", ".join(
                    [
                        f"[green]{b}" if a == b else f"[red]{b}"
                        for a, b in itertools.zip_longest(
                            prepped_decoded, prepped_label, fillvalue=""
                        )
                        if b
                    ]
                )
            )

        self.total += 1

    def compute(self):
        return self.correct / self.total


# class RecallAcc:
#     def __init__(self):
#         self.recall_accuracies = []

#     @beartype
#     def add(self, pred: list, label: list, do_print: bool = False, descr=""):
#         recall_acc_decoded = list(pred)
#         recall_acc_label = list(label)

#         if len(recall_acc_decoded) < len(recall_acc_label):
#             recall_acc_decoded += [0] * (len(recall_acc_la el) - len(recall_acc_decoded))
#         elif len(recall_acc_decoded) > len(recall_acc_label):
#             recall_acc_decoded = recall_acc_decoded[:len(recall_acc_label)]

#         recall_acc_label_np =   np.array(recall_acc_label,   dtype=np.int64)
#         recall_acc_decoded_np = np.array(recall_acc_decoded, dtype=np.int64)
#         recall_acc =            np.mean(recall_acc_decoded_np == recall_acc_label_np)

#         self.recall_accuracies.append(recall_acc)

#     def compute(self):
#         return np.mean(self.recall_accuracies)


# class PrecisionAcc:
#     def __init__(self):
#         self.precision_accuracies = []

#     @beartype
#     def add(self, pred: list, label: list, do_print: bool = False, descr=""):
#         precision_acc_decoded = list(pred)
#         precision_acc_label = list(label)

#         if len(precision_acc_decoded) > len(precision_acc_label):
#             precision_acc_label += [0] * (len(precision_acc_decoded) - len(precision_acc_label))
#         elif len(precision_acc_decoded) < len(precision_acc_label):
#             precision_acc_label = precision_acc_label[:len(precision_acc_decoded)]

#         precision_acc_label_np =   np.array(precision_acc_label,   dtype=np.int64)
#         precision_acc_decoded_np = np.array(precision_acc_decoded, dtype=np.int64)
#         precision_acc =            np.mean(precision_acc_decoded_np == precision_acc_label_np)

#         self.precision_accuracies.append(precision_acc)

#     def compute(self):
#         return np.mean(self.precision_accuracies)
