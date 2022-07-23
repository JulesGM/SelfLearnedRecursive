import collections
import inspect
import itertools
from pathlib import Path

from beartype.typing import *
import rich

SCRIPT_DIR = Path(__file__).absolute().parent

def check_and_print_args(all_arguments, function):
    check_args(all_arguments, function)
    rich.print("[bold]Arguments:")
    print_dict(all_arguments)
    print()


def check_args(all_arguments, function):
    # We get the arguments by calling `locals`. This makes sure that we
    # really called locals at the very beginning of the function, otherwise
    # we have supplementary keys.
    assert all_arguments.keys() == inspect.signature(function).parameters.keys(), (
        f"\n{sorted(all_arguments.keys())} != "
        f"{sorted(inspect.signature(function).parameters.keys())}"
    )


def print_dict(_dict: dict[str, Any]) -> None:
    # Pad by key length
    max_len = len(max(_dict, key=lambda key: len(str(key)))) + 1
    for k, value in _dict.items():
        if isinstance(value, Path):
            if value.is_relative_to(SCRIPT_DIR):
                value = "<pwd> /" + str(value.relative_to(SCRIPT_DIR))
            else:
                value = str(value)

        rich.print(f"\t- {k} =" + (max_len - len(k)) * " " + f" {value}")


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


def dict_unzip(list_of_dicts):
    """
    Unzips a list of dicts into a dict of lists
    """
    keys = list_of_dicts[0].keys()
    dict_of_lists = collections.defaultdict(list)
    for i, ld in enumerate(list_of_dicts):
        assert ld.keys() == keys, f"{ld.keys()} != {keys}"
        for k in keys:
            dict_of_lists[k].append(ld[k])
    return dict_of_lists


def find_last(seq, item):
    return len(seq) - seq[::-1].index(item) - 1


def concat_lists(lists):
    assert all(isinstance(l, list) for l in lists)
    return sum(lists, [])


def concat_tuples(tuples):
    assert all(isinstance(l, tuple) for l in tuples)
    return sum(tuples, ())


def concat_iters(iters):
    return list(itertools.chain.from_iterable(iters))
