import collections
import itertools

from beartype.typing import *
import rich



def print_dict(d: dict[str, Any]) -> None:
    max_len = len(max(d, key=lambda k: len(str(d[k])))) + 3
    for k, v in d.items():
        rich.print(f"\t- {k} =" + (max_len - len(k)) * " " + f" {v}")


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
