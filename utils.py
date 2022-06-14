import collections
import itertools


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


def concat_lists(lists):
    [isinstance(l, list,) for l in lists]
    return sum(lists, [])

def concat_tuples(tuples):
    [isinstance(l, tuple) for l in tuples]
    return sum(tuples, ())

def concat_iters(iters):
    return list(itertools.chain.from_iterable(iters))