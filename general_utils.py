import collections
import h5py
import functools
import inspect
import itertools
import math
from pathlib import Path
import subprocess

from beartype.typing import *
import numpy as np
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

def shorten_path(path):
    path = Path(path)
    if path.is_relative_to(SCRIPT_DIR):
        path = "<pwd> /" + str(path.relative_to(SCRIPT_DIR))
    else:
        path = str(path)
    return path

def print_list(_list):
    at_least_one = False
    for line in _list:
        at_least_one = True
        if isinstance(line, Path):
            line = shorten_path(line)
        rich.print(f"\t- {line}")

    if not at_least_one:
        rich.print("\t[bright_black]<empty list>")

def print_dict(_dict: dict[str, Any]) -> None:
    # Pad by key length
    max_len = len(max(_dict, key=lambda key: len(str(key)))) + 1
    at_least_one = False
    for k, value in _dict.items():
        at_least_one = True
        if isinstance(value, Path):
            value = shorten_path(value)

        rich.print(f"\t- {k} =" + (max_len - len(k)) * " " + f" {value}")

    if not at_least_one:
        rich.print("\t[bright_black]<empty dict>")

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
    assert all(isinstance(l, list) for l in lists)
    return sum(lists, [])


def concat_tuples(tuples):
    assert all(isinstance(l, tuple) for l in tuples)
    return sum(tuples, ())


def concat_iters(iters):
    return list(itertools.chain.from_iterable(iters))


def str_gt(a: str, b: str, verbose=False):
    """
    Function to compare strings that have numbers in them, to 
    get a natural sorting order. Useful for paths. 
    """
    number_chars_a = ""
    number_chars_b = ""
    
    if a == b:
        return False
    
    for aa, bb in itertools.zip_longest(a, b, fillvalue=None):
        # Break both streaks or just one of them
        if aa is None or bb is None:
            if aa is not None:
                if aa.isnumeric():
                    number_chars_a += aa
                if verbose:
                    print("Break. bb is None, aa is not None", (aa, bb), (number_chars_a, number_chars_b))
                break

            if bb is not None:
                if bb.isnumeric():
                    number_chars_b += bb
                if verbose:
                    print("Break. aa is None, bb is not None", (aa, bb), (number_chars_a, number_chars_b))
                break

        both_not_numeric_not_equal = not aa.isnumeric() and not bb.isnumeric() and aa != bb
        one_numeric_one_not = not aa.isnumeric() and bb.isnumeric() or aa.isnumeric() and not bb.isnumeric()
        
        if one_numeric_one_not:
            if aa.isnumeric():
                number_chars_a += aa
                if verbose:
                    print("Break. aa is numeric, not bb", (aa, bb), (number_chars_a, number_chars_b))
                break

            if bb.isnumeric():
                number_chars_b += bb
                if verbose:
                    print("Break. bb is numeric, not aa", (aa, bb), (number_chars_a, number_chars_b))
                break
            if verbose:
                print("End. One numeric one not.", (aa, bb), (number_chars_a, number_chars_b))            
            return aa > bb

        if both_not_numeric_not_equal:
            if number_chars_a or number_chars_b:
                if verbose:    
                    print("Streak Break. Both not numeric not equal.", (aa, bb), (number_chars_a, number_chars_b))
                break
            if verbose:
                print("End. No streak, both not numeric not equal.", (aa, bb), (number_chars_a, number_chars_b))
            return aa > bb

        # Streak continues
        if aa.isnumeric() and bb.isnumeric():
            number_chars_a += aa
            number_chars_b += bb
            if verbose:
                print("Numeric streak", (aa, bb), (number_chars_a, number_chars_b))
            continue

        if aa == bb and not aa.isnumeric() and not bb.isnumeric():
            if number_chars_a != number_chars_b:
                if verbose:
                    print("Streak Break. Both not numeric.", (aa, bb), (number_chars_a, number_chars_b))
                break
            else:
                number_chars_a = ""
                number_chars_b = ""

            if verbose:
                print("Continuing. Both not numeric, no numbers thus far.", (aa, bb), (number_chars_a, number_chars_b))
            continue
    
    if number_chars_a != number_chars_b:
        if number_chars_a and not number_chars_b:
            return True
        if not number_chars_a and number_chars_b:
            return False

        if verbose:
            print("End of loop break, Numbers are compared", (a, b), (number_chars_a, number_chars_b))
            
        return int(number_chars_a) > int(number_chars_b)

    assert False


def cmp_str(a, b):
    assert isinstance(a, (str, Path))
    assert isinstance(b, (str, Path))

    a = str(a)
    b = str(b)

    if a == b:
        return 0
    if str_gt(a, b):
        return 1
    else:
        return -1

def sort_iterable_text(list_text):
    return sorted(list_text, key=functools.cmp_to_key(cmp_str))

def find_last(seq: Sequence[Any], item: Any) -> int:
    return len(seq) - seq[::-1].index(item) - 1

def cmd(command: list[str]) -> list[str]:
    return subprocess.check_output(command).decode("utf-8").strip().split("\n")

def only_one(it: Iterable):
    iterated = iter(it)
    good = next(iterated)
    for bad in iterated:
        raise ValueError("Expected only one item, got more than one.")
    return good


def count_lines(path: Path) -> int:
    return int(check_len(only_one(cmd(["wc", "-l", str(path)])).split(), 2)[0])

def check_len(seq: Sequence, expected_len: int) -> Sequence:
    if not len(seq) == expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(seq)}.")
    return seq

SIZE_HUMAN_NAMES = {
        0: "B",
        1: "KB",
        2: "MB",
        3: "GB",
    }

def to_human_size(size: int) -> str:
    if size == 0:
        return "0 B"

    exponent = int(math.log(size, 1000))
    mantissa = size / 1000 ** exponent
    return f"{mantissa:.2f} {SIZE_HUMAN_NAMES[exponent]}"

def print_structure_h5(file_object: h5py.File):
    work_stack = [(file_object, "")]
    POSSIBLE_TYPES = (h5py.Dataset, h5py.Group, h5py.File)

    all_text = []

    while work_stack:

        obj, parent_name = work_stack.pop()
        assert isinstance(obj, (h5py.Group, h5py.Dataset, h5py.File))

        if obj.name == "":
            obj_name = "<root>"
        else:
            obj_name = obj.name
        
        if parent_name:    
            name = parent_name + "/" + obj_name
        else:
            name = obj_name

        message = f"\"{name}\": {type(obj).__name__}"
        if isinstance(obj, h5py.Dataset):
            message += f" {obj.shape} {obj.dtype}"
        all_text.append(message)

        if obj.attrs:
            for k, v in obj.attrs.items():
                message_attr = f"\t- {k}: {type(v).__name__}"
                if isinstance(v, (str, int, float)):
                    message_attr += f" value=\"{v}\""
                elif isinstance(v, np.ndarray):
                    message_attr += f" shape={v.shape} dtype={v.dtype}"
                elif isinstance(v, (tuple, list, dict)):
                    message_attr += f" {type(v)} {len(v)}"
                all_text.append(message_attr)

        if hasattr(obj, "items"):
            assert isinstance(obj, (h5py.Group, h5py.File))
            for key, value in obj.items():
                work_stack.append((value, parent_name))

    rich.print("\n".join(all_text))