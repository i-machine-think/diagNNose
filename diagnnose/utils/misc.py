import io
from collections import MutableMapping
from contextlib import redirect_stderr, redirect_stdout
from functools import wraps
from typing import Any, Callable, Dict


def suppress_print(func: Callable) -> Callable:
    """
    Function decorator to suppress output via print for testing purposed. Thanks to
    https://codingdose.info/2018/03/22/supress-print-output-in-python/ for text "entrapment".
    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:

        trap = io.StringIO()

        with redirect_stdout(trap), redirect_stderr(trap):
            result = func(*args, **kwargs)

        return result

    return wrapped


def merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """
    Update two dicts of dicts recursively, if either mapping has leaves
    that are non-dicts, the second's leaf overwrites the first's.
    Taken from: https://stackoverflow.com/a/24088493/3511979
    """
    for k, v in d1.items():
        if k in d2:
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = merge_dicts(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)

    return d3
