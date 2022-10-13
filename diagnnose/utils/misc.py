import contextlib
import cProfile
import io
import pstats
from collections.abc import MutableMapping
from functools import wraps
from typing import Any, Callable, Dict


def suppress_print(func: Callable) -> Callable:
    """
    Function decorator to suppress print output for testing purposes.

    If ``suppress_print: False`` is part of the ``**kwargs`` of the
    wrapped method the output won't be suppressed.
    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not kwargs.pop("suppress_print", True):
            return func(*args, **kwargs)

        trap = io.StringIO()

        with contextlib.redirect_stdout(trap), contextlib.redirect_stderr(trap):
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


@contextlib.contextmanager
def profile() -> None:
    """
    Profiler that operates as a context manager. Example usage:

    .. code-block:: python

        with profile():
            foo()
            bar()
    """
    pr = cProfile.Profile()
    pr.enable()

    yield

    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats()
