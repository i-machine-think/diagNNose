import io
from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from typing import Any, Callable


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
