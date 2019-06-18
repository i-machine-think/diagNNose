import os
import pickle
from typing import Any


# https://stackoverflow.com/a/1176023
def camel2snake(string: str) -> str:
    import re

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def dump_pickle(content: Any, path: str) -> None:
    with open(os.path.expanduser(path), 'wb') as f:
        pickle.dump(content, f)
