import os
import pickle
from typing import Any


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        out = pickle.load(f)
    return out


def dump_pickle(content: Any, path: str) -> None:
    with open(os.path.expanduser(path), "wb") as f:
        pickle.dump(content, f)
