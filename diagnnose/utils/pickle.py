import os
import pickle
import dill
from typing import Any


def load_pickle(path: str, use_dill: bool = False) -> Any:
    with open(path, "rb") as f:
        if use_dill:
            out = dill.load(f)
        else:
            out = pickle.load(f)
    return out


def dump_pickle(content: Any, path: str) -> None:
    with open(os.path.expanduser(path), "wb") as f:
        pickle.dump(content, f)
