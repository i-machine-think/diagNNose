import pickle
from typing import Any


def trim(path: str) -> str:
    return path[:-1] if path[-1] == '/' else path


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def dump_pickle(content: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(content, f)
