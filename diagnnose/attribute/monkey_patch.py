from functools import wraps

import torch
from torch import Tensor
from torch._overrides import handle_torch_function, has_torch_function

MONKEY_PATCH_PERFORMED = False


# Not all torch functions correctly implement __torch_function__ yet:
# https://github.com/pytorch/pytorch/issues/34294
def monkey_patch_fn(original_fn):
    @wraps(original_fn)
    def fn(tensors, dim=0, out=None):
        if not torch.jit.is_scripting():
            if any(type(t) is not Tensor for t in tensors) and has_torch_function(
                tensors
            ):
                return handle_torch_function(fn, tensors, tensors, dim=dim, out=out)
        return original_fn(tensors, dim=dim, out=out)

    return fn


def monkey_patch_tensor(original_fn):
    @wraps(original_fn)
    def fn(self, other):
        if isinstance(other, Tensor):
            return original_fn(self, other)
        return original_fn(self, other.data)

    return fn


def monkey_patch():
    torch.cat = monkey_patch_fn(torch.cat)
    torch.stack = monkey_patch_fn(torch.stack)
    Tensor.expand_as = monkey_patch_tensor(Tensor.expand_as)
    Tensor.type_as = monkey_patch_tensor(Tensor.type_as)
