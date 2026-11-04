"""Immutable small device constants, not cached observations or model outputs."""
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def _constant(device,dtype,values):
    return torch.tensor(values,device=device,dtype=dtype)


def constant_like(reference,values):
    """Same dtype/rounding as reference.new_tensor; callers never mutate it."""
    return _constant(reference.device,reference.dtype,tuple(values))
