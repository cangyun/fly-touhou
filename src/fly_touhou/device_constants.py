"""Persistent small constants, including during CUDA graph capture."""
import torch
_constants={}


def constant_like(reference, values):
    key=(reference.device,reference.dtype,values)
    if key not in _constants:
        _constants[key]=torch.tensor(values,device=reference.device,dtype=reference.dtype)
    return _constants[key]


def movement_velocities(reference):
    key=(reference.device,reference.dtype,'source18_velocities')
    if key not in _constants:
        from .thrl_movement_teacher import MovementSafetyTeacher
        _constants[key]=reference.new_tensor(MovementSafetyTeacher().velocities)/4.5
    return _constants[key]
