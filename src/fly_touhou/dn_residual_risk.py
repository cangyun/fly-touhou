"""Compatibility import for existing scripts and serialized references."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("fly_touhou.policy.dn_residual_risk")
