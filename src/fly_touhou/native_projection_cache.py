"""Reusable motion and collision inputs within one read-only observation."""

from contextlib import contextmanager
from contextvars import ContextVar

_PROJECTIONS = ContextVar("native_projections", default=None)


@contextmanager
def share_native_projections():
    if _PROJECTIONS.get() is not None:
        yield
        return
    token = _PROJECTIONS.set({})
    try:
        yield
    finally:
        _PROJECTIONS.reset(token)


def projection_cache(name):
    projections = _PROJECTIONS.get()
    return None if projections is None else projections.setdefault(name, {})
