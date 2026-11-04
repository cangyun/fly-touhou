"""Complete JSON replacement with bounded Windows sharing-lock retries.

Only this writer's unique temporary file is cleaned up. Existing permissions
are never changed, and replacement errors retain their original exception.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time


def atomic_write_json(path, value, *, replace_timeout_seconds=2.0,
                      retry_interval_seconds=0.01):
    """Write closed JSON beside ``path``, then atomically replace its target.

    Retry only Windows errors 5/32/33. The replace deadline starts after JSON
    serialization and file close; exhausted retries re-raise the actual last
    OSError, rather than replacing it with a synthetic timeout exception.
    """
    timeout, interval = float(replace_timeout_seconds), float(retry_interval_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('replace_timeout_seconds must be finite and nonnegative')
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('retry_interval_seconds must be finite and positive')
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=target.parent, prefix='.'+target.name+'.', suffix='.tmp',
                delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        deadline = time.monotonic()+timeout
        while True:
            try:
                os.replace(temporary, target)
                temporary = None  # Replacement consumed only our temp file.
                return
            except OSError as error:
                remaining = deadline-time.monotonic()
                if getattr(error, 'winerror', None) not in (5, 32, 33) or remaining <= 0:
                    raise
                time.sleep(min(interval, remaining))
                if time.monotonic() >= deadline:
                    raise
    except BaseException as error:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                # Preserve the original write/replace failure even when the
                # OS also refuses cleanup; never touch another writer's temp.
                error.add_note('Own JSON temporary cleanup failed: '+repr(cleanup_error))
        raise
