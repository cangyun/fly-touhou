"""Flush important evaluation records before acknowledging their completion."""
import json
import os
from pathlib import Path

from .atomic_json_io import atomic_write_json


def write_json(path, value):
    atomic_write_json(path, value)
    # Atomic replacement alone does not flush the replaced file to storage.
    # This reduces loss on reboot; it cannot guarantee faulty hardware caches.
    with Path(path).open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path, value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
