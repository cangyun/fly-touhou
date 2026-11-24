"""Allow concurrent frozen NBNM jobs and give training exclusive use of their device."""
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import time


@contextmanager
def gpu_phase(mode, *, database=None):
    if mode not in ('training', 'evaluation'):
        raise ValueError('Expected training or evaluation')
    if database is None:
        default = Path(__file__).resolve().parents[3] / 'outputs' / 'control' / 'nbnm-gpu0.sqlite3'
        database = Path(os.environ.get('FLY_TOUHOU_NBNM_PHASE_DB', default))
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=1.0, isolation_level=None)
    announced = False
    try:
        while True:
            try:
                connection.execute('BEGIN EXCLUSIVE' if mode == 'training' else 'BEGIN')
                # Reading the schema acquires a shared lock even in an empty database.
                # This dedicated rollback-journal database is never switched to WAL.
                connection.execute('SELECT name FROM sqlite_master LIMIT 1').fetchall()
                break
            except sqlite3.OperationalError as error:
                if getattr(error, 'sqlite_errorcode', None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise
                connection.rollback()
                if not announced:
                    print(f'Waiting for compatible NBNM jobs before {mode}.', flush=True)
                    announced = True
                time.sleep(.25)
        yield
    finally:
        connection.close()
