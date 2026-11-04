"""Validate saved full-game endpoints without inventing interrupted outcomes."""
from pathlib import Path
import hashlib
import json
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def journal_rows(path):
    path = Path(path)
    if not path.exists():
        return [], dict(exists=False, corrupt_records=[])
    data = path.read_bytes()
    rows, corrupt = [], []
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode('utf-8-sig')))
        except (ValueError, UnicodeError):
            corrupt.append(dict(line=number, bytes=len(line), NUL_bytes=line.count(b'\0')))
    return rows, dict(exists=True, sha256=hashlib.sha256(data).hexdigest(),
                      corrupt_records=corrupt, source_modified=False)


def verify_episode_trace(row, folder, checkpoint_sha256):
    """Check real source19 tapes, exact endpoint and array/receipt provenance."""
    folder = (Path(folder) / 'full_game_traces').resolve()
    events = row.get('diagnostic_traces', {}).get('events', [])
    terminals = []
    for pointer in events:
        receipt_path, arrays = Path(pointer['receipt']).resolve(), Path(pointer['arrays']).resolve()
        if not receipt_path.is_relative_to(folder) or not arrays.is_relative_to(folder):
            raise ValueError('Foreign full-game diagnostic path')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        if (receipt.get('checkpoint_sha256') != checkpoint_sha256
                or receipt.get('game') != row['evaluation_game']-1
                or any(receipt.get(k) != row[k] for k in ('evaluation_seed', 'action_seed', 'native_seed'))
                or receipt.get('policy_action_count') != 19 or receipt.get('Bomb_allowed') is not True
                or receipt.get('actions_changed') is not False
                or receipt.get('sha256') != pointer['sha256'] or digest(arrays) != pointer['sha256']):
            raise ValueError('Original full-game trace identity differs')
        with zipfile.ZipFile(arrays) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('Damaged original full-game input arrays')
        tape_info = receipt['action_tape']
        tape_path = Path(tape_info['path']).resolve()
        if not tape_path.is_relative_to(folder):
            raise ValueError('Foreign original action tape')
        tape = tape_path.read_bytes()
        if (hashlib.sha256(tape).hexdigest() != tape_info['sha256']
                or len(tape) != tape_info['decisions'] or not tape or max(tape) >= 19):
            raise ValueError('Original acknowledged source19 action tape differs')
        if receipt.get('natural_terminal'):
            if (tape_info.get('complete_natural_episode') is not True
                    or len(tape) != row['evaluation_decisions']
                    or tape_info['sha256'] != row['action_sha256']
                    or any(row.get(k) != v for k, v in receipt['episode_summary'].items())):
                raise ValueError('Saved terminal does not match the acknowledged full-game row')
            terminals.append(receipt)
    if len(terminals) != 1:
        raise ValueError('Exactly one verified natural terminal is required for reuse')
    return dict(game=row['evaluation_game'], native_seed=row['native_seed'],
                verified_events=len(events), natural_terminal=True,
                checkpoint_sha256=checkpoint_sha256)
