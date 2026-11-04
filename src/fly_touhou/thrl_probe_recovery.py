"""Explicit frozen-evaluation recovery from sealed episodes, never live worlds.

Only complete acknowledged episodes are reusable. Interrupted action prefixes
are discarded and their original jobs start again from the declared opening.
The source is immutable; copied trace receipts retain their source identities.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .atomic_json_io import atomic_write_json as write
from .thrl_study_io import read_shared_json as read, utc_now
from .thrl_dodge_diagnostics import _shared_binary
from .thrl_probe_metrics import score_frames


def digest(path):
    with _shared_binary(Path(path)) as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sealed_rows(path):
    with _shared_binary(Path(path)) as stream:
        data = stream.read()
    end = data.rfind(b'\n') + 1
    rows = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return rows, dict(sha256=hashlib.sha256(data[:end]).hexdigest(),
                     sealed_bytes=end, ignored_incomplete_bytes=len(data)-end)


def recover_episodes(source, output, *, jobs, pins, frame_cap, action_mapping,
                     capture_maps, expected_native_seeds, config):
    """Validate all sealed provenance before copying or starting any worlds."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError('Recovery must preserve the separate immutable source')
    plan = read(source/'plan.json')
    if (plan.get('jobs') != jobs or plan.get('diagnostic_native_frame_cap') != frame_cap
            or plan.get('policy_action_mapping') != action_mapping
            or plan.get('capture_failure_history_maps') != capture_maps
            or plan.get('Bomb_forbidden_in_all_conditions') is not True
            or plan.get('training_modified') is not False
            or plan.get('device') != 'cuda:0'):
        raise ValueError('Interrupted evaluation protocol or complete job plan differs')
    if config != read(source.parent/'evaluation_input/config.json'):
        raise ValueError('Interrupted frozen evaluation input configuration differs')
    if len(expected_native_seeds) != len({job['game'] for job in jobs}):
        raise ValueError('Original native test cohort must be explicit')
    for name in ('initial', 'trained'):
        old = plan['pins'][name]
        if (old['sha256'] != pins[name]['sha256']
                or digest(source/(name+'.pt')) != old['sha256']):
            raise ValueError('Interrupted frozen checkpoint identity differs')
    rows, journal = sealed_rows(source/'episodes.jsonl')
    declared = {(job['policy'], job['game']): job for job in jobs}
    completed, traces = set(), []
    from .thrl_native_trace_replay import RecordedAvoidanceCase
    for row in rows:
        key = row['policy'], row['game']
        if key not in declared or key in completed:
            raise ValueError('Duplicate or undeclared sealed evaluation episode')
        job = declared[key]
        if (any(row.get(k) != v for k, v in job.items())
                or row.get('seed_applied') is not True
                or row.get('native_seed') != expected_native_seeds[row['game']]
                or row.get('actual_bombs_used') != 0
                or row.get('diagnostic_native_frame_cap') != frame_cap
                or row.get('schema') != plan['schema']
                or type(row.get('policy_decisions')) is not int or row['policy_decisions'] <= 0
                or len(row.get('action_sha256', '')) != 64):
            raise ValueError('Sealed native endpoint or seed provenance differs')
        hit, capped = row.get('first_collision_observed'), row.get('right_censored_at_cap')
        if (type(hit) is not bool or type(capped) is not bool or hit == capped
                or (capped and row['native_frames'] < frame_cap)):
            raise ValueError('Sealed episode has no valid terminal observation')
        score_frames(row, frame_cap)
        if row['policy'] in ('trained_sampled', 'trained_greedy'):
            case = RecordedAvoidanceCase(source/'report.json', row)
            receipt = case.receipt
            if (receipt['checkpoint_sha256'] != pins['trained']['sha256']
                    or any(row.get(k) != v for k, v in receipt['endpoint'].items())
                    or (capture_maps and 'maps' not in case.arrays)):
                raise ValueError('Sealed trace checkpoint, endpoint or map history differs')
            traces.append((row, deepcopy(receipt)))
            del case
        elif 'failure_trace' in row:
            raise ValueError('Unexpected baseline-policy trace')
        completed.add(key)
    # Relocate exact sensor/tape bytes; rewrite only the path-bearing receipt.
    result = deepcopy(rows)
    indexed = {(r['policy'], r['game']): r for r in result}
    copied = []
    for row, receipt in traces:
        pointer = row['failure_trace']
        folder = output/'failure_traces'; folder.mkdir(exist_ok=True)
        source_arrays, source_receipt = Path(pointer['arrays']), Path(pointer['receipt'])
        source_tape = Path(receipt['full_episode_action_tape']['path'])
        arrays, tape = folder/source_arrays.name, folder/source_tape.name
        for src, dest in ((source_arrays, arrays), (source_tape, tape)):
            if dest.exists(): raise FileExistsError(dest)
            shutil.copyfile(src, dest)
            if digest(src) != digest(dest): raise ValueError('Recovery copy hash differs')
        descriptor = dict(receipt['full_episode_action_tape'], path=str(tape))
        receipt.update(arrays_file=str(arrays), full_episode_action_tape=descriptor,
                       recovered_from_receipt=str(source_receipt),
                       recovered_from_receipt_sha256=digest(source_receipt))
        target = folder/source_receipt.name
        if target.exists(): raise FileExistsError(target)
        write(target, receipt)
        indexed[row['policy'], row['game']]['failure_trace'] = dict(
            pointer, receipt=str(target), arrays=str(arrays), full_episode_action_tape=descriptor)
        copied.append(dict(policy=row['policy'], game=row['game'], source_receipt=str(source_receipt),
                           source_receipt_sha256=digest(source_receipt), arrays_sha256=pointer['sha256']))
    missing = [job for job in jobs if (job['policy'], job['game']) not in completed]
    recovery = dict(schema='B1-frozen-probe-sealed-episode-recovery-v1', utc=utc_now(),
        source=str(source), source_plan_sha256=digest(source/'plan.json'), journal=journal,
        inherited_completed_episodes=len(result), missing_jobs=missing,
        incomplete_native_worlds_restarted_from_original_opening=True,
        incomplete_action_prefixes_are_not_scores=True, copied_traces=copied,
        source_modified=False, optimizer_updates=0)
    write(output/'recovery.json', recovery)
    with (output/'episodes.jsonl').open('x', encoding='utf-8') as stream:
        for row in result: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    return result, missing, recovery
