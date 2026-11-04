"""Stream original FP32 V5 packets into small signed4f geometric-proxy labels.

No perception/core/neural teacher is constructed. Original native snapshots
were not archived, so chosen transition outcomes remain explicitly unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SCHEMA = 'rule-world000-original-V5-signed4f-proxy-cache-v1'
INDEX_SHA = 'f289a401ed208d1430c0b3cd30d9ed5d6cb999218d64ca00b41b5fc95df8487e'
SOURCE = Path('./outputs/engineering/B1-autonomous-dodge-20261024/teacher-full-natural30-v1/trajectories/world000')
SCALAR_FIELDS = ('actions', 'dones', 'episode_starts', 'native_episodes', 'native_seeds',
    'before_frames', 'after_observation_frames', 'stage', 'stage_frame',
    'before_episode_frames', 'after_episode_frames', 'native_misses', 'native_bombs_used')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def write_packet(path, arrays):
    if path.exists():
        raise FileExistsError('Preserve an existing label packet')
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('xb') as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    return dict(name=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args(argv)
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import torch
    torch.set_num_threads(2)
    from fly_touhou.thrl_teacher_archive import TeacherTrajectoryReader
    from fly_touhou.thrl_dodge_supervision import prepare_dense_targets
    from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
    from fly_touhou.thrl_movement_teacher import normal_unprotected
    if a.output.exists():
        raise FileExistsError('Fresh output only; no overwrite or automatic resume')
    a.output.mkdir(parents=True)
    (a.output / 'labels').mkdir()
    a.receipt.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reader = TeacherTrajectoryReader(SOURCE, require_complete=True, cache_bytes=0)
    index = reader.index
    if sha(SOURCE / 'index.json') != INDEX_SHA or index['rows'] != 24755 or len(index['shards']) != 774:
        raise ValueError('Exactly the audited complete world000 source is required')
    if index['metadata']['input_schema'] != SEPARATED_LASER_INPUT:
        raise ValueError('Original V5 only; no padding or schema conversion')
    if index['native_episode_summary']['outcome'] != 'clear':
        raise ValueError('This bounded export is the audited six-stage clear')
    source_files = ['thrl_teacher_archive.py', 'thrl_dodge_supervision.py', 'thrl_movement_teacher.py',
        'thrl_laser_observation.py', 'thrl_geometry_observation.py', 'thrl_inputs.py']
    pins = {name: sha(ROOT / 'src/fly_touhou' / name) for name in source_files}
    identity = dict(schema=SCHEMA, source_root=str(SOURCE), source_index_sha256=INDEX_SHA,
        original_archive_schema=index['schema'], input_schema=SEPARATED_LASER_INPUT,
        source_metadata=index['metadata'], source_summary={k:index['native_episode_summary'][k]
            for k in ('outcome', 'stage_history', 'native_frames', 'policy_decisions')},
        producer='unmodified thrl_dodge_supervision.prepare_dense_targets', producer_source_SHA=pins,
        producer_arguments=dict(horizon_specific=True, max_labels='entire packet rows<=32, no successful candidate subsampling'),
        target='original clipped[-64,64] 4-native-frame current-motion geometric proxy over18 movement actions',
        factual_outcomes='all -1 unknown: original paired raw before/after info objects not archived; no fabricated factual veto',
        original_Bomb_or_unknown_does_not_automatically_discard_before_geometry=True,
        compression='np.savez_compressed, original float32 features and generated float32 margins, no quantization',
        source_raw_maps_objects_copied_to_output=False, source_raw_native_snapshot_not_available=True,
        finite_scope=dict(episodes=1, rows=24755, packets=774), CPU_only=True,
        process=dict(pid=process.pid, create_time=process.create_time()))
    atomic_json(a.output / 'identity.json', identity)
    counts = Counter()
    by_stage = defaultdict(Counter)
    abstentions = Counter()
    records = []
    progress = dict(schema=SCHEMA, status='preflight_original_geometry_API', rows=0, packets=0)
    atomic_json(a.output / 'progress.json', progress)
    geometry_seconds = source_seconds = write_seconds = 0.
    preflight_cache = None
    previous_after = previous_after_episode = None

    def produce(part_number):
        nonlocal geometry_seconds, source_seconds
        spec = index['shards'][part_number]
        begin = time.monotonic()
        block = reader.fetch(spec['start'], spec['start'] + spec['rows'],
            fields=['features', 'maps', 'objects', *SCALAR_FIELDS])
        source_seconds += time.monotonic() - begin
        n = spec['rows']
        if any(not np.isfinite(block[k]).all() for k in ('features', 'maps', 'objects')):
            raise ValueError('Nonfinite original observation')
        unknown = np.full((n, 1), -1, dtype=np.int8)
        begin = time.monotonic()
        margins, trust, evidence = prepare_dense_targets(block['features'][:, None],
            block['objects'][:, None], block['maps'][:, None], block['actions'][:, None], unknown,
            input_schema=SEPARATED_LASER_INPUT, max_labels=n, horizon_specific=True)
        geometry_seconds += time.monotonic() - begin
        signed = margins[:, 0, 0].numpy().copy()
        trusted4 = trust[:, 0, 0].numpy().copy()
        normal = normal_unprotected(block['features']).copy()
        if signed.shape != (n, 18) or signed.dtype != np.float32 or not np.isfinite(signed).all():
            raise ValueError('Original signed4f shape/FP32/finiteness contract failed')
        if not np.array_equal(trusted4 & normal, trusted4):
            raise ValueError('Original trusted labels must remain normal/unprotected')
        features = block['features']
        present = block['objects'][:, :, 0] > .5
        laser = present & ((block['objects'][:, :, 3] > .5) | (block['objects'][:, :, 4] > .5))
        arrays = {k: block[k].copy() for k in SCALAR_FIELDS}
        arrays.update(features=features.copy(), source_row=np.arange(spec['start'], spec['start'] + n, dtype=np.int64),
            source_part=np.full(n, part_number, dtype=np.int32), source_local_row=np.arange(n, dtype=np.int16),
            signed_margin4=signed, trusted4=trusted4, normal_unprotected_mask=normal,
            protection_or_non_normal_mask=~normal, normal_state_mask=features[:, 21] > .5,
            invincibility_positive_mask=features[:, 5] > 0, protection_timer_positive_mask=features[:, 20] > 0,
            geometry_valid_mask=trusted4 & normal, original_row_valid_mask=np.ones(n, bool),
            elapsed_native_frames=block['after_observation_frames'] - block['before_frames'],
            four_frame_transition_mask=(block['after_observation_frames'] - block['before_frames']) == 4,
            outcomes=np.full(n, -1, np.int8), fact_known_mask=np.zeros(n, bool),
            selected_laser_mask=laser.any(1),
            selected_laser_unknown_motion_mask=(laser & (block['objects'][:, :, 26] <= .5)).any(1),
            player_circle_component=features[:, 13] * 64.)
        del block, margins, trust
        return arrays, evidence

    def checkpoint_progress(status='running'):
        progress.update(status=status, rows=int(counts['rows']), packets=len(records),
            elapsed_seconds=time.monotonic()-started, geometry_seconds=geometry_seconds,
            source_seconds=source_seconds, write_seconds=write_seconds,
            trusted4=int(counts['trusted4']), mixed_danger_safe=int(counts['mixed_danger_safe']))
        atomic_json(a.output / 'progress.json', progress)
        print(json.dumps(progress), flush=True)

    try:
        # One actual six-stage packet establishes the original API's trusted
        # proxy. Its SMALL output is retained and used at packet702, so all774
        # original packets are decoded and geometrically labeled exactly once.
        preflight_cache = produce(702)
        if not preflight_cache[0]['trusted4'].any():
            raise ValueError('Known six-stage packet gave no original trusted proxy; stop, do not invent targets')
        identity['API_preflight'] = dict(source_part=702, rows=32,
            trusted4=int(preflight_cache[0]['trusted4'].sum()),
            original_evidence=preflight_cache[1], reused_at_source_part702=True)
        atomic_json(a.output / 'identity.json', identity)
        print(json.dumps(dict(API_preflight=identity['API_preflight'])), flush=True)
        for part_number, spec in enumerate(index['shards']):
            if part_number == 702:
                arrays, evidence = preflight_cache
                preflight_cache = None
            else:
                arrays, evidence = produce(part_number)
            n = spec['rows']
            before, after = arrays['before_frames'], arrays['after_observation_frames']
            before_ep, after_ep = arrays['before_episode_frames'], arrays['after_episode_frames']
            if (np.any(after <= before) or not np.array_equal(before[1:], after[:-1])
                    or not np.array_equal(before_ep[1:], after_ep[:-1])):
                raise ValueError('Original row clocks do not form an acknowledged observation stream')
            if previous_after is not None and (int(before[0]) != previous_after or int(before_ep[0]) != previous_after_episode):
                raise ValueError('Cross-packet clock discontinuity; preserve source, no fabricated bridge')
            if not np.all(arrays['native_seeds'] == 11655) or not np.all(arrays['native_episodes'] == 1):
                raise ValueError('Original episode/seed changed within complete world000')
            rows = arrays['source_row']
            if not np.array_equal(arrays['episode_starts'], rows == 0):
                raise ValueError('Only real natural row0 may reset')
            if not np.array_equal(arrays['dones'], rows == 24754):
                raise ValueError('Only true final row may be terminal')
            previous_after, previous_after_episode = int(after[-1]), int(after_ep[-1])
            trusted = arrays['geometry_valid_mask']
            signed = arrays['signed_margin4']
            danger, safe = signed <= 0, signed > 1
            hard = trusted & danger.any(1) & safe.any(1)
            arrays['mixed_danger_safe_mask'] = hard
            for mask, counter in [(np.ones(n, bool), counts),
                    *[(arrays['stage'] == stage, by_stage[str(int(stage))]) for stage in np.unique(arrays['stage'])]]:
                good = mask & trusted
                counter.update(rows=int(mask.sum()), trusted4=int(good.sum()),
                    normal_unprotected=int((mask & arrays['normal_unprotected_mask']).sum()),
                    protection_or_non_normal=int((mask & arrays['protection_or_non_normal_mask']).sum()),
                    mixed_danger_safe=int((mask & hard).sum()),
                    trusted_any_danger=int((good & danger.any(1)).sum()),
                    trusted_all_safe=int((good & safe.all(1)).sum()),
                    dangerous_direction_queries=int((danger & good[:, None]).sum()),
                    safe_direction_queries=int((safe & good[:, None]).sum()),
                    non4frame_transitions=int((mask & ~arrays['four_frame_transition_mask']).sum()),
                    selected_laser=int((mask & arrays['selected_laser_mask']).sum()),
                    selected_laser_unknown_motion=int((mask & arrays['selected_laser_unknown_motion_mask']).sum()),
                    player_circle_gt8=int((mask & (arrays['player_circle_component'] > 8)).sum()),
                    episode_starts=int((mask & arrays['episode_starts']).sum()),
                    terminals=int((mask & arrays['dones']).sum()))
            abstentions.update(evidence['abstentions'])
            start_write = time.monotonic()
            record = write_packet(a.output / 'labels' / f'labels-{part_number:06d}.npz', arrays)
            write_seconds += time.monotonic() - start_write
            record.update(start=spec['start'], rows=n, source_shard=dict(spec),
                source_compressed_and_uncompressed_SHA_verified=True, producer_evidence=evidence)
            records.append(record)
            del arrays, evidence
            if len(records) == 1 or len(records) % 32 == 0:
                checkpoint_progress()
        if counts['rows'] != 24755 or len(records) != 774 or counts['episode_starts'] != 1 or counts['terminals'] != 1:
            raise ValueError('Complete natural episode contract failed')
        output_index = dict(schema=SCHEMA, status='completed_one_original_rule_episode',
            identity_sha256=sha(a.output / 'identity.json'), rows=24755, complete_episode=True,
            source_index_sha256=INDEX_SHA, source_behaviour=index['metadata']['teacher_revision'],
            stage_counts=dict(by_stage), counts=dict(counts), producer_abstentions=dict(abstentions),
            label_packets=records, labelled_geometry_is_native_counterfactual_truth=False,
            fact_outcomes_created=0, source_arrays_modified=False,
            source_maps_objects_duplicated_to_output=False, Torch_only_for_CPU_geometry_tensor_API=True,
            perception_or_neural_teacher_or_brain_constructed=False, neural_forwards=0, CUDA_calls=0,
            native_actions=0, parameter_updates=0, source_runtime_modified=False,
            process=identity['process'], resources=dict(threads=2, affinity=process.cpu_affinity(), below_normal=True,
                CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], RSS_bytes=process.memory_info().rss),
            timings=dict(total_CPU_wall_seconds=time.monotonic()-started, source_seconds=source_seconds,
                geometry_seconds=geometry_seconds, output_write_seconds=write_seconds))
        atomic_json(a.output / 'index.json', output_index)
        commit = dict(schema=SCHEMA, index_sha256=sha(a.output / 'index.json'),
            identity_sha256=sha(a.output / 'identity.json'), rows=24755, packets=774, complete_episode=True)
        atomic_json(a.output / 'COMMITTED.json', commit)
        checkpoint_progress('completed_one_original_rule_episode')
        receipt = dict(schema=SCHEMA, status=progress['status'], output=str(a.output), commit=commit,
            commit_sha256=sha(a.output / 'COMMITTED.json'), counts=dict(counts), stage_counts=dict(by_stage),
            producer_abstentions=dict(abstentions), timings=output_index['timings'],
            script_sha256=sha(Path(__file__)), source_index_sha256=INDEX_SHA,
            resources=output_index['resources'], actual_neural_forwards=0, actual_native_actions=0,
            actual_parameter_updates=0)
        atomic_json(a.receipt / 'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        failure = dict(schema=SCHEMA, status='failed_preserved_partial_no_retry', error=repr(error),
            rows=int(counts['rows']), completed_packets=len(records), elapsed_seconds=time.monotonic()-started,
            output=str(a.output), COMMITTED=False, source_unmodified=True, source_runtime_unmodified=True)
        atomic_json(a.output / 'failure.json', failure)
        atomic_json(a.receipt / 'failure.json', failure)
        print(json.dumps(failure), flush=True)
        raise


if __name__ == '__main__':
    main()
