"""Prepare a complete rule-episode visual384 cache; execution is explicit CPU/CUDA.

The default path reads only the original and signed-label index JSON files.
It imports neither NumPy nor Torch and never creates the data output root.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026'
SOURCE_ROOT = Path('./outputs/engineering/B1-autonomous-dodge-20261024/teacher-full-natural30-v1/trajectories')
D95 = Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
STATISTICS = Path('./outputs/distillation/D95-frozen-visual384-prefix2048-v1/train-statistics.npz')
STATISTICS_SHA = 'c7041498d7d5d4aaa3b68be36c4b96dbd4adf535a2146a421a3c4876d2f16627'
V5 = 'semantic-separated-observed-laser-segments-v5'
SCHEMA = 'rule-complete-V5-fixed-standardized-visual384-history-v1'
LABEL_SCHEMAS = {'rule-world000-original-V5-signed4f-proxy-cache-v1',
    'rule-original-V5-signed4f-proxy-cache-v1'}
META = ('actions', 'dones', 'episode_starts', 'native_episodes', 'native_seeds',
    'before_frames', 'after_observation_frames', 'stage', 'stage_frame',
    'before_episode_frames', 'after_episode_frames')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_metadata(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def cpu_controls(execute_cpu=False):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1' if execute_cpu else '2'
    import psutil
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if execute_cpu:
        if 18 not in p.cpu_affinity():
            raise ValueError('CPU encoding requires caller-allowed CPU18')
        p.cpu_affinity([18])
    else:
        p.cpu_affinity(p.cpu_affinity()[-2:])
    return p


def build_plan(a):
    if not re.fullmatch(r'world\d{3}', a.episode) or not 0 <= int(a.episode[5:]) < 30:
        raise ValueError('Explicit existing world000..029 source episode required')
    if not a.cuda_visible_device.isdecimal():
        raise ValueError('One explicit numeric CUDA_VISIBLE_DEVICES index is required')
    if a.max_seconds is not None and (not math.isfinite(a.max_seconds) or a.max_seconds <= 0):
        raise ValueError('Wall budget must be positive finite seconds')
    if (a.execute_cuda or a.execute_cpu) and a.max_seconds is None:
        raise ValueError('Explicit execution also requires --max-seconds')
    source = SOURCE_ROOT / a.episode
    raw, raw_sha = read_metadata(source / 'index.json')
    labels, labels_sha = read_metadata(a.label_cache / 'index.json')
    if (raw.get('schema') != 'B1-lossless-natural-teacher-trajectory-shards-v1'
            or raw.get('status') != 'completed' or raw.get('natural_episode_complete') is not True):
        raise ValueError('Complete original numeric episode archive required')
    if (raw['metadata'].get('input_schema') != V5
            or not all(s in raw['native_episode_summary'].get('stage_history', []) for s in range(1, 7))):
        raise ValueError('Same V5 complete six-stage source required')
    for key, shape in (('features', [25]), ('maps', [16, 92, 96]), ('objects', [96, 29])):
        if raw['fields'][key] != dict(dtype='<f4', shape=shape):
            raise ValueError('Original FP32 modality differs; never pad or convert schema: ' + key)
    if (labels.get('schema') not in LABEL_SCHEMAS or labels.get('complete_episode') is not True
            or labels.get('status') != 'completed_one_original_rule_episode'
            or labels.get('source_index_sha256') != raw_sha or labels.get('rows') != raw['rows']):
        raise ValueError('Complete signed4f labels must bind this exact source episode')
    if not (a.label_cache / 'COMMITTED.json').is_file():
        raise ValueError('No complete label cache commit marker')
    source_parts, target_parts = raw['shards'], labels['label_packets']
    if len(source_parts) != len(target_parts):
        raise ValueError('Source/label packet count differs')
    cursor = 0
    for src, target in zip(source_parts, target_parts):
        if (src['start'] != cursor or not 1 <= src['rows'] <= 32
                or target['start'] != src['start'] or target['rows'] != src['rows']
                or target.get('source_shard') != src
                or Path(src['name']).name != src['name'] or Path(target['name']).name != target['name']):
            raise ValueError('Packet source identity/row order differs')
        cursor += src['rows']
    if cursor != raw['rows']:
        raise ValueError('Incomplete indexed history')
    if not a.checkpoint.is_file() or not a.statistics.is_file():
        raise ValueError('Explicit original checkpoint and fixed statistics paths must exist')
    if a.output.resolve() in (source.resolve(), a.label_cache.resolve()):
        raise ValueError('Output cannot replace an original source or label cache')
    source_pins = {name: file_sha(ROOT / 'src/fly_touhou' / name) for name in (
        'fly_frozen_v5_perception.py', 'fly_visual_distillation.py', 'thrl_teacher_archive.py')}
    plan = dict(schema=SCHEMA, status='prepared_not_executed', episode=a.episode,
        source_root=str(source), source_index_sha256=raw_sha,
        source_behaviour=raw['metadata']['teacher_revision'], source_native_seed=raw['metadata']['native_seed'],
        source_stage_history=raw['native_episode_summary']['stage_history'], source_outcome=raw['native_episode_summary']['outcome'],
        label_cache=str(a.label_cache), label_index_sha256=labels_sha,
        label_COMMITTED=str(a.label_cache / 'COMMITTED.json'), label_identity_sha256=labels['identity_sha256'],
        rows=raw['rows'], packets=len(source_parts), input_schema=V5, output_root=str(a.output),
        checkpoint=str(a.checkpoint), checkpoint_sha256=D95_SHA,
        statistics=str(a.statistics), statistics_sha256=STATISTICS_SHA,
        source_module_sha256=source_pins,
        frontend='existing FrozenV5Perception.from_d95_checkpoint + load_visual_statistics + FixedStandardizedVisual',
        formula='(FrozenV5Perception(originalFP32V5)-original_fixed_train_mean)/original_fixed_train_std',
        output=dict(dtype='float32', shape=[raw['rows'], 384], uncompressed_visual_bytes=raw['rows'] * 384 * 4,
            packet_rows='original source rows<=32, including unchanged tail length',
            payload_fields=['encoded_visual', 'source_row', 'source_part', 'source_local_row', *META],
            signed_targets='immutable source label packet reference+SHA; no copied or relabeled targets'),
        history='all source0->terminal observations preserved; stateless perception has no reset. Later neural replay starts only at row0 and must retain own activity across Miss/stage/packets.',
        clocks='original before/after clocks; no interpolation or assumption elapsed==4',
        runtime=dict(device='cpu' if a.execute_cpu else 'cuda:0 mapped from explicit visible device',
            backend='CPU' if a.execute_cpu else 'CUDA',
            CUDA_VISIBLE_DEVICES='-1' if a.execute_cpu else a.cuda_visible_device,
            CPU_threads=1 if a.execute_cpu else 2,
            CPU_affinity=[18] if a.execute_cpu else 'last2 caller-allowed CPUs', priority='BelowNormal',
            FP32=True, autocast=False, TF32=False, memory_cap=None, max_seconds=a.max_seconds),
        prepare_mode=dict(existing_cache_JSON_files_read=2, array_payloads_read=0, checkpoint_payload_read=False,
            Torch_imported=False, NumPy_imported=False, perception_forwards=0, brain_forwards=0,
            neural_teacher_queries=0, CUDA_calls=0, native_actions=0, parameter_updates=0,
            data_output_root_created=False),
        limits=['Labels are trusted current-motion geometric proxies, not native counterfactual truth.',
            'Complete encoded history is not a brain-state or optimizer numerical-resume checkpoint.',
            'No encoding or CPU/CUDA equivalence measurement has run in preparation mode.',
            'Same FP32 inputs, weights and statistics do not imply CPU/CUDA bitwise equality.'])
    return plan, raw, labels


def execute_cuda(a, plan, raw_index, label_index, process):
    if a.output.exists():
        raise FileExistsError('Fresh output only; preserve incomplete or existing history')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.cuda_visible_device
    # All GPU/model imports and parameter reads occur only past --execute-cuda.
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from fly_touhou.fly_frozen_v5_perception import FrozenV5Perception
    from fly_touhou.fly_visual_distillation import FixedStandardizedVisual, load_visual_statistics
    from fly_touhou.thrl_teacher_archive import TeacherTrajectoryReader
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA execution has no visible CUDA device')
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()

    def check_stop():
        if time.monotonic() - started >= a.max_seconds:
            raise TimeoutError('Explicit finite encoding wall budget reached; no retry or invented terminal')

    commit, _ = read_metadata(a.label_cache / 'COMMITTED.json')
    if (commit['index_sha256'] != plan['label_index_sha256']
            or commit['identity_sha256'] != plan['label_identity_sha256']
            or commit['rows'] != plan['rows'] or commit['packets'] != plan['packets']
            or commit.get('complete_episode') is not True):
        raise ValueError('Label commit does not match inspected source/target history')
    if file_sha(a.label_cache / 'identity.json') != commit['identity_sha256']:
        raise ValueError('Signed label identity content differs')
    mean, std, provenance = load_visual_statistics(a.statistics)
    perception = FrozenV5Perception.from_d95_checkpoint(a.checkpoint, expected_sha256=D95_SHA)
    adapter = FixedStandardizedVisual(perception, mean, std, provenance).to('cuda:0').eval().requires_grad_(False)
    check_stop()
    source = Path(plan['source_root'])
    reader = TeacherTrajectoryReader(source, require_complete=True, cache_bytes=0)
    a.output.mkdir(parents=True)
    (a.output / 'visual').mkdir()
    identity = dict(plan, status='executing_explicit_CUDA',
        process=dict(pid=process.pid, create_time=process.create_time()),
        actual_perception_source=perception.source_provenance(), actual_statistics=provenance)
    atomic_json(a.output / 'identity.json', identity)
    records = []
    rows = forwards = starts = terminals = 0
    previous_after = previous_after_ep = None
    try:
        for number, (src, label) in enumerate(zip(raw_index['shards'], label_index['label_packets'])):
            check_stop()
            n = src['rows']
            block = reader.fetch(src['start'], src['start'] + n, fields=['features', 'maps', 'objects', *META])
            label_path = a.label_cache / 'labels' / label['name']
            if file_sha(label_path) != label['sha256']:
                raise ValueError('Immutable source label packet SHA differs')
            with np.load(label_path, allow_pickle=False) as z:
                if (not np.array_equal(z['features'], block['features'])
                        or not np.array_equal(z['source_row'], np.arange(src['start'], src['start'] + n))
                        or any(not np.array_equal(z[name], block[name]) for name in META)):
                    raise ValueError('Label/raw observation row, clocks, seed, action or reset differ')
            if (not np.array_equal(block['before_frames'][1:], block['after_observation_frames'][:-1])
                    or not np.array_equal(block['before_episode_frames'][1:], block['after_episode_frames'][:-1])
                    or (previous_after is not None and (int(block['before_frames'][0]) != previous_after
                        or int(block['before_episode_frames'][0]) != previous_after_ep))):
                raise ValueError('True complete observation clock chain differs; never bridge gaps')
            source_rows = np.arange(src['start'], src['start'] + n, dtype=np.int64)
            if (not np.array_equal(block['episode_starts'], source_rows == 0)
                    or not np.array_equal(block['dones'], source_rows == plan['rows'] - 1)
                    or not np.all(block['native_seeds'] == plan['source_native_seed'])):
                raise ValueError('Natural initial/terminal episode identity differs')
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                modalities = [torch.as_tensor(block[k], device='cuda:0', dtype=torch.float32)
                    for k in ('features', 'maps', 'objects')]
                encoded = adapter(*modalities).cpu().numpy()
            forwards += n
            if encoded.shape != (n, 384) or encoded.dtype != np.float32 or not np.isfinite(encoded).all():
                raise ValueError('Exact finite FP32 visual384 output required')
            arrays = dict(encoded_visual=encoded, source_row=source_rows,
                source_part=np.full(n, number, np.int32), source_local_row=np.arange(n, dtype=np.int16),
                **{k:block[k] for k in META})
            check_stop()
            payload_io = io.BytesIO()
            np.savez_compressed(payload_io, **arrays)
            payload = payload_io.getvalue()
            name = f'visual-{number:06d}.npz'
            temporary = a.output / 'visual' / (name + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, a.output / 'visual' / name)
            records.append(dict(name=name, start=src['start'], rows=n, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), encoded_raw_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
                original_source_shard=dict(src), label_reference=dict(root=str(a.label_cache), packet=dict(label))))
            rows += n
            starts += int(block['episode_starts'].sum())
            terminals += int(block['dones'].sum())
            previous_after = int(block['after_observation_frames'][-1])
            previous_after_ep = int(block['after_episode_frames'][-1])
            del block, modalities, encoded, arrays, payload_io, payload
            if number == 0 or len(records) % 32 == 0:
                progress = dict(schema=SCHEMA, status='encoding', episode=a.episode, rows=rows,
                    packets=len(records), perception_rows=forwards, elapsed_seconds=time.monotonic()-started,
                    brain_forwards=0, neural_teacher_queries=0, native_actions=0)
                atomic_json(a.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if rows != plan['rows'] or starts != 1 or terminals != 1:
            raise ValueError('Full natural episode count differs')
        result = dict(schema=SCHEMA, status='completed_one_full_rule_visual_history', episode=a.episode,
            identity_sha256=file_sha(a.output / 'identity.json'), rows=rows, complete_episode=True,
            packets=records, source_index_sha256=plan['source_index_sha256'], label_index_sha256=plan['label_index_sha256'],
            process=identity['process'], perception_rows=forwards, brain_forwards=0, neural_teacher_queries=0,
            native_actions=0, parameter_updates=0, original_maps_or_objects_copied_to_output=False,
            actor_critic_GRU_risk_or_fullbrain_constructed=False, fixed_statistics_refit=False,
            clocks='all source rows and actual native before/after clocks, including elapsed!=4',
            elapsed_seconds=time.monotonic()-started, script_sha256=file_sha(Path(__file__)),
            CUDA=dict(name=torch.cuda.get_device_name(0), TF32=False, autocast=False,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0)),
            CPU=dict(threads=2, affinity=process.cpu_affinity(), below_normal=True))
        atomic_json(a.output / 'index.json', result)
        commit = dict(schema=SCHEMA, index_sha256=file_sha(a.output / 'index.json'),
            identity_sha256=result['identity_sha256'], rows=rows, packets=len(records), complete_episode=True)
        atomic_json(a.output / 'COMMITTED.json', commit)
        atomic_json(a.preparation / 'execution-receipt.json', dict(result={k:v for k,v in result.items() if k!='packets'},
            output=str(a.output), commit=commit, COMMITTED_sha256=file_sha(a.output / 'COMMITTED.json')))
        print(json.dumps(dict(status=result['status'], rows=rows, packets=len(records), output=str(a.output), commit=commit)), flush=True)
    except Exception as error:
        failure = dict(schema=SCHEMA, status='failed_preserved_partial_no_retry', error=repr(error),
            rows_committed=rows, packets=len(records), actual_perception_rows=forwards,
            elapsed_seconds=time.monotonic()-started, process=identity['process'],
            brain_forwards=0, neural_teacher_queries=0, native_actions=0, parameter_updates=0)
        atomic_json(a.output / 'failure.json', failure)
        atomic_json(a.preparation / 'execution-failure.json', failure)
        raise


def execute_cpu(a, plan, raw_index, label_index, process):
    if a.output.exists():
        raise FileExistsError('Fresh output only; preserve incomplete or existing history')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # CPU-only perception imports occur only past --execute-cpu, after CVD=-1.
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from fly_touhou.fly_frozen_v5_perception import FrozenV5Perception
    from fly_touhou.fly_visual_distillation import FixedStandardizedVisual, load_visual_statistics
    from fly_touhou.thrl_teacher_archive import TeacherTrajectoryReader
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()

    def check_stop():
        if time.monotonic() - started >= a.max_seconds:
            raise TimeoutError('Explicit finite encoding wall budget reached; no retry or invented terminal')

    commit, _ = read_metadata(a.label_cache / 'COMMITTED.json')
    if (commit['index_sha256'] != plan['label_index_sha256']
            or commit['identity_sha256'] != plan['label_identity_sha256']
            or commit['rows'] != plan['rows'] or commit['packets'] != plan['packets']
            or commit.get('complete_episode') is not True):
        raise ValueError('Label commit does not match inspected source/target history')
    if file_sha(a.label_cache / 'identity.json') != commit['identity_sha256']:
        raise ValueError('Signed label identity content differs')
    mean, std, provenance = load_visual_statistics(a.statistics)
    perception = FrozenV5Perception.from_d95_checkpoint(a.checkpoint, expected_sha256=D95_SHA)
    adapter = FixedStandardizedVisual(perception, mean, std, provenance).to('cpu').eval().requires_grad_(False)
    check_stop()
    source = Path(plan['source_root'])
    reader = TeacherTrajectoryReader(source, require_complete=True, cache_bytes=0)
    a.output.mkdir(parents=True)
    (a.output / 'visual').mkdir()
    identity = dict(plan, status='executing_explicit_CPU', backend='CPU', cross_CPU_CUDA_bitwise_equivalence_claimed=False,
        process=dict(pid=process.pid, create_time=process.create_time()),
        actual_perception_source=perception.source_provenance(), actual_statistics=provenance)
    atomic_json(a.output / 'identity.json', identity)
    records = []
    rows = forwards = starts = terminals = 0
    previous_after = previous_after_ep = None
    source_seconds = perception_seconds = write_seconds = 0.
    try:
        for number, (src, label) in enumerate(zip(raw_index['shards'], label_index['label_packets'])):
            check_stop()
            packet_started = time.monotonic()
            n = src['rows']
            block = reader.fetch(src['start'], src['start'] + n, fields=['features', 'maps', 'objects', *META])
            label_path = a.label_cache / 'labels' / label['name']
            if file_sha(label_path) != label['sha256']:
                raise ValueError('Immutable source label packet SHA differs')
            with np.load(label_path, allow_pickle=False) as z:
                if (not np.array_equal(z['features'], block['features'])
                        or not np.array_equal(z['source_row'], np.arange(src['start'], src['start'] + n))
                        or any(not np.array_equal(z[name], block[name]) for name in META)):
                    raise ValueError('Label/raw observation row, clocks, seed, action or reset differ')
            if (not np.array_equal(block['before_frames'][1:], block['after_observation_frames'][:-1])
                    or not np.array_equal(block['before_episode_frames'][1:], block['after_episode_frames'][:-1])
                    or (previous_after is not None and (int(block['before_frames'][0]) != previous_after
                        or int(block['before_episode_frames'][0]) != previous_after_ep))):
                raise ValueError('True complete observation clock chain differs; never bridge gaps')
            source_rows = np.arange(src['start'], src['start'] + n, dtype=np.int64)
            if (not np.array_equal(block['episode_starts'], source_rows == 0)
                    or not np.array_equal(block['dones'], source_rows == plan['rows'] - 1)
                    or not np.all(block['native_seeds'] == plan['source_native_seed'])):
                raise ValueError('Natural initial/terminal episode identity differs')
            perception_started = time.monotonic()
            source_seconds += perception_started - packet_started
            with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
                modalities = [torch.as_tensor(block[k], device='cpu', dtype=torch.float32)
                    for k in ('features', 'maps', 'objects')]
                encoded = adapter(*modalities).cpu().numpy()
            perception_seconds += time.monotonic() - perception_started
            forwards += n
            if encoded.shape != (n, 384) or encoded.dtype != np.float32 or not np.isfinite(encoded).all():
                raise ValueError('Exact finite FP32 visual384 output required')
            arrays = dict(encoded_visual=encoded, source_row=source_rows,
                source_part=np.full(n, number, np.int32), source_local_row=np.arange(n, dtype=np.int16),
                **{k:block[k] for k in META})
            check_stop()
            write_started = time.monotonic()
            payload_io = io.BytesIO()
            np.savez_compressed(payload_io, **arrays)
            payload = payload_io.getvalue()
            name = f'visual-{number:06d}.npz'
            temporary = a.output / 'visual' / (name + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, a.output / 'visual' / name)
            write_seconds += time.monotonic() - write_started
            records.append(dict(name=name, start=src['start'], rows=n, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), encoded_raw_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
                original_source_shard=dict(src), label_reference=dict(root=str(a.label_cache), packet=dict(label))))
            rows += n
            starts += int(block['episode_starts'].sum())
            terminals += int(block['dones'].sum())
            previous_after = int(block['after_observation_frames'][-1])
            previous_after_ep = int(block['after_episode_frames'][-1])
            del block, modalities, encoded, arrays, payload_io, payload
            if number == 0 or len(records) % 32 == 0:
                progress = dict(schema=SCHEMA, status='encoding', episode=a.episode, rows=rows,
                    packets=len(records), perception_rows=forwards, elapsed_seconds=time.monotonic()-started,
                    brain_forwards=0, neural_teacher_queries=0, native_actions=0, backend='CPU',
                    source_seconds=source_seconds, perception_seconds=perception_seconds, write_seconds=write_seconds,
                    overall_rows_per_second=rows/(time.monotonic()-started))
                atomic_json(a.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if rows != plan['rows'] or starts != 1 or terminals != 1:
            raise ValueError('Full natural episode count differs')
        result = dict(schema=SCHEMA, status='completed_one_full_rule_visual_history', episode=a.episode,
            identity_sha256=file_sha(a.output / 'identity.json'), rows=rows, complete_episode=True,
            packets=records, source_index_sha256=plan['source_index_sha256'], label_index_sha256=plan['label_index_sha256'],
            process=identity['process'], perception_rows=forwards, brain_forwards=0, neural_teacher_queries=0,
            native_actions=0, parameter_updates=0, original_maps_or_objects_copied_to_output=False,
            actor_critic_GRU_risk_or_fullbrain_constructed=False, fixed_statistics_refit=False,
            clocks='all source rows and actual native before/after clocks, including elapsed!=4',
            elapsed_seconds=time.monotonic()-started, script_sha256=file_sha(Path(__file__)),
            backend='CPU', CUDA=None, CUDA_calls=0, cross_CPU_CUDA_bitwise_equivalence_claimed=False,
            timings=dict(source_seconds=source_seconds, perception_seconds=perception_seconds,
                output_write_seconds=write_seconds, overall_rows_per_second=rows/(time.monotonic()-started)),
            CPU=dict(threads=1, interop_threads=1, affinity=process.cpu_affinity(), below_normal=True,
                OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'], MKL_NUM_THREADS=os.environ['MKL_NUM_THREADS'],
                OPENBLAS_NUM_THREADS=os.environ['OPENBLAS_NUM_THREADS'], CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        atomic_json(a.output / 'index.json', result)
        commit = dict(schema=SCHEMA, index_sha256=file_sha(a.output / 'index.json'),
            identity_sha256=result['identity_sha256'], rows=rows, packets=len(records), complete_episode=True)
        atomic_json(a.output / 'COMMITTED.json', commit)
        atomic_json(a.preparation / 'execution-receipt.json', dict(result={k:v for k,v in result.items() if k!='packets'},
            output=str(a.output), commit=commit, COMMITTED_sha256=file_sha(a.output / 'COMMITTED.json')))
        print(json.dumps(dict(status=result['status'], rows=rows, packets=len(records), output=str(a.output), commit=commit)), flush=True)
    except Exception as error:
        failure = dict(schema=SCHEMA, status='failed_preserved_partial_no_retry', error=repr(error),
            rows_committed=rows, packets=len(records), actual_perception_rows=forwards,
            elapsed_seconds=time.monotonic()-started, process=identity['process'],
            brain_forwards=0, neural_teacher_queries=0, native_actions=0, parameter_updates=0)
        atomic_json(a.output / 'failure.json', failure)
        atomic_json(a.preparation / 'execution-failure.json', failure)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', default='world000')
    p.add_argument('--label-cache', type=Path, default=Path('./outputs/distillation/teacher-rule-world000-signed4f-cache-v1'))
    p.add_argument('--output', type=Path, default=Path('./outputs/distillation/teacher-rule-world000-visual384-cache-v1'))
    p.add_argument('--preparation', type=Path, default=M / 'teacher-rule-world000-visual384-preparation-v1')
    p.add_argument('--checkpoint', type=Path, default=D95)
    p.add_argument('--statistics', type=Path, default=STATISTICS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute-cuda', action='store_true')
    mode.add_argument('--execute-cpu', action='store_true')
    p.add_argument('--cuda-visible-device', default='0')
    p.add_argument('--max-seconds', type=float)
    a = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    process = cpu_controls(a.execute_cpu)
    plan, raw, labels = build_plan(a)
    a.preparation.mkdir(parents=True, exist_ok=True)
    atomic_json(a.preparation / 'prepared-plan.json', plan)
    if not (a.execute_cuda or a.execute_cpu):
        receipt = dict(schema=SCHEMA, status='prepared_not_executed', episode=a.episode,
            prepared_plan=str(a.preparation / 'prepared-plan.json'), prepared_plan_sha256=file_sha(a.preparation / 'prepared-plan.json'),
            script_sha256=file_sha(Path(__file__)), Torch_in_sys_modules='torch' in sys.modules,
            NumPy_in_sys_modules='numpy' in sys.modules, CUDA_calls=0, perception_forwards=0, brain_forwards=0,
            native_actions=0, parameter_updates=0, source_payloads_read=0, existing_cache_JSON_files_read=2,
            data_output_root_created=False, process=dict(pid=process.pid, create_time=process.create_time()),
            CPU=dict(threads=2, affinity=process.cpu_affinity(), below_normal=True))
        atomic_json(a.preparation / 'prepare-receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
        return plan
    if a.execute_cuda:
        return execute_cuda(a, plan, raw, labels, process)
    return execute_cpu(a, plan, raw, labels, process)


if __name__ == '__main__':
    main()
