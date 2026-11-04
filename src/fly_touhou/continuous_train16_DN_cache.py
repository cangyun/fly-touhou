"""Extend the sealed final-theta eight-history DN cache, without replaying it.

Import and inspection use JSON only. Explicit execution advances ten NEW
row0-origin histories at physical width16 and materializes a separate18-episode
cache. The old cache and all source labels/encoded observations stay read-only.
"""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'continuous-finaltheta-train16-DN-cache-v1'
NEW_EPISODES = tuple(f'world{i:03d}' for i in range(9, 19))
OLD_EPISODES = tuple(f'world{i:03d}' for i in (*range(6), 7, 8))
IDENTITY_KEYS = ('episode', 'role', 'rows', 'source_index_sha256',
                 'label_index_sha256', 'visual_index_sha256')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path):
    from .thrl_study_io import read_shared_json
    return read_shared_json(Path(path))


def inspect_extension(manifest, old_cache, *, manifest_sha256,
                      old_commit_sha256, source_sha256, DN_source_identity):
    """Only committed JSON metadata, never DN/NPZ/model arrays."""
    from .continuous_rule_train16_pretraining import inspect_train16
    data = inspect_train16(manifest, expected_sha256=manifest_sha256)
    old_cache = Path(old_cache)
    commit_path = old_cache / 'COMMITTED.json'
    if sha(commit_path) != old_commit_sha256:
        raise ValueError('Explicit old-cache COMMITTED identity differs')
    commit = _json(commit_path)
    index = _json(old_cache / 'index.json')
    if (commit.get('complete_all8') is not True
            or sha(old_cache / 'index.json') != commit['index_sha256']
            or index['data_binding'] != commit['data_binding']
            or index['source_checkpoint_sha256'] != source_sha256
            or index['DN_source_identity'] != DN_source_identity
            or commit['DN_source_identity'] != DN_source_identity
            or index['rows'] != 200538 or index['DN_shape'] != [200538, 1299]
            or index['core_K'] != 8 or not index['true_row0_complete_histories']):
        raise ValueError('Complete fixed-theta original8 cache identity differs')
    old_specs = {s['episode']: s for s in data['specs'] if s['episode'] in OLD_EPISODES}
    old_bound = commit['data_binding']['episodes']
    if [s['episode'] for s in old_bound] != list(OLD_EPISODES):
        raise ValueError('Exactly the original8 reused histories required')
    cursor = 0
    if len(index['episode_offsets']) != 8:
        raise ValueError('Exactly eight original row ranges required')
    for bound, offset in zip(old_bound, index['episode_offsets']):
        spec = old_specs[bound['episode']]
        if (any(bound[k] != spec[k] for k in IDENTITY_KEYS)
                or offset != dict(episode=spec['episode'], role=spec['role'],
                                  offset=cursor, rows=spec['rows'])):
            raise ValueError('Reused original episode source/label/visual/range differs')
        cursor += spec['rows']
    new = [s for s in data['specs'] if s['episode'] in NEW_EPISODES]
    if [s['episode'] for s in new] != list(NEW_EPISODES):
        raise ValueError('Exactly world009..018 are new core histories')
    return dict(schema=SCHEMA, ready=data['ready'], data=data,
        old_cache=str(old_cache.resolve()), old_commit_sha256=old_commit_sha256,
        old_commit=commit, old_index=index, DN_source_identity=DN_source_identity,
        source_sha256=source_sha256, new_rows=sum(s['rows'] for s in new),
        reused_rows=cursor, total_rows=data['train_rows'] + data['held_rows'],
        old_payloads_checked=False, array_payloads_read=0)


def _label_arrays(total):
    import numpy as np
    fields = dict(features=np.empty((total, 25), np.float32),
        margin4=np.empty((total, 18), np.float32), trusted4=np.empty(total, bool),
        margins=np.zeros((total, 3, 18), np.float32),
        trusted_horizons=np.zeros((total, 3), bool),
        unprotected=np.empty(total, bool), actions=np.empty(total, np.int64),
        outcomes=np.full(total, -1, np.int8), source_row=np.empty(total, np.int64),
        stage=np.empty(total, np.int16), episode_index=np.empty(total, np.int8),
        train_role=np.empty(total, bool), source_rule_targets=np.zeros((total, 19), np.float32),
        source_rule_weights=np.zeros(total, np.float32), source_rule_available=np.zeros(total, bool),
        source_part=np.empty(total, np.int32), source_local_row=np.empty(total, np.int16),
        episode_starts=np.empty(total, bool), dones=np.empty(total, bool),
        native_episodes=np.empty(total, np.int64), native_seeds=np.empty(total, np.int64),
        before_frames=np.empty(total, np.int64), after_observation_frames=np.empty(total, np.int64),
        stage_frame=np.empty(total, np.int64), before_episode_frames=np.empty(total, np.int64),
        after_episode_frames=np.empty(total, np.int64), elapsed_native_frames=np.empty(total, np.int64),
        four_frame_transition_mask=np.empty(total, bool), fact_known_mask=np.zeros(total, bool))
    return fields


def _put_labels(labels, reader, start, block, offset, episode_index):
    import numpy as np
    n = len(block['actions']); sl = slice(offset + start, offset + start + n)
    direct = ('features', 'actions', 'outcomes', 'stage', 'episode_starts', 'dones',
              'native_episodes', 'native_seeds', 'before_frames', 'after_observation_frames',
              'stage_frame', 'before_episode_frames', 'after_episode_frames')
    for key in direct:
        labels[key][sl] = block[key]
    labels['margin4'][sl] = block['signed_margin4']
    labels['trusted4'][sl] = block['trusted4']
    labels['margins'][sl] = block['margins']
    labels['trusted_horizons'][sl] = block['trusted_horizons']
    labels['unprotected'][sl] = block['normal_unprotected_mask']
    labels['source_row'][sl] = np.arange(start, start + n)
    labels['episode_index'][sl] = episode_index
    labels['train_role'][sl] = reader.spec['role'] == 'train'
    available = reader.spec['original_rule_available']
    labels['source_rule_available'][sl] = available
    if available:
        labels['source_rule_targets'][sl] = block['source_rule_targets']
        labels['source_rule_weights'][sl] = block['source_teacher_guidance_weight']
    elapsed = block['after_observation_frames'] - block['before_frames']
    labels['elapsed_native_frames'][sl] = elapsed
    labels['four_frame_transition_mask'][sl] = elapsed == 4
    row = start
    while row < start + n:
        part = bisect.bisect_right(reader.starts, row) - 1
        packet = reader._packet(part)
        local = row - reader.parts[part]['start']
        take = min(start + n - row, reader.parts[part]['rows'] - local)
        target = slice(offset + row, offset + row + take)
        for key in ('source_part', 'source_local_row'):
            labels[key][target] = packet[key][local:local + take]
        row += take
    return sl


def build_DN_extension(model, prepared, output, *, check_stop=lambda: None,
                       progress=lambda row: None):
    """Frozen core-only execution; no actor/critic/perception/teacher/optimizer call."""
    import numpy as np
    import torch
    from .continuous_connectome import ContinuousConnectomeState
    from .continuous_fullhistory_head_calibration import DN_source_identity
    from .continuous_rule_train16_pretraining import Train16RuleEpisodeReader
    from .thrl_durable_records import write_json

    output = Path(output)
    if output.exists():
        raise FileExistsError('Fresh extension root required; preserve incomplete data, no implicit resume')
    if not prepared['ready'] or model.core.K != 8:
        raise ValueError('Complete18 data and original K8 required')
    body = DN_source_identity(model)
    if body != prepared['DN_source_identity']:
        raise ValueError('Source brain/input/visual identity differs from reused original8')
    old = Path(prepared['old_cache']); commit = prepared['old_commit']
    for name, key in (('DN.npy', 'DN_sha256'), ('labels.npz', 'labels_sha256')):
        check_stop()
        if sha(old / name) != commit[key]:
            raise ValueError('Committed original8 payload bytes differ')
    old_DN = np.load(old / 'DN.npy', mmap_mode='r', allow_pickle=False)
    if old_DN.shape != tuple(prepared['old_index']['DN_shape']) or old_DN.dtype != np.float32:
        raise ValueError('Original8 DN shape/dtype differs')
    with np.load(old / 'labels.npz', allow_pickle=False) as z:
        old_labels = {k: z[k] for k in z.files}
    readers = [Train16RuleEpisodeReader(s, check_stop=check_stop) for s in prepared['data']['specs']]
    for r in readers:
        if (r.identity['actual_perception_source'] != model.sensory_adapter.perception.source_provenance()
                or r.identity['actual_statistics'] != model.sensory_adapter.statistics_provenance):
            raise ValueError('Actual frozen perception/statistics differs')
    offsets = {}; cursor = 0
    for r in readers:
        offsets[r.spec['episode']] = cursor; cursor += r.rows
    total = cursor
    if total != prepared['total_rows']:
        raise ValueError('Complete18 row count differs')
    output.mkdir(parents=True)
    DN = np.lib.format.open_memmap(output / 'DN.npy', mode='w+', dtype=np.float32,
        shape=(total, len(model.core.output_indices)))
    labels = _label_arrays(total)
    counts = dict(reused_real_rows=0, new_real_rows=0, core_batch_calls=0,
        physical_lane_forwards=0, padding_lane_forwards=0, core_updates=0,
        optimizer_instances=0, optimizer_updates=0, teacher_queries=0, native_actions=0,
        perception_forwards=0, old_history_core_forwards=0)
    versions = tuple(p._version for p in model.core.parameters())
    cursors = [0] * 10
    model.eval()
    try:
        old_offsets = {e['episode']: e for e in prepared['old_index']['episode_offsets']}
        for ei, r in enumerate(readers):
            name = r.spec['episode']
            if name not in old_offsets:
                continue
            source_offset = old_offsets[name]['offset']; target_offset = offsets[name]
            for start in range(0, r.rows, 16):
                check_stop(); block = r.fetch(start, 16); n = len(block['actions'])
                src = slice(source_offset + start, source_offset + start + n)
                sl = _put_labels(labels, r, start, block, target_offset, ei)
                for key in old_labels:
                    if key == 'episode_index':
                        continue  # original8 index changes only for held7/8 after insertion
                    if key not in labels or not np.array_equal(labels[key][sl], old_labels[key][src]):
                        raise ValueError('Original8 cached label differs from same immutable source: ' + key)
                values = old_DN[src]
                if not np.isfinite(values).all():
                    raise FloatingPointError('Original8 DN payload has nonfinite values')
                DN[sl] = values; counts['reused_real_rows'] += n
            progress(dict(phase='CPU_reuse_original8', episode=name, counts=copy.deepcopy(counts)))
        del old_DN, old_labels
        new = [(ei, r) for ei, r in enumerate(readers) if r.spec['episode'] in NEW_EPISODES]
        if len(new) != 10 or tuple(r.spec['episode'] for _, r in new) != NEW_EPISODES:
            raise ValueError('Exactly ten NEW source histories required')
        activity = model.initial_state(16).activity
        with torch.no_grad():
            while any(c < r.rows for c, (_, r) in zip(cursors, new)):
                check_stop()
                blocks = [r.fetch(c, 16) for c, (_, r) in zip(cursors, new)]
                used = [0 if b is None else len(b['actions']) for b in blocks]
                length = max(used)
                encoded = np.zeros((length, 16, 384), np.float32)
                starts = np.zeros((length, 16), bool); valid = np.zeros((length, 16), bool)
                for lane, (block, n) in enumerate(zip(blocks, used)):
                    if n:
                        encoded[:n, lane] = block['encoded_visual']
                        starts[:n, lane] = block['episode_starts']; valid[:n, lane] = True
                z = torch.from_numpy(encoded).to(model.device)
                resets = torch.from_numpy(starts).to(model.device)
                active = torch.from_numpy(valid).to(model.device)
                window_DN = []
                for t in range(length):
                    check_stop()
                    activity = torch.where((resets[t] & active[t])[:, None],
                                           torch.zeros_like(activity), activity)
                    dn, following = model.core(z[t], ContinuousConnectomeState(activity))
                    counts['core_batch_calls'] += 1; counts['physical_lane_forwards'] += 16
                    counts['padding_lane_forwards'] += 16 - int(valid[t].sum())
                    if not bool(torch.isfinite(following.activity).all()) or not bool(torch.isfinite(dn).all()):
                        raise FloatingPointError('Nonfinite new fixed-theta continuous history')
                    activity = torch.where(active[t, :, None], following.activity, activity)
                    window_DN.append(dn[:10])
                copied = torch.stack(window_DN).cpu().numpy()
                for lane, ((ei, r), start, n, block) in enumerate(zip(new, cursors, used, blocks)):
                    if not n:
                        continue
                    sl = _put_labels(labels, r, start, block, offsets[r.spec['episode']], ei)
                    DN[sl] = copied[:n, lane]
                cursors = [c + n for c, n in zip(cursors, used)]
                counts['new_real_rows'] += sum(used)
                if counts['core_batch_calls'] % 512 == 0:
                    DN.flush()
                    progress(dict(phase='new10_complete_histories', cursors=list(cursors), counts=copy.deepcopy(counts)))
        if versions != tuple(p._version for p in model.core.parameters()) or DN_source_identity(model) != body:
            raise RuntimeError('Fixed DN source parameters changed')
        DN.flush()
        with (output / 'labels.npz.tmp').open('wb') as stream:
            np.savez(stream, **labels); stream.flush(); os.fsync(stream.fileno())
        os.replace(output / 'labels.npz.tmp', output / 'labels.npz')
        binding = dict(manifest_sha256=prepared['data']['manifest_sha256'],
            dataset_commit_sha256=prepared['data']['dataset_commit_sha256'],
            episodes=[{k: s[k] for k in (*IDENTITY_KEYS, 'native_seed',
                'retained_horizons_native_frames', 'original_rule_available')} for s in prepared['data']['specs']],
            original8=dict(root=str(old), COMMITTED_sha256=prepared['old_commit_sha256'],
                index_sha256=commit['index_sha256'], DN_sha256=commit['DN_sha256'], labels_sha256=commit['labels_sha256']))
        result = dict(schema=SCHEMA, status='complete_finaltheta_train16_held2_DN', rows=total,
            train_rows=prepared['data']['train_rows'], held_rows=prepared['data']['held_rows'],
            source_checkpoint_sha256=prepared['source_sha256'], DN_source_identity=body,
            DN_shape=list(DN.shape), DN_bytes=int(DN.nbytes), data_binding=binding, counts=counts,
            episode_offsets=[dict(episode=r.spec['episode'], role=r.spec['role'], offset=offsets[r.spec['episode']],
                rows=r.rows, DN_origin='reused_original8' if r.spec['episode'] in OLD_EPISODES else 'new_fixed_theta_core',
                retained_horizons_native_frames=r.spec['retained_horizons_native_frames']) for r in readers],
            graph=model.core.graph_provenance, core_K=model.core.K, physical_width=16, new_real_lanes=10,
            true_row0_complete_histories=True, Miss_or_stage_resets=0, full_activity_per_frame_stored=False,
            old8_payloads_preserved=True, old8_future_trust=False,
            future_margins_scope='original current-motion geometry proxies;not native counterfactual truth',
            original_world000_rule_soft_targets_absent=True,
            reader_costs={r.spec['episode']:r.paid for r in readers})
        write_json(output / 'index.json', result)
        check_stop()
        write_json(output / 'COMMITTED.json', dict(schema=SCHEMA, complete_all18=True,
            rows=total, train_episodes=16, held_episodes=2, DN_source_identity=body,
            index_sha256=sha(output / 'index.json'), DN_sha256=sha(output / 'DN.npy'),
            labels_sha256=sha(output / 'labels.npz'), data_binding=binding))
        return result
    except BaseException as error:
        error.DN_cache_progress = dict(counts=copy.deepcopy(counts), cursors=list(cursors),
                                      partial_output_preserved=True)
        raise
    finally:
        DN.flush(); del DN
