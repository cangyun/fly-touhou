"""Lossless numeric extraction and rare-state coverage of training teacher data.

Only immutable training-partition trajectories are read. This CPU-only audit
never loads model weights, opens original-game worlds or changes supervision.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
import uuid
import zlib

FIELDS = ('features', 'actions', 'dones', 'episode_starts', 'native_episodes',
          'native_seeds', 'before_frames', 'after_observation_frames', 'stage',
          'stage_frame', 'before_episode_frames', 'after_episode_frames',
          'native_misses', 'native_bombs_used', 'observed_deathbomb_windows')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_fields(reader, part):
    import numpy as np
    spec = reader.index['shards'][part]
    payload = (reader.folder/spec['name']).read_bytes()
    if len(payload) != spec['bytes'] or hashlib.sha256(payload).hexdigest() != spec['sha256']:
        raise ValueError('Original teacher shard identity changed')
    raw = zlib.decompress(payload)
    if len(raw) != spec['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest() != spec['uncompressed_sha256']:
        raise ValueError('Lossless source contents changed')
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        if set(bundle.files) != set(reader.fields):
            raise ValueError('Source field schema differs')
        result = {name: bundle[name] for name in FIELDS}
    for name, value in result.items():
        dtype, shape = reader.fields[name]
        if value.dtype != dtype or value.shape != (spec['rows'], *shape):
            raise ValueError('Original numeric precision or field shape differs')
    return result


def coverage(arrays):
    import numpy as np
    from fly_touhou.thrl_deathbomb_learning import deathbomb_opportunity_mask
    from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
    f, actions, stages = arrays['features'], arrays['actions'], arrays['stage']
    state4 = f[:, 24] > .5
    eligible = deathbomb_opportunity_mask(f, SEPARATED_LASER_INPUT)
    enlarged = f[:, 13]*64 > 8
    chosen = actions == 18
    return dict(rows=len(f), teacher_Bomb_action_rows=int(chosen.sum()),
        state4_rows=int(state4.sum()), usable_deathbomb_rows=int(eligible.sum()),
        usable_deathbomb_teacher_Bomb_rows=int((eligible & chosen).sum()),
        usable_deathbomb_teacher_nonBomb_rows=int((eligible & ~chosen).sum()),
        enlarged_player_rows=int(enlarged.sum()),
        enlarged_usable_deathbomb_rows=int((enlarged & eligible).sum()),
        normal_usable_deathbomb_rows=int((~enlarged & eligible).sum()),
        usable_by_stage=dict(Counter(str(int(x)) for x in stages[eligible])),
        usable_by_remaining_frames=dict(Counter(str(int(round(float(x)*8))) for x in f[eligible, 19])),
        usable_enlarged_by_stage=dict(Counter(str(int(x)) for x in stages[eligible & enlarged])),
        per_stage_rows=dict(Counter(str(int(x)) for x in stages)),
        player_circle_range=[float(f[:, 13].min()*64), float(f[:, 13].max()*64)],
        critical_label_definition='native state4,positive Bomb stock,remaining2..8 frames',
        observations_are_not_independent_collision_events=True)


def run(plan, qualify_only=None):
    import numpy as np
    import psutil
    runtime = Path(plan['source_runtime'])
    if digest(__file__) != plan['worker_sha256'] or digest(runtime/'source_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Pinned diagnostic source differs')
    for name, expected in plan['source_modules'].items():
        if digest(runtime/'src/fly_touhou'/name) != expected:
            raise ValueError('Pinned diagnostic dependency changed')
    sys.path.insert(0, str(runtime/'src'))
    from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    if (digest(Path(plan['demonstrations'])/'demonstrations.json') != plan['demonstrations_sha256']
            or digest(plan['original_BC_split']) != plan['original_BC_split_sha256']):
        raise ValueError('Teacher archive or historical split changed')
    data = FullTeacherDataset(plan['demonstrations'], reader_cache_bytes=0)
    split = read(plan['original_BC_split'])
    if (sorted(split['train']+split['heldout']) != list(range(len(data.records)))
            or [data.records[i]['native_seed'] for i in split['train']] != split['train_native_seeds']
            or [data.records[i]['native_seed'] for i in split['heldout']] != split['heldout_native_seeds']):
        raise ValueError('Historical BC whole-trajectory split changed')
    if qualify_only:
        reader = data.records[0]['reader']
        first, second = load_fields(reader, 0), load_fields(reader, 1)
        expected = reader.fetch(0, len(first['actions'])+len(second['actions']), fields=FIELDS)
        if any(not np.array_equal(np.concatenate((first[name], second[name])), expected[name]) for name in FIELDS):
            raise ValueError('Selective numeric extraction differs from original archive reader')
        write(qualify_only, dict(status='passed', utc=utc_now(), original_shards_checked=2,
            all_selected_fields_bitwise_equal=True, original_numeric_precision_preserved=True,
            native_worlds_started=0, optimizer_updates=0))
        return
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    run.created_output = output
    write(output/'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), original_worlds=0, optimizer_updates=0, GPU_training=False,
        role='read-only-training-data-coverage', native_execution_modified=False))
    write(output/'plan.json', plan)
    started, last, committed = time.monotonic(), 0., 0
    records = []
    for index, record in enumerate(data.records):
        reader, n = record['reader'], record['rows']
        arrays = {name: np.empty((n, *reader.fields[name][1]), dtype=reader.fields[name][0]) for name in FIELDS}
        for part, spec in enumerate(reader.index['shards']):
            if (output/'STOP').exists():
                raise RuntimeError('Operator stopped read-only extraction; committed trajectories remain')
            values = load_fields(reader, part)
            lo, hi = spec['start'], spec['start']+spec['rows']
            for name in FIELDS:
                arrays[name][lo:hi] = values[name]
            now = time.monotonic()
            if now-last >= 5:
                write(output/'progress.json', dict(status='analysis', phase='teacher_numeric_coverage',
                    updated_utc=utc_now(), completed_trajectories=len(records), total_trajectories=len(data.records),
                    source_rows_read=committed+hi, target_source_rows=data.rows,
                    wall_seconds=now-started, original_worlds=0, optimizer_updates=0,
                    active_trajectory=index, active_native_seed=record['native_seed']))
                last = now
        if (not np.all(arrays['native_seeds'] == record['native_seed'])
                or not np.array_equal(arrays['episode_starts'], np.arange(n)==0)
                or not np.array_equal(arrays['dones'], np.arange(n)==n-1)
                or not np.array_equal(arrays['before_frames'][1:], arrays['after_observation_frames'][:-1])
                or not np.array_equal(arrays['before_episode_frames'][1:], arrays['after_episode_frames'][:-1])
                or not np.all(arrays['after_episode_frames'] > arrays['before_episode_frames'])
                or arrays['before_episode_frames'][0] != 0
                or arrays['after_episode_frames'][-1] != record['entry']['native_frames']
                or not np.isfinite(arrays['features']).all()
                or np.any((arrays['actions']<0)|(arrays['actions']>18))):
            raise ValueError('Original numeric trajectory is not complete and chronological')
        target = output/f'game{index:03d}-seed{record["native_seed"]}.npz'
        with target.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        with np.load(target, allow_pickle=False) as sealed:
            if any(not np.array_equal(sealed[name], arrays[name]) for name in FIELDS):
                raise ValueError('Committed compact data differs from original precision')
        item = dict(index=index, native_seed=record['native_seed'], original_BC_split=(
            'train' if index in split['train'] else 'heldout'), source_index=str(reader.folder/'index.json'),
            source_index_sha256=digest(reader.folder/'index.json'), compact=str(target), compact_sha256=digest(target),
            coverage=coverage(arrays))
        write(output/f'game{index:03d}-receipt.json', item)
        records.append(item); committed += n
    if committed != data.rows:
        raise ValueError('Every archived training teacher observation must be accounted for')
    summaries = {}
    for group in ('all', 'train', 'heldout'):
        subset = [x for x in records if group=='all' or x['original_BC_split']==group]
        stats = [x['coverage'] for x in subset]
        summaries[group] = {key: sum(x[key] for x in stats) for key in (
            'rows', 'teacher_Bomb_action_rows', 'state4_rows', 'usable_deathbomb_rows',
            'usable_deathbomb_teacher_Bomb_rows', 'usable_deathbomb_teacher_nonBomb_rows',
            'enlarged_player_rows', 'enlarged_usable_deathbomb_rows', 'normal_usable_deathbomb_rows')}
        for key in ('usable_by_stage', 'usable_by_remaining_frames', 'usable_enlarged_by_stage', 'per_stage_rows'):
            total = Counter()
            for item in stats:
                total.update(item[key])
            summaries[group][key] = dict(total)
        summaries[group]['native_seeds'] = [x['native_seed'] for x in subset]
    write(output/'report.json', dict(schema='B1-full-teacher-native-critical-coverage-v1', status='completed',
        utc=utc_now(), original_manifest_sha256=data.manifest_sha256, original_BC_split=split,
        original_numeric_precision_preserved=True, native_seed_partition='train-only0..5',
        separate_whole_trajectory_heldout=True, summaries=summaries, trajectories=records,
        actual_source_rows=committed, wall_seconds=time.monotonic()-started,
        new_optimizer_updates=0, new_original_worlds=0, rewards_modified=False,
        current_native_evaluation_modified=False, no_model_fitting_or_selection=True))
    write(output/'progress.json', dict(status='completed', updated_utc=utc_now(), source_rows_read=committed,
        completed_trajectories=len(records), wall_seconds=time.monotonic()-started, original_worlds=0, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualify-only', type=Path)
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text(encoding='utf-8-sig'))
    try:
        run(plan, arguments.qualify_only)
    except BaseException:
        output = getattr(run, 'created_output', None)
        if output is not None:
            write(output/'failure.json', dict(traceback=traceback.format_exc(), no_automatic_restart=True))
        raise
