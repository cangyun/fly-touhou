"""Replay a recorded teacher game to measure the stage-six reactive wall.

Only recorded actions control this private original instance. No learner,
teacher planning, policy features or PPO samples are produced during replay.
Native events and phase clocks are compared with every recorded action.
"""
from pathlib import Path
import argparse
import gzip
import os
import pickle
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
WALL_CALLBACK = 0x42cb00


class ReplayOnlyEncoder:
    last_result = None

    def __init__(self, space):
        import numpy as np
        self.placeholder = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in space.spaces.items()}

    def encode(self, snapshot):
        return self.placeholder

    def specification(self):
        return dict(schema='native-action-replay-no-policy-encoding-v1')


class FixedRecordedSeed:
    def __init__(self, seed):
        self.seed = seed

    def integers(self, low, high):
        return self.seed


def recorded_rows(source, index):
    import numpy as np
    for part in index['chunks']:
        with np.load(source / part['file'], allow_pickle=False) as data:
            for i in range(len(data['actions'])):
                yield {key: data[key][i] for key in (
                    'actions', 'stage', 'stage_frame', 'before_episode_frames',
                    'native_action_frames', 'misses_before', 'bombs_used_before')}


def check_prefix(row, info):
    fields = dict(stage='stage', stage_frame='stage_frame',
        before_episode_frames='episode_frames', misses_before='misses',
        bombs_used_before='bombs_used')
    differences = {key: dict(recorded=int(row[key]), actual=int(info[actual]))
        for key, actual in fields.items() if int(row[key]) != int(info[actual])}
    if differences:
        raise ValueError(f'Recorded teacher prefix diverged: {differences}')


def phase(snapshot):
    roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
    bosses = roles.get('main_bosses', [])
    names = [r.get('ecl_sub_name') for r in bosses]
    resources = snapshot.extra.get('resources', {})
    bodies = resources.get('enemies', []) + resources.get('hidden_enemy_contacts', [])
    reactive = [r['native_contact'] for r in bodies
                if r.get('native_contact', {}).get('special_tick_callback') == WALL_CALLBACK]
    return names, reactive


def capture_reactive(snapshot, memory, contacts, programs):
    """Read callback parameters and EX programs, without predicting semantics."""
    bodies = []
    for contact in contacts:
        # 424866..424872 passes EnemyData as ECX to callback 42cb00.
        near, far = struct.unpack('<ff', memory.read(contact['data_address'] + 0x29c, 8))
        bodies.append(dict(enemy_base=contact['enemy_base'], near_radius=near, far_radius=far))
    points = []
    by_id = {b.id: b for b in snapshot.bullets}
    for raw in snapshot.extra['hazard_metadata']['bullets']:
        if raw['native_state'] not in (1, 2):
            continue
        identity = raw['id']
        marker = struct.unpack('<i', memory.read(identity + 0xbe8, 4))[0]
        if marker not in (1, 2):
            continue
        program = memory.read(identity + 0xc28, 18 * 44).hex()
        program_index = programs.setdefault(program, len(programs))
        timer, _, index = struct.unpack('<iii', memory.read(identity + 0xc18, 12))
        raw = dict(raw, native_ex_index=index,
            native_ex_current_record_hex=program[index*88:(index+1)*88] if 0<=index<18 else None)
        bullet = by_id[identity]
        points.append(dict(id=identity, marker=marker, reaction_timer=timer,
            program_index=program_index, kind=bullet.kind, native=raw))
    return dict(callback=WALL_CALLBACK, bodies=bodies, points=points,
        player=[snapshot.player.x, snapshot.player.y], native_frame=snapshot.frame,
        stage=snapshot.stage, stage_frame=snapshot.stage_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-wall-replay-' + str(os.getpid()))
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json, append_jsonl
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.compact_full_story import factory, story_kwargs
    env = None
    started = time.monotonic()
    decisions = 0
    wall_rows = 0
    programs = {}
    phase_rows = []
    previous_markers = {}
    snapshots = 0
    try:
        configure_native_temp(args.output)
        index = read_shared_json(args.source / 'index.json')
        cfg = read_shared_json(args.environment_config)
        expected = index['episode_summary']
        split = expected['native_seed_split']
        cfg['seed'] = 202611021
        options = story_kwargs(cfg, args.output, 1, split, teacher=False)[0]
        env = factory(options)
        env.encoder = ReplayOnlyEncoder(env.observation_space)
        env._full_story_rng = FixedRecordedSeed(expected['full_story_native_seed'])
        _, info = env.reset()
        write_json(args.output / 'ownership.json', dict(pid=os.getpid(),
            create_time=psutil.Process().create_time(), native_pid=env.pid,
            native_create_time=psutil.Process(env.pid).create_time(), source=str(args.source)))
        for row in recorded_rows(args.source, index):
            check_prefix(row, info)
            snapshot = env.snapshot
            names, contacts = phase(snapshot) if snapshot.stage == 6 else ([], [])
            if snapshot.stage == 6:
                key = (tuple(names), bool(contacts))
                if not phase_rows or key != phase_rows[-1]['key']:
                    phase_rows.append(dict(key=key, names=names, reactive_wall=bool(contacts),
                        first_stage_frame=snapshot.stage_frame, misses_start=info['misses'],
                        bombs_start=info['bombs_used'], decisions=0, player_x_sum=0., player_y_sum=0.))
                phase_rows[-1].update(last_stage_frame=snapshot.stage_frame,
                    misses_end=info['misses'], bombs_end=info['bombs_used'])
                phase_rows[-1]['decisions'] += 1
                phase_rows[-1]['player_x_sum'] += snapshot.player.x
                phase_rows[-1]['player_y_sum'] += snapshot.player.y
            if contacts:
                record = capture_reactive(snapshot, env.controller._memory, contacts, programs)
                markers = {r['id']: r['marker'] for r in record['points']}
                changes = [i for i, value in markers.items()
                           if i in previous_markers and previous_markers[i] != value]
                record.update(decision=decisions, recorded_action=int(row['actions']),
                    phase_names=names, misses=info['misses'], bombs_used=info['bombs_used'],
                    marker_changes=changes)
                append_jsonl(args.output / 'wall-native.jsonl', record)
                if snapshots < 12 and (wall_rows == 0 or changes):
                    with gzip.open(args.output / f'native-scene-{snapshots:02}.pickle.gz', 'wb', compresslevel=1) as stream:
                        pickle.dump(snapshot, stream, protocol=5)
                    snapshots += 1
                previous_markers = markers
                wall_rows += 1
            before = info['episode_frames']
            _, _, done, truncated, info = env.step(int(row['actions']))
            if truncated or info['episode_frames'] - before != int(row['native_action_frames']):
                raise ValueError('Recorded action native frame duration diverged')
            decisions += 1
            if decisions % 128 == 0 or done:
                write_json(args.output / 'progress.json', dict(phase='recorded_teacher_replay',
                    decisions=decisions, stage=info['stage'], stage_frame=info['stage_frame'],
                    wall_rows=wall_rows, wall_programs=len(programs), native_pid=env.pid,
                    seconds=time.monotonic()-started, PPO_samples=0))
            if done:
                break
        actual = info.get('episode_summary', {})
        if decisions != index['rows'] or actual.get('outcome') != expected['outcome']:
            raise ValueError('Replay did not reproduce the source complete native outcome')
        write_json(args.output / 'result.json', dict(status='completed', decisions=decisions,
            source=str(args.source), source_native_seed=expected['full_story_native_seed'],
            episode_summary=actual, wall_rows=wall_rows, phases=phase_rows,
            native_snapshots=snapshots, seconds=time.monotonic()-started, PPO_samples=0,
            evidence='Recorded actions replayed; no new teacher choices or student actions'))
    except BaseException as error:
        write_json(args.output / 'failure.json', dict(error=repr(error),
            traceback=traceback.format_exc(), decisions=decisions, wall_rows=wall_rows))
        raise
    finally:
        write_json(args.output / 'wall-programs.json', dict(records=[
            dict(index=i, program_hex=code) for code, i in programs.items()]))
        write_json(args.output / 'phase-summary.json', dict(phases=phase_rows, wall_rows=wall_rows))
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
