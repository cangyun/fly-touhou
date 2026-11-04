"""One natural original-game teacher episode per world, no episode horizon.

Supervised demonstrations only. No Torch, policy optimizer, stage restore,
auto-reset, or fixed-length per-world prefix is used by this collector.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .atomic_json_io import atomic_write_json as write
from .thrl_study_io import read_shared_json as read, utc_now
from .thrl_inputs import SEPARATED_LASER_INPUT, observation_shapes
from .thrl_teacher_archive import TeacherTrajectoryWriter


def unique_teacher_world_options(config, output, workers):
    from .thrl_teacher_training import _env_kwargs
    from .thrl_training_mode import draw_native_seed
    options = _env_kwargs(config, output, workers)
    planned, used = [], set()
    for row in options:
        seed = row['full_story_seed']
        actual = draw_native_seed(np.random.default_rng(seed), 'train')
        while actual in used:
            seed += 1
            actual = draw_native_seed(np.random.default_rng(seed), 'train')
        row['full_story_seed'] = seed
        used.add(actual)
        planned.append(actual)
    return options, planned


def archive_fields(schema):
    return {**{name: (np.float32, shape) for name, shape in observation_shapes(schema).items()},
        'targets': (np.float32, (19,)), 'actions': (np.int64, ()),
        'dones': (np.bool_, ()), 'episode_starts': (np.bool_, ()),
        **{name: (np.int64, ()) for name in ('native_episodes', 'native_seeds', 'before_frames',
            'after_observation_frames', 'stage', 'stage_frame', 'before_episode_frames',
            'after_episode_frames', 'native_misses', 'native_bombs_used', 'observed_deathbomb_windows')},
        'reward_vector': (np.float32, (3,)), 'teacher_guidance_weight': (np.float32, ())}


def collect(config, output, workers, *, on_progress, check_stop,
            chunk_rows=32, reserve_bytes=20 * 1024**3, vector_class=None):
    from . import thrl_vector
    from .thrl_teacher import teacher_env_factory, teacher_schema
    from .thrl_teacher_training import _validate_label
    from .thrl_training_mode import validate_training_mode
    if (not validate_training_mode(config) or config['input_schema'] != SEPARATED_LASER_INPUT
            or type(workers) is not int or not 1 <= workers <= 30):
        raise ValueError('Bounded natural full-story current-V5 collection required')
    expected_schema = teacher_schema(config['teacher_revision'])
    output = Path(output)
    options, planned_seeds = unique_teacher_world_options(config, output, workers)
    write(output / 'native_seed_plan.json', dict(split='train', actual_seeds=planned_seeds,
        full_story_seeds=[row['full_story_seed'] for row in options], unique=True))
    fields = archive_fields(config['input_schema'])
    started, last = time.monotonic(), 0.
    vector, previous_factory = None, thrl_vector.worker_factory
    thrl_vector.worker_factory = teacher_env_factory
    writers, pending, infos, complete = {}, {}, {}, []
    acknowledged, failure, result = 0, None, None

    def publish(force=False):
        nonlocal last
        now = time.monotonic()
        if not force and now - last < 5:
            return
        on_progress(dict(stage='teacher_full_story_collection', utc=utc_now(),
            acknowledged_native_actions=acknowledged,
            recorded_samples=sum(w.accepted_rows for w in writers.values()),
            committed_samples=sum(w.committed_rows for w in writers.values()),
            complete_native_episodes=len(complete), target_native_episodes=workers,
            outcomes=dict(Counter(row['outcome'] for row in complete)),
            workers=workers, active_workers=workers-len(complete), wall_seconds=now-started,
            input_schema=config['input_schema'], policy_actions=19, episode_frame_horizon=None,
            optimizer_updates=0, neural_model_test=False, desktop_input=False,
            current_environments=[dict(worker=i, **{key: info.get(key) for key in
                ('pid', 'stage', 'stage_frame', 'episode_frames', 'lives', 'bombs', 'misses', 'bombs_used')})
                for i, info in infos.items() if not writers.get(i) or not writers[i].closed]))
        last = now

    try:
        check_stop()
        cls = thrl_vector.THRLNativeVector if vector_class is None else vector_class
        vector = cls(options, output / 'workers', timeout=180)
        vector.submit_at(list(range(workers)), 'reset')
        publish(True)
        while vector._inflight:
            check_stop()
            messages = vector.receive_ready(max_wait_ms=2, check_stop=check_stop)
            # Count the entire returned acknowledged batch before validating
            # individual rows; later processing errors do not erase paid work.
            acknowledged += sum(msg['operation'] == 'step' for msg in messages)
            ready, actions = [], []
            for msg in messages:
                i, info, obs = msg['index'], msg['info'], msg['observation']
                infos[i] = info
                if (info.get('continues') != 0 or info.get('desktop_input') is not False
                        or info.get('safety_takeovers') != 0
                        or info.get('full_story_native_seed') != planned_seeds[i]):
                    raise ValueError('Original action/seed/no-continue contract changed')
                if msg['operation'] == 'reset':
                    if (info.get('game_rng_seed_applied') is not True or info.get('stage') != 1
                            or info.get('full_story_seed_split') != 'train'
                            or not 0 <= info.get('stage_frame', -1) <= 64
                            or info.get('episode_frames') != 0 or i in writers):
                        raise ValueError('Teacher must start once from a fresh training-seed opening')
                    writers[i] = TeacherTrajectoryWriter(output / 'trajectories' / f'world{i:03d}',
                        fields, chunk_rows=chunk_rows, reserve_bytes=reserve_bytes,
                        metadata=dict(worker=i, native_seed=planned_seeds[i], native_seed_split='train',
                            native_episode=info['native_episode'], input_schema=config['input_schema'],
                            teacher_revision=config['teacher_revision'], teacher_action_selection='source19 argmax',
                            full_story_start='fresh native opening', frame_horizon=None, continues_allowed=False,
                            demonstrations_are_not_neural_model_performance=True, use_for_training=True,
                            source_runtime=config.get('source_runtime'),
                            source_manifest_sha256=config.get('source_manifest_sha256')))
                else:
                    old = pending.pop(i)
                    before, label = old['info'], old['label']
                    if (msg.get('truncated') is not False
                            or info.get('executed_teacher_label_frame') != label['frame']
                            or info.get('executed_teacher_native_episode') != label['native_episode']
                            or info.get('policy_action') != label['action']
                            or info.get('teacher_actions_executed') != before['teacher_actions_executed'] + 1
                            or info.get('native_episode') != before['native_episode']
                            or info['episode_frames'] <= before['episode_frames']):
                        raise ValueError('Saved before-state label differs from acknowledged native action')
                    scalar = lambda value: np.asarray(value, dtype=np.int64)
                    row = {**old['observation'], 'targets': old['target'],
                        'actions': scalar(label['action']), 'dones': np.asarray(msg['terminated'], dtype=np.bool_),
                        'episode_starts': np.asarray(writers[i].accepted_rows == 0, dtype=np.bool_),
                        'native_episodes': scalar(label['native_episode']), 'native_seeds': scalar(planned_seeds[i]),
                        'before_frames': scalar(label['frame']),
                        'after_observation_frames': scalar(info['observation_native_frame']),
                        'stage': scalar(before['stage']), 'stage_frame': scalar(before['stage_frame']),
                        'before_episode_frames': scalar(before['episode_frames']),
                        'after_episode_frames': scalar(info['episode_frames']),
                        'native_misses': scalar(info['misses']), 'native_bombs_used': scalar(info['bombs_used']),
                        'observed_deathbomb_windows': scalar(info['episode_reward_diagnostics']['deathbomb_observed_window_count']),
                        'reward_vector': np.asarray(info['reward_vector'], dtype=np.float32),
                        'teacher_guidance_weight': np.asarray(label.get('guidance_weight', 1.), dtype=np.float32)}
                    writers[i].append(row)
                    if msg['terminated']:
                        summary = dict(info['episode_summary'], continues=info['continues'],
                                       native_seed=planned_seeds[i], worker=i)
                        if (summary['policy_decisions'] != writers[i].accepted_rows
                                or (summary['outcome'] == 'clear' and (summary['stage_history'] != list(range(1, 7))
                                    or info.get('lunatic_six_stage_no_continue') is not True))
                                or (summary['outcome'] == 'gameover' and info.get('native_loss_evidence') not in
                                    ('native-loss-menu', 'complete-native-miss-and-lives-minus-one'))):
                            raise ValueError('Teacher natural endpoint lacks original-game evidence')
                        index = writers[i].finish(summary)
                        entry = dict(summary, trajectory_index=str((writers[i].folder/'index.json').resolve()),
                                     trajectory_index_sha256=hashlib.sha256((writers[i].folder/'index.json').read_bytes()).hexdigest(),
                                     recorded_samples=index['rows'])
                        complete.append(entry)
                        with (output / 'native_episodes.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        continue
                label, target = _validate_label(info, expected_schema)
                pending[i] = dict(info=info, label=label, target=target,
                                  observation={name: obs[name] for name in observation_shapes(config['input_schema'])})
                ready.append(i)
                actions.append(int(label['action']))
            if ready:
                vector.submit_at(ready, 'step', actions)
            publish()
        if len(complete) != workers or pending:
            raise ValueError('Natural teacher cohort ended without all requested endpoints')
        if sum(w.accepted_rows for w in writers.values()) != acknowledged:
            raise ValueError('Every acknowledged teacher action must have exactly one saved row')
        result = dict(schema='B1-current-V5-full-natural-teacher-demonstrations-v1', status='completed',
            utc=utc_now(), input_schema=config['input_schema'], teacher_revision=config['teacher_revision'],
            native_seed_split='train', actual_initial_native_seeds=planned_seeds,
            complete_native_episodes=len(complete), samples=sum(w.accepted_rows for w in writers.values()),
            acknowledged_native_actions=acknowledged, native_frame_horizon=None, first_hit_terminal=False,
            policy_actions=19, autonomous_teacher_Bomb=True, continues_allowed=False,
            neural_model_test=False, PPO_samples=0, optimizer_updates=0,
            source_runtime=config.get('source_runtime'), source_manifest_sha256=config.get('source_manifest_sha256'),
            outcomes=dict(Counter(row['outcome'] for row in complete)), episodes=complete,
            unequal_trajectory_lengths_preserved=True, zero_fabricated_terminal_rows=True,
            wall_seconds=time.monotonic()-started)
    except BaseException as error:
        failure = repr(error)
        write(output / 'collection_failure_prefix.json', dict(utc=utc_now(), error=failure,
            acknowledged_native_actions=acknowledged, completed_native_episodes=len(complete),
            recorded_samples=sum(w.accepted_rows for w in writers.values()),
            committed_samples=sum(w.committed_rows for w in writers.values()),
            unknown_inflight_cost_possible=bool(vector and vector._inflight),
            unfinished_games_counted_as_failures=False, optimizer_updates=0))
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
                write(output / 'native_cleanup.json', dict(close_errors=vector.close_errors,
                    ownership=read(vector.ownership_path)))
                if vector.close_errors and failure is None:
                    raise RuntimeError('Unresolved full-teacher original-world cleanup')
        finally:
            try:
                for writer in writers.values():
                    if not writer.closed:
                        writer.close_prefix(failure or 'collection ended before a natural endpoint')
            finally:
                thrl_vector.worker_factory = previous_factory
    write(output / 'demonstrations.json', result)
    publish(True)
    return result
