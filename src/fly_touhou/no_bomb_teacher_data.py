"""Full raw ResourceV10 NoBomb first-life demonstrations, not PPO samples.

The real original-game contact ends this avoidance task with lives remaining.
It is explicitly not a natural Game Over and cannot enter the old natural-
episode loader. No game, Torch, scheduling, or learning starts on import.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .thrl_teacher_archive import TeacherTrajectoryWriter, TeacherTrajectoryReader
from .thrl_full_teacher_data import FullTeacherDataset, sha
from .thrl_full_teacher_collection import archive_fields, unique_teacher_world_options
from .thrl_inputs import SEPARATED_LASER_INPUT, observation_shapes
from .thrl_study_io import read_shared_json as read
from .thrl_durable_records import write_json, append_jsonl

SCHEMA = 'ResourceV10-raw-NoBomb-first-life-training-demonstrations-v1'
ENDPOINTS = ('first_observed_contact', 'natural_clear_without_observed_contact')


def fields():
    return dict(archive_fields(SEPARATED_LASER_INPUT),
                outcomes=(np.int8, ()), raw_targets=(np.float32, (19,)))


class NoBombTeacherWriter(TeacherTrajectoryWriter):
    def append(self, row):
        stage, first = str(int(row['stage'])), self.accepted_rows
        super().append(row)
        spans = self.index.setdefault('original_stage_row_ranges', {})
        if stage not in spans:
            spans[stage] = dict(start=first, stop=first+1)
        else:
            if spans[stage]['stop'] != first:
                raise ValueError('An original stage cannot be resumed after another stage')
            spans[stage]['stop'] = first+1

    def finish(self, summary):
        hit = summary.get('outcome') == ENDPOINTS[0]
        if (summary.get('outcome') not in ENDPOINTS or summary.get('actual_bombs_used') != 0
                or summary.get('continues') != 0 or summary.get('native_frame_cap', 'missing') is not None
                or summary.get('policy_decisions') != self.accepted_rows or not self.accepted_rows
                or (hit and (summary.get('contact_is_native_accepted_not_geometric_prediction') is not True
                             or summary.get('first_collision_observed') is not True))
                or (not hit and summary.get('natural_clear_claim') is not True)):
            raise ValueError('Actual uncapped native first-contact/clear endpoint required')
        if not self.last_done or self.closed:
            raise ValueError('A recording budget cut is not an avoidance endpoint')
        self.flush()
        self.index.update(status='avoidance_episode_completed', natural_episode_complete=False,
                          avoidance_episode_complete=True, native_avoidance_summary=summary)
        self._publish()
        self.closed = True
        self.arrays.clear()
        return dict(self.index)


class NoBombTeacherDataset(FullTeacherDataset):
    """Reuse seed grouping and chronological iteration with a distinct endpoint."""
    def __init__(self, folder, *, reader_cache_bytes=32 * 1024**2):
        self.folder = Path(folder).resolve()
        self.manifest_path = self.folder/'demonstrations.json'
        self.manifest = read(self.manifest_path)
        m = self.manifest
        if (m.get('schema') != SCHEMA or m.get('status') != 'completed'
                or m.get('input_schema') != SEPARATED_LASER_INPUT or m.get('native_seed_split') != 'train'
                or m.get('first_hit_terminal') is not True or m.get('native_frame_horizon', 'missing') is not None
                or m.get('continues_allowed') is not False or m.get('Bomb_forbidden') is not True
                or m.get('PPO_samples') != 0 or m.get('optimizer_updates') != 0
                or m.get('neural_model_test') is not False):
            raise ValueError('Sealed independently typed NoBomb training demonstrations required')
        self.records, used, offset = [], set(), 0
        required = {key: (np.dtype(dtype), tuple(shape)) for key, (dtype, shape) in fields().items()}
        for entry in m['episodes']:
            path = Path(entry['trajectory_index']).resolve()
            if not path.is_relative_to(self.folder/'trajectories') or sha(path) != entry['trajectory_index_sha256']:
                raise ValueError('NoBomb trajectory index changed or is outside this archive')
            reader = TeacherTrajectoryReader(path.parent, require_complete=False, cache_bytes=reader_cache_bytes)
            meta, index = reader.index['metadata'], reader.index
            seed = meta['native_seed']
            if (type(seed) is not int or not 0 <= seed <= 65535 or seed % 10 >= 6 or seed in used
                    or meta.get('native_seed_split') != 'train' or meta.get('input_schema') != SEPARATED_LASER_INPUT
                    or meta.get('use_for_training') is not True or meta.get('teacher_revision') != m['teacher_revision']
                    or index.get('avoidance_episode_complete') is not True
                    or index.get('natural_episode_complete') is not False
                    or index.get('status') != 'avoidance_episode_completed'
                    or index['native_avoidance_summary'] != entry['avoidance_summary']
                    or entry.get('outcome') not in ENDPOINTS or entry.get('actual_bombs_used') != 0
                    or reader.rows < 1 or reader.rows != entry['recorded_samples']
                    or reader.rows != entry['policy_decisions']
                    or any(reader.fields.get(key) != value for key, value in required.items())):
                raise ValueError('Whole NoBomb training trajectory, native endpoint, or shape differs')
            self.records.append(dict(index=len(self.records), reader=reader, rows=reader.rows,
                offset=offset, native_seed=seed, native_episode=meta['native_episode'], entry=entry))
            used.add(seed)
            offset += reader.rows
        if (not self.records or len(self.records) != m['complete_avoidance_episodes']
                or offset != m['samples'] or offset != m['acknowledged_native_actions']
                or used != set(m['actual_initial_native_seeds'])):
            raise ValueError('NoBomb cohort row or actual seed bookkeeping differs')
        self.rows, self.manifest_sha256 = offset, sha(self.manifest_path)

    def iter_chunks(self, index, *, chunk_rows=32):
        reader = self.records[index]['reader']
        for first, stop, data, legal in super().iter_chunks(index, chunk_rows=chunk_rows):
            last = np.arange(first, stop) == reader.rows-1
            if (np.any(data['actions'] >= 18) or np.any(data['native_bombs_used'] != 0)
                    or np.any(data['targets'][:, 18] != 0)
                    or np.any(data['native_misses'][~last] != 0)
                    or np.any((data['outcomes'] == 1) & (data['teacher_guidance_weight'] != 0))):
                raise ValueError('NoBomb imitation must not reward a failed teacher contact action')
            yield first, stop, data, legal


def collect(config, output, workers, *, on_progress, check_stop,
            chunk_rows=32, reserve_bytes=1024**3, vector_class=None):
    from . import thrl_vector
    from .direct_teacher_no_bomb_reference import direct_teacher_first_contact_factory, movement_reference
    from .resource_teacher_v10 import REVISION
    from .thrl_online_reference import current_targets
    from .thrl_movement_learning import transition_outcome
    from .thrl_training_mode import validate_training_mode
    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('teacher_revision') != REVISION or not 1 <= workers <= 30):
        raise ValueError('Original full-story V5 ResourceV10 training configuration required')
    output = Path(output)
    options, planned = unique_teacher_world_options(config, output, workers)
    write_json(output/'native_seed_plan.json', dict(split='train', actual_seeds=planned,
        full_story_seeds=[row['full_story_seed'] for row in options], unique=True))
    begun, last, decisions = time.monotonic(), 0., 0
    vector, old_factory = None, thrl_vector.worker_factory
    thrl_vector.worker_factory = direct_teacher_first_contact_factory
    writers, pending, complete = {}, {}, []
    failure = None

    def publish(force=False):
        nonlocal last
        now = time.monotonic()
        if force or now-last >= 5:
            on_progress(dict(phase='NoBomb_teacher_raw_collection', teacher_decisions=decisions,
                recorded_samples=sum(w.accepted_rows for w in writers.values()),
                committed_samples=sum(w.committed_rows for w in writers.values()),
                completed=len(complete), target=workers, native_worlds=workers-len(complete),
                elapsed_seconds=now-begun, optimizer_updates=0, PPO_samples=0, neural_model_test=False))
            last = now

    try:
        check_stop()
        Vector = thrl_vector.THRLNativeVector if vector_class is None else vector_class
        vector = Vector(options, output/'workers', timeout=180)
        vector.submit_at(list(range(workers)), 'reset',
                         options=[dict(evaluation_seed=row['full_story_seed']) for row in options])
        while vector._inflight:
            check_stop()
            messages = vector.receive_ready(max_wait_ms=2, check_stop=check_stop)
            decisions += sum(message['operation'] == 'step' for message in messages)
            ready, actions = [], []
            for message in messages:
                i, info, obs = message['index'], message['info'], message['observation']
                if (info.get('continues') != 0 or info.get('bombs_used') != 0
                        or info.get('desktop_input') is not False or info.get('safety_takeovers') != 0
                        or info.get('full_story_native_seed') != planned[i]):
                    raise ValueError('Actual NoBomb, private desktop, or training RNG identity differs')
                if message['operation'] == 'reset':
                    if (info.get('game_rng_seed_applied') is not True or info.get('stage') != 1
                            or info.get('full_story_seed_split') != 'train' or info.get('episode_frames') != 0):
                        raise ValueError('One fresh original opening per training-seed world required')
                    writers[i] = NoBombTeacherWriter(output/'trajectories'/f'world{i:03d}', fields(),
                        chunk_rows=chunk_rows, reserve_bytes=reserve_bytes,
                        metadata=dict(worker=i, native_seed=planned[i], native_seed_split='train',
                            native_episode=info['native_episode'], input_schema=SEPARATED_LASER_INPUT,
                            use_for_training=True, teacher_revision=REVISION, frame_horizon=None,
                            teacher_action_selection='ResourceV10 movement-conditional19 argmax',
                            full_story_start='fresh native opening', continues_allowed=False,
                            demonstrations_are_not_neural_model_performance=True))
                else:
                    old = pending.pop(i)
                    before, action = old['info'], old['action']
                    if (message.get('truncated') is not False or info.get('policy_action') != action
                            or info.get('reference_label_frame') != before['observation_native_frame']
                            or info.get('reference_label_episode') != before['native_episode']
                            or info.get('native_episode') != before['native_episode']
                            or info['episode_frames'] <= before['episode_frames']):
                        raise ValueError('Saved current-before-action teacher data and native acknowledgement differ')
                    outcome = transition_outcome(old['observation']['features'], before, action,
                        obs['features'], dict(info, terminated=bool(message['terminated']), truncated=False))
                    contacted = bool(message['terminated'] and info['avoidance_episode']['first_collision_observed'])
                    scalar = lambda value: np.asarray(value, dtype=np.int64)
                    row = dict(old['observation'], targets=old['target'], raw_targets=old['raw_target'],
                        actions=scalar(action), dones=np.asarray(message['terminated'], np.bool_),
                        episode_starts=np.asarray(writers[i].accepted_rows == 0, np.bool_),
                        native_episodes=scalar(before['native_episode']), native_seeds=scalar(planned[i]),
                        before_frames=scalar(before['observation_native_frame']),
                        after_observation_frames=scalar(info['observation_native_frame']),
                        stage=scalar(before['stage']), stage_frame=scalar(before['stage_frame']),
                        before_episode_frames=scalar(before['episode_frames']), after_episode_frames=scalar(info['episode_frames']),
                        native_misses=scalar(info['misses']), native_bombs_used=scalar(info['bombs_used']),
                        observed_deathbomb_windows=scalar(info['episode_reward_diagnostics']['deathbomb_observed_window_count']),
                        reward_vector=np.asarray(info['reward_vector'], np.float32),
                        teacher_guidance_weight=np.asarray(0. if contacted else old['weight'], np.float32),
                        outcomes=np.asarray(outcome, np.int8))
                    writers[i].append(row)
                    if message['terminated']:
                        summary = dict(info['avoidance_episode'], continues=info['continues'])
                        writers[i].finish(summary)
                        index_path = writers[i].folder/'index.json'
                        entry = dict(summary, worker=i, recorded_samples=writers[i].accepted_rows,
                            trajectory_index=str(index_path.resolve()), trajectory_index_sha256=sha(index_path),
                            avoidance_summary=summary)
                        complete.append(entry)
                        append_jsonl(output/'avoidance_episodes.jsonl', entry)
                        continue
                p, weight = movement_reference(info, input_schema=SEPARATED_LASER_INPUT)
                target = np.concatenate((p.astype(np.float32), np.zeros(1, np.float32)))
                action = int(p.argmax())
                pending[i] = dict(info=info, action=action, target=target, weight=weight,
                    raw_target=current_targets(info, input_schema=SEPARATED_LASER_INPUT).astype(np.float32),
                    observation={key: np.array(obs[key], copy=True) for key in observation_shapes(SEPARATED_LASER_INPUT)})
                ready.append(i)
                actions.append(action)
            if ready:
                vector.submit_at(ready, 'step', actions)
            publish()
        if len(complete) != workers or pending or decisions != sum(w.accepted_rows for w in writers.values()):
            raise ValueError('Every acknowledged action and whole first-life endpoint must be recorded')
        result = dict(schema=SCHEMA, status='completed', input_schema=SEPARATED_LASER_INPUT,
            teacher_revision=REVISION, native_seed_split='train', actual_initial_native_seeds=planned,
            complete_avoidance_episodes=len(complete), samples=decisions, acknowledged_native_actions=decisions,
            native_frame_horizon=None, first_hit_terminal=True, Bomb_forbidden=True,
            continues_allowed=False, neural_model_test=False, PPO_samples=0, optimizer_updates=0,
            episodes=sorted(complete, key=lambda entry: entry['worker']),
            wall_seconds=time.monotonic()-begun, raw_original_FP32_observations_retained=True,
            teacher_failed_contact_action_imitation_weight_zero=True, student_policy_performance_claim=False)
    except BaseException as error:
        failure = repr(error)
        write_json(output/'collection_failure_prefix.json', dict(error=failure,
            acknowledged_actions=decisions, completed=len(complete),
            recorded_samples=sum(w.accepted_rows for w in writers.values()),
            unfinished_games_counted_as_failures=False, no_automatic_retry=True))
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
                write_json(output/'native_cleanup.json', dict(close_errors=vector.close_errors))
                if vector.close_errors and failure is None:
                    raise RuntimeError('Unresolved owned NoBomb teacher native worlds')
        finally:
            for writer in writers.values():
                if not writer.closed:
                    writer.close_prefix(failure or 'unfinished avoidance demonstration')
            thrl_vector.worker_factory = old_factory
    write_json(output/'demonstrations.json', result)
    publish(True)
    return result
