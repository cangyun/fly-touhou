"""Collect 20 student-controlled native episodes and label their visited states.

The CUDA student always selects executed actions. The common-information rule
teacher supplies supervision only. No learning update occurs, and all short
failed episodes retain their complete recurrent prefix.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import gymnasium as gym
import numpy as np
import torch

from fly_touhou.engineering_contract import sha256_file
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.gru_imitation import NATIVE_BOUNDS, state_subset_sha256, validate_teacher_qualification
from fly_touhou.shared_observation_teacher import SharedObservationTeacher
from fly_touhou.spatial_native_features import SpatialNativeFeatures
from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv, NativeGRUPolicy, load_bc_actor


OBSERVATION_VERSION = 'native-collision-geometry-v4-with-finite-task-time-v1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sequence_sha256(observations, actions):
    """The same whole-sequence fingerprint used by the BC demonstration loader."""
    digest = hashlib.sha256()
    for observation, action in zip(observations, actions):
        digest.update(np.round(np.asarray(observation, dtype=np.float32), 4).tobytes())
        digest.update(bytes([int(action)]))
    return digest.hexdigest()


def validate_collection_inputs(args):
    if args.episodes != 20 or args.round_index not in (1, 2):
        raise ValueError('DAgger has exactly 20 attempts per round and at most rounds 1 and 2')
    if args.wall_cap_seconds <= 0:
        raise ValueError('DAgger requires a positive wall-time cap')
    protocol = json.loads(Path(args.base_protocol).read_text(encoding='utf-8'))
    encoder = GeometryObservationEncoder.from_specification(protocol['encoder'])
    names = list(encoder.feature_names)+['task_progress']
    if len(names) != 1912:
        raise ValueError('DAgger requires the exact common1912 input protocol')
    qualification = validate_teacher_qualification(args.teacher_qualification)
    report = json.loads(Path(args.teacher_qualification).read_text(encoding='utf-8'))
    protocol_hash = sha256_file(args.base_protocol)
    teacher_hash = sha256_file(ROOT/'src/fly_touhou/shared_observation_teacher.py')
    if report.get('protocol_sha256') != protocol_hash or report.get('teacher_source_sha256') != teacher_hash:
        raise ValueError('Qualified teacher or protocol changed before DAgger labeling')
    if report.get('task_start_stage_frame') != 12 or any(
            row.get('native_start_stage_frame') != 12 for row in report.get('episode_results', [])):
        raise ValueError('DAgger needs the new matched stage_frame12 teacher qualification')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('metadata', {})
    names_hash = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    if (metadata.get('schema') != 'native-gru-bc-actor-state-v1'
            or metadata.get('hidden_size') not in (128, 256)
            or metadata.get('observation_size') != 1912
            or metadata.get('observation_version') != OBSERVATION_VERSION
            or metadata.get('feature_names_sha256') != names_hash
            or tuple(metadata.get('bounds', ())) != NATIVE_BOUNDS):
        raise ValueError('Student checkpoint architecture/observation meaning differs from the qualified protocol')
    previous = None
    if args.round_index == 2:
        if args.previous_dagger_manifest is None:
            raise ValueError('Round 2 requires the completed round-1 manifest to account for the 40-game total cap')
        previous_path = Path(args.previous_dagger_manifest).resolve()
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if (previous.get('schema') != 'native-shared-teacher-dagger-v1'
                or previous.get('round_index') != 1 or previous.get('attempts') != 20
                or len(previous.get('episodes', [])) != 20
                or previous.get('protocol_sha256') != protocol_hash
                or previous.get('qualification_sha256') != qualification['sha256']):
            raise ValueError('Previous DAgger round is incomplete or belongs to another protocol/teacher')
    elif args.previous_dagger_manifest is not None:
        raise ValueError('Round 1 cannot include a previous DAgger round')
    return names, metadata, qualification, {
        'protocol_sha256': protocol_hash, 'teacher_source_sha256': teacher_hash,
        'feature_names_sha256': names_hash, 'previous_round_count': 20 if previous else 0}


def collect_student_episode(env, teacher, policy, *, seed, deterministic, deadline,
                            stop_path, decision_stream=None, episode_id=''):
    observation, reset_info = env.reset(seed=seed)
    teacher.reset()
    state = None
    episode_start = np.ones(1, dtype=bool)
    observations, labels, targets, executed, danger = [], [], [], [], []
    first = env.native.snapshot
    if first.stage_frame != 12:
        raise ValueError('DAgger and teacher must begin at the same fixed stage_frame12 checkpoint')
    try:
        while True:
            if time.monotonic() > deadline or Path(stop_path).exists():
                raise TimeoutError('DAgger reached its bounded wall cap or STOP')
            vector = np.asarray(observation, dtype=np.float32)
            if vector.shape != (1912,) or not np.isfinite(vector).all():
                raise ValueError('Student received an invalid common observation')
            if vector[-1] != np.float32(len(executed)*4/1024):
                raise ValueError('DAgger must retain the complete prefix from reset')
            # Student executes its own prediction. The teacher is queried only
            # after that action is fixed; its label never overrides this action.
            with torch.inference_mode():
                selected, state = policy.predict(vector[None], state=state, episode_start=episode_start,
                                                 deterministic=deterministic)
            action = int(np.asarray(selected).item())
            if not 0 <= action < 18:
                raise ValueError('Student selected a Bomb or invalid action')
            detail = teacher.query(vector)
            label = int(detail['action'])
            observations.append(vector.copy())
            labels.append(label)
            targets.append(detail['target18'])
            executed.append(action)
            danger.append(detail.get('mode') == 'maximum_margin')
            observation, _, terminal, truncated, info = env.step(action)
            if info['elapsed_frames'] != 4:
                raise ValueError('DAgger native action interval changed')
            if decision_stream is not None:
                decision_stream.write(json.dumps({'episode_id': episode_id, 'step': len(executed),
                    'executed_action': action, 'teacher_label': label, 'teacher_action_executed': False,
                    'teacher_label_only': True, 'teacher': detail, 'hit': bool(info['hit'])},
                    allow_nan=False, separators=(',', ':'))+'\n')
            episode_start[:] = False
            if terminal or truncated:
                episode = dict(info['native_episode'])
                break
            if len(executed) >= 256:
                raise ValueError('Original task failed to terminate within 1024 logical frames')
    except BaseException as error:
        # System failures are retained as partial diagnostics, never supplied
        # to training as if they were ordinary terminated native episodes.
        error.dagger_partial_arrays = {'observations': np.asarray(observations, dtype=np.float32).reshape(-1, 1912),
            'actions': np.asarray(labels, dtype=np.int64), 'target18': np.asarray(targets, dtype=np.float32).reshape(-1, 18),
            'executed_actions': np.asarray(executed, dtype=np.int64)}
        raise
    arrays = {'observations': np.asarray(observations, dtype=np.float32),
        'actions': np.asarray(labels, dtype=np.int64), 'target18': np.asarray(targets, dtype=np.float32),
        'executed_actions': np.asarray(executed, dtype=np.int64),
        'behavior_actions': np.asarray(executed, dtype=np.int64), 'danger': np.asarray(danger, dtype=bool)}
    teacher_fingerprint = sequence_sha256(arrays['observations'], arrays['actions'])
    behavior_fingerprint = sequence_sha256(arrays['observations'], arrays['executed_actions'])
    episode.update(episode_id=episode_id, policy_kind='student_gru',
        native_start_frame=first.frame, native_start_stage_frame=first.stage_frame,
        task_start_stage_frame=12,
        native_rng_at_reset=reset_info.get('native_rng'), teacher_actions=0,
        teacher_label_queries=len(executed), teacher_label_only=True, no_takeover=True,
        trajectory_sha256=teacher_fingerprint, teacher_label_trajectory_sha256=teacher_fingerprint,
        behavior_trajectory_sha256=behavior_fingerprint,
        full_prefix_from_reset=True, eligible_dagger_training=True,
        eligible_successful_demonstration=bool(episode['horizon_completed_without_hit']))
    return arrays, episode


def run(args):
    names, metadata, qualification, identities = validate_collection_inputs(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {'schema': 'native-shared-teacher-dagger-v1', 'dataset_kind': 'dagger',
        'mode': 'dagger', 'split': 'train', 'round_index': args.round_index,
        'planned_attempts': 20, 'max_attempts': 20, 'lifetime_max_rounds': 2,
        'run_id': uuid.uuid4().hex, 'base_protocol_path': str(Path(args.base_protocol).resolve()),
        **identities, 'feature_names': names, 'observation_size': 1912,
        'observation_version': OBSERVATION_VERSION, 'frames_per_action': 4,
        'task_start_stage_frame': 12,
        'student_checkpoint': str(Path(args.checkpoint).resolve()),
        'student_checkpoint_sha256': sha256_file(args.checkpoint),
        'source_checkpoint_sha256': sha256_file(args.checkpoint),
        'qualification': qualification['path'], 'qualification_sha256': qualification['sha256'],
        'teacher_qualification_sha256': qualification['sha256'],
        'previous_dagger_manifest': str(Path(args.previous_dagger_manifest).resolve()) if args.previous_dagger_manifest else None,
        'previous_dagger_manifest_sha256': sha256_file(args.previous_dagger_manifest) if args.previous_dagger_manifest else None,
        'behavior_mode': args.behavior_mode, 'seed': args.seed,
        'teacher_label_only': True, 'no_takeover': True, 'teacher_actions_executed': 0,
        'bomb_actions': 0, 'desktop_input': False, 'network_training': False,
        'student_device_required': 'cuda', 'native_rng_seed_applied': False,
        'native_rng_controlled': False, 'cross_rng_generalization_proven': False,
        'fingerprint_semantics': 'trajectory_sha256/teacher_label_trajectory_sha256=obs+teacher label; behavior_trajectory_sha256=obs+executed student action',
        'successful_only': False, 'all_attempts_retained': True,
        'source_hashes': {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT/'src/fly_touhou/shared_observation_teacher.py',
            ROOT/'src/fly_touhou/standard_gru_ppo.py', ROOT/'src/fly_touhou/spatial_native_features.py')}}
    write_json(output/'config.json', dict(config, status='validated_only' if args.validate_only else 'ready'))
    if args.validate_only:
        return dict(config, status='validated_only', game_process_started=False, CUDA_inference_started=False)
    if not torch.cuda.is_available():
        raise RuntimeError('DAgger student inference requires CUDA; silent CPU fallback is forbidden')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = NativeGRUPolicy(gym.spaces.Box(-1., 1., shape=(1912,), dtype=np.float32),
        gym.spaces.Discrete(18), lambda _: 1e-4, lstm_hidden_size=metadata['hidden_size'],
        features_extractor_class=SpatialNativeFeatures,
        features_extractor_kwargs={'feature_names': names, 'bounds': NATIVE_BOUNDS}).to('cuda')
    receipt = load_bc_actor(policy, args.checkpoint)
    policy.set_training_mode(False)
    before_actor = state_subset_sha256(policy, private_critic=False)
    env = None
    episodes = []
    started = time.monotonic()
    try:
        env = NativeAvoidanceEnv(args.base_protocol, output/'native', role=f'dagger_round{args.round_index}',
            task_reward='completion', append_task_time=True, capture_death_windows=True)
        if list(env.feature_names) != names:
            raise ValueError('Student environment changed the declared common feature names')
        teacher = SharedObservationTeacher(env.feature_names)
        with gzip.open(output/'student_decisions.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as stream:
            for index in range(20):
                identity = f'{config["run_id"]}:dagger{args.round_index}:{index+1:03d}'
                arrays, episode = collect_student_episode(env, teacher, policy, seed=args.seed+index,
                    deterministic=args.behavior_mode == 'argmax', deadline=started+args.wall_cap_seconds,
                    stop_path=output/'STOP', decision_stream=stream, episode_id=identity)
                path = output/f'episode_{index+1:03d}.npz'
                np.savez_compressed(path, **arrays)
                episode.update(dataset_path=str(path), dataset_sha256=sha256_file(path), split='train',
                    student_checkpoint_sha256=config['student_checkpoint_sha256'])
                episodes.append(episode)
                write_json(output/'progress.json', {'round_index': args.round_index,
                    'attempts': len(episodes), 'target_attempts': 20,
                    'successes': sum(e['horizon_completed_without_hit'] for e in episodes),
                    'label_decisions': sum(e['decisions'] for e in episodes),
                    'teacher_actions_executed': 0, 'wall_seconds': time.monotonic()-started})
                print(json.dumps({'attempts': len(episodes), 'last_frames': episode['frames'],
                                  'teacher_actions_executed': 0}), flush=True)
        if state_subset_sha256(policy, private_critic=False) != before_actor or policy.optimizer.state:
            raise ValueError('DAgger collection unexpectedly changed student weights or optimizer')
        result = dict(config, status='finished', attempts=len(episodes), episodes=episodes,
            completed_round=len(episodes) == 20, teacher_label_queries=sum(e['decisions'] for e in episodes),
            successful_demonstrations=sum(e['horizon_completed_without_hit'] for e in episodes),
            student_weights_unchanged=True, warmstart_receipt=receipt,
            wall_seconds=time.monotonic()-started, total_dagger_attempts_through_round=20*args.round_index)
        write_json(output/'manifest.json', result)
        write_json(output/'report.json', {key: value for key, value in result.items() if key != 'episodes'})
        return result
    except BaseException as error:
        if getattr(error, 'dagger_partial_arrays', None) is not None:
            np.savez_compressed(output/'interrupted_episode_partial.npz', **error.dagger_partial_arrays)
        write_json(output/'failure.json', {'status': 'failed', 'round_index': args.round_index,
            'completed_attempts': len(episodes), 'episode_results': episodes,
            'partial_prefix_retained_not_training_eligible': getattr(error, 'dagger_partial_arrays', None) is not None,
            'teacher_actions_executed': 0, 'network_training': False, 'exception': traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base-protocol', type=Path, required=True)
    parser.add_argument('--teacher-qualification', type=Path, required=True)
    parser.add_argument('--round-index', type=int, choices=(1, 2), required=True)
    parser.add_argument('--previous-dagger-manifest', type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--behavior-mode', choices=('argmax', 'categorical_sampling'), default='argmax')
    parser.add_argument('--seed', type=int, default=400011)
    parser.add_argument('--wall-cap-seconds', type=float, default=1800.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('episodes', 'feature_names', 'source_hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
