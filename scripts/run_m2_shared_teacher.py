"""Bounded matched-information diagnostics, qualification and demonstration data.

Only collect/diagnose/qualify create original-game processes. audit reads saved
native captures only. This entry point never trains a neural network.
"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.native_window_audit import audit_paths, audit_window
from fly_touhou.shared_observation_teacher import SharedObservationTeacher


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sensor_status(path):
    if path is None:
        return False, {'reason': 'death-window audit not supplied'}
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    counts = report.get('classification_counts', {})
    issues = (counts.get('potential_critical_object_omission', 0)
        + counts.get('recorded_observation_differs_from_native_snapshot', 0)
        + counts.get('action_delivery_or_motion_calibration_mismatch', 0))
    unresolved = report.get('unresolved_contact_windows', 0)
    lasers = sum(bool(r.get('terminal_geometry', {}).get('live_lasers_require_separate_geometry_review', 0))
                 for r in report.get('windows', []))
    passed = bool(report.get('at_least20_distinct_windows') and not issues and not unresolved and not lasers)
    return passed, {'distinct_windows': report.get('distinct_geometric_windows', 0),
        'critical_or_observation_issues': issues, 'unresolved_contact_windows': unresolved,
        'laser_windows_not_reconstructed': lasers,
        'reason': '20 distinct localizations with covered hazards and no unreviewed laser geometry'
                  if passed else 'insufficient distinct windows or unresolved critical geometry'}


def expand_inputs(values):
    paths = []
    for item in values:
        path = Path(item)
        if path.is_dir():
            windows = sorted(path.rglob('death_windows.jsonl.gz'))
            paths.extend(windows if windows else sorted(path.rglob('native_*.jsonl.gz')))
        else:
            paths.append(path)
    return paths


def run_native(args):
    from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == 'qualify' and args.episodes not in (None, 20):
        raise ValueError('Qualification requires exactly 20 fresh games')
    default_count = 40 if args.mode == 'diagnose' else (16 if args.split == 'validation' else 64)
    count = 20 if args.mode == 'qualify' else (args.episodes or default_count)
    attempt_limit = count
    if args.mode == 'diagnose' and not 1 <= count <= 40:
        raise ValueError('Initial native diagnostic cap is 40 games')
    if args.mode == 'collect':
        if args.split not in ('train', 'validation'):
            raise ValueError('Successful demonstrations must belong to a whole-episode train or validation split')
        target_cap, attempt_cap = (64, 80) if args.split == 'train' else (16, 20)
        attempt_limit = args.max_attempts or math.ceil(count*1.25)
        if not 1 <= count <= target_cap or not count <= attempt_limit <= attempt_cap:
            raise ValueError('Demonstration target/attempt budget exceeds the declared 64/80 train or 16/20 validation limits')
        if not args.qualification:
            raise ValueError('Demonstrations require the completed matched-teacher qualification')
        qualified = json.loads(Path(args.qualification).read_text(encoding='utf-8'))
        if not qualified.get('teacher_qualified') or qualified.get('observation_size') != 1912:
            raise ValueError('Teacher has not qualified on the exact common1912 protocol')
        if qualified.get('task_start_stage_frame') != 12 or any(
                row.get('native_start_stage_frame') != 12 for row in qualified.get('episode_results', [])):
            raise ValueError('Demonstrations require the new fixed-stage12 teacher qualification')
        if (qualified.get('protocol_sha256') != hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
                or qualified.get('teacher_source_sha256') != hashlib.sha256(
                    (ROOT/'src/fly_touhou/shared_observation_teacher.py').read_bytes()).hexdigest()):
            raise ValueError('Qualified teacher source or base protocol differs from collection')
    run_id = uuid.uuid4().hex
    run_config = {'schema': 'native-shared-teacher-run-v1', 'mode': args.mode, 'run_id': run_id,
        'planned_episodes': count, 'frame_skip': 4, 'horizon_frames': 1024,
        'max_attempts': attempt_limit,
        'target_successful_demonstrations': count if args.mode == 'collect' else None,
        'shared_information': True, 'observation_size': 1912, 'seed': args.seed,
        'task_start_stage_frame': 12,
        'network_training': False, 'desktop_input': False,
        'qualification': args.qualification, 'split': args.split,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/fly_touhou/shared_observation_teacher.py',
                      ROOT/'src/fly_touhou/native_window_audit.py',
                      ROOT/'src/fly_touhou/standard_gru_ppo.py',
                      ROOT/'src/fly_touhou/fixed_start_controller.py',
                      ROOT/'src/fly_touhou/native_diagnostic.py', Path(__file__))}}
    write_json(out/'config.json', run_config)
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    env = None
    results, fingerprints = [], set()
    teacher_actions = 0
    try:
        env = NativeAvoidanceEnv(args.protocol, out/'native', role='m2_'+args.mode,
            task_reward='completion', append_task_time=True, capture_death_windows=True)
        teacher = SharedObservationTeacher(env.feature_names)
        write_json(out/'teacher_specification.json', teacher.specification())
        with gzip.open(out/'teacher_decisions.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as decisions:
            for episode_index in range(attempt_limit):
                if time.monotonic()-started > args.wall_cap_seconds or (out/'STOP').exists():
                    break
                observation, reset_info = env.reset(seed=args.seed+episode_index)
                teacher.reset()
                policy_kind = (('static', 'random')[episode_index % 2] if args.mode == 'diagnose' else 'shared_teacher')
                samples, actions, targets = [], [], []
                ring = deque(maxlen=32)
                total_reward = 0.
                fingerprint = hashlib.sha256()
                first_snapshot = env.native.snapshot
                if first_snapshot.stage_frame != 12:
                    raise ValueError('Matched task requires the original reset checkpoint at stage_frame 12')
                while True:
                    if time.monotonic()-started > args.wall_cap_seconds or (out/'STOP').exists():
                        raise TimeoutError('Bounded M2 run reached wall cap or STOP during an episode')
                    before = env.native.snapshot
                    detail = teacher.query(observation)
                    action = int(detail['action'] if policy_kind == 'shared_teacher' else
                                 0 if policy_kind == 'static' else rng.integers(18))
                    before_observation = observation.copy()
                    observation, reward, terminal, truncated, info = env.step(action)
                    after = env.native.snapshot
                    teacher_actions += int(policy_kind == 'shared_teacher')
                    record = {'episode': episode_index+1, 'step': len(actions)+1,
                        'policy_kind': policy_kind, 'action': action, 'teacher': detail,
                        'hit': bool(info['hit']), 'elapsed_frames': info['elapsed_frames'],
                        'teacher_action_executed': policy_kind == 'shared_teacher'}
                    decisions.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                    samples.append(before_observation)
                    actions.append(action)
                    targets.append(detail['target18'])
                    fingerprint.update(np.round(before_observation, 4).tobytes())
                    fingerprint.update(bytes([action]))
                    ring.append({'before': before.to_dict(), 'after': after.to_dict(),
                        'observation': before_observation.tolist(), 'action': action,
                        'elapsed_frames': 4, 'hit': bool(info['hit'])})
                    total_reward += reward
                    if terminal or truncated:
                        break
                episode = dict(info['native_episode'])
                episode.update(episode_id=f'{run_id}:{args.split}:{episode_index+1:03d}', policy_kind=policy_kind,
                    source='new_original_game', native_start_frame=first_snapshot.frame,
                    native_start_stage_frame=first_snapshot.stage_frame,
                    native_rng_at_reset=reset_info['native_rng'],
                    teacher_actions=len(actions) if policy_kind == 'shared_teacher' else 0,
                    teacher_label_queries=len(actions), trajectory_sha256=fingerprint.hexdigest(),
                    split=args.split, dataset_path=None)
                episode['duplicate_encoded_trajectory'] = episode['trajectory_sha256'] in fingerprints
                fingerprints.add(episode['trajectory_sha256'])
                if info['hit']:
                    episode['death_audit'] = audit_window({'transitions': list(ring),
                        'episode': episode_index+1, 'role': 'm2_'+args.mode}, env.encoder)
                if args.mode == 'collect':
                    dataset_path = out/f'episode_{episode_index+1:03d}.npz'
                    # All episodes are retained; manifest eligibility is explicit.
                    np.savez_compressed(dataset_path, observations=np.asarray(samples, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.int64), target18=np.asarray(targets, dtype=np.float32))
                    episode['dataset_path'] = str(dataset_path)
                    episode['eligible_successful_demonstration'] = bool(episode['horizon_completed_without_hit'])
                results.append(episode)
                progress = {'mode': args.mode, 'episodes_completed': len(results),
                    'successes': sum(r['horizon_completed_without_hit'] for r in results),
                    'teacher_actions_executed': teacher_actions,
                    'wall_seconds': time.monotonic()-started,
                    'latest_frames': episode['frames'], 'latest_policy': policy_kind}
                write_json(out/'progress.json', progress)
                print(json.dumps(progress), flush=True)
                if args.mode == 'diagnose':
                    distinct = {r['death_audit']['fingerprint'] for r in results if 'death_audit' in r}
                    if len(distinct) >= 20:
                        break
                elif args.mode == 'collect' and sum(r['horizon_completed_without_hit'] for r in results) >= count:
                    break
        common_encoder = env.encoder
        env.close()
        env = None
        own_audit = audit_paths([out/'native/death_windows.jsonl.gz'], common_encoder, max_windows=40)
        write_json(out/'death_audit.json', own_audit)
        audit_file = Path(args.sensor_audit) if args.sensor_audit else out/'death_audit.json'
        sensor_passed, sensor_detail = sensor_status(audit_file)
        own_issues = any(r['classification'] != 'covered_geometry_policy_or_timing_failure'
                        or r['terminal_geometry'].get('live_lasers_require_separate_geometry_review', 0)
                        for r in own_audit['windows'])
        sensor_passed = sensor_passed and not own_issues
        sensor_detail.update(current_run_new_critical_or_unresolved_issues=bool(own_issues))
        successes = sum(bool(r['horizon_completed_without_hit']) for r in results)
        report = {'schema': 'native-shared-teacher-qualification-v1' if args.mode == 'qualify'
                  else 'native-shared-teacher-collection-v1',
            'mode': args.mode, 'teacher_qualified': bool(args.mode == 'qualify'
                and len(results) == 20 and successes >= 19 and sensor_passed),
            'episodes': len(results), 'planned_episodes': count, 'successes': successes,
            'max_attempts': attempt_limit,
            'target_successful_demonstrations': count if args.mode == 'collect' else None,
            'critical_sensor_issues_resolved': sensor_passed, 'sensor_details': sensor_detail,
            'observation_size': 1912, 'frame_skip': 4, 'shared_information': True,
            'task_start_stage_frame': 12,
            'teacher_actions_executed': teacher_actions, 'sensor_audit_report': str(audit_file.resolve()),
            'current_run_death_audit_report': str((out/'death_audit.json').resolve()),
            'protocol_sha256': hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
            'teacher_source_sha256': hashlib.sha256((ROOT/'src/fly_touhou/shared_observation_teacher.py').read_bytes()).hexdigest(),
            'desktop_input': False, 'bomb_actions': sum(r['bomb_actions'] for r in results),
            'safety_takeovers': 0, 'network_training': False,
            'native_rng_seed_applied': False,
            'distinct_encoded_trajectories': len(fingerprints),
            'cross_rng_generalization_proven': False,
            'wall_seconds': time.monotonic()-started, 'episode_results': results}
        write_json(out/'report.json', report)
        if args.mode == 'collect':
            write_json(out/'manifest.json', {'schema': 'native-shared-teacher-demonstrations-v1',
                'split': args.split, 'observation_size': 1912, 'feature_names': teacher.feature_names,
                'observation_version': 'native-collision-geometry-v4-with-finite-task-time-v1',
                'base_protocol_path': str(Path(args.protocol).resolve()),
                'protocol_sha256': hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
                'frames_per_action': 4, 'successful_only': True,
                'task_start_stage_frame': 12,
                'all_attempts_retained': True,
                'training_selection': 'only eligible_successful_demonstration == true; failed NPZ retained for audit',
                'qualification': str(Path(args.qualification).resolve()), 'episodes': results,
                'successful_demonstrations': successes, 'requested_demonstrations': count,
                'attempts': len(results), 'max_attempts': attempt_limit,
                'demonstration_target_met': successes >= count, 'run_id': run_id})
        return report
    except BaseException:
        write_json(out/'failure.json', {'status': 'failed', 'mode': args.mode,
            'completed_episodes': len(results), 'exception': traceback.format_exc(),
            'episode_results': results, 'teacher_qualified': False})
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'diagnose', 'qualify', 'collect'), required=True)
    parser.add_argument('--protocol', default=str(ROOT/'artifacts/diagnostic_protocol/native-list-v1.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs', nargs='+', default=[])
    parser.add_argument('--max-windows', type=int, default=40)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--max-attempts', type=int,
        help='collect only: bound all attempts while --episodes names the successful demonstration target')
    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--wall-cap-seconds', type=float, default=3600.)
    parser.add_argument('--sensor-audit')
    parser.add_argument('--qualification')
    parser.add_argument('--split', choices=('train', 'validation', 'diagnostic'), default='diagnostic')
    args = parser.parse_args()
    if args.mode == 'audit':
        if not args.inputs:
            parser.error('audit requires saved --inputs')
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
        encoder = GeometryObservationEncoder.from_specification(protocol['encoder'])
        report = audit_paths(expand_inputs(args.inputs), encoder, max_windows=args.max_windows)
        report['base_protocol_path'] = str(Path(args.protocol).resolve())
        report['base_protocol_sha256'] = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
        write_json(output, report)
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ('windows', 'observation_encoder_specification')}, ensure_ascii=False))
    else:
        report = run_native(args)
        print(json.dumps({k: v for k, v in report.items() if k != 'episode_results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
