"""Verify a teacher continuation from an actual saved student world.

The original task and RNG are restored unchanged.  The collection endpoint can
be the current stage's end; a Story takeover need not clear all six stages to
provide a verified stage trajectory.  Nothing here supplies PPO samples.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def reached_stage_end(start_stage, snapshot, info, task):
    summary = info.get('episode_summary', {})
    if (info.get('nb_nm_contact') or summary.get('nb_nm_contact')
            or info.get('task_misses', summary.get('task_misses', 0))
            or info.get('task_Bombs', summary.get('task_Bombs', 0))):
        return False
    if snapshot.stage > start_stage:
        return True
    if not info.get('nbnm_success', summary.get('nbnm_success', False)):
        return False
    if task.get('mode') == 'stage' and task.get('stage') == start_stage:
        return True
    reason = info.get('nb_nm_reason') or summary.get('end_reason')
    return bool(reason == 'native-stage-practice-complete' and task.get('mode') == 'chain'
                and task.get('end_stage') == start_stage and not task.get('end_attack'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-decisions', type=int, default=10000)
    parser.add_argument('--road-end-frame', type=int,
                        help='Offline teacher experiment: extend this stage road-search interval')
    parser.add_argument('--search-beam', type=int,
                        help='Offline teacher experiment: candidate beam for the starting stage')
    parser.add_argument('--reference-continuity',type=float,
                        help='Offline teacher experiment: retain a suitable demonstration source across nearby decisions')
    parser.add_argument('--route-bank',type=Path,
                        help='Offline teacher experiment: use this native-verified demonstration bank')
    args = parser.parse_args()
    import torch
    import psutil
    from fly_touhou.nbnm_run import environment, runtime, task_settings
    from fly_touhou.compact_full_story import factory, story_kwargs
    from fly_touhou.resource_teacher_v10 import REVISION
    from fly_touhou.thrl_training_mode import seed_split
    from fly_touhou.thrl_durable_records import write_json, append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    capsule = saved['native_continuation']['worlds'][args.world]
    spec = copy.deepcopy(saved['spec'])
    del saved
    if seed_split(capsule['native_seed']) != 'train':
        raise ValueError('Teacher learning data require a training seed')
    stage_key=str(capsule['landing']['stage'])
    planner=spec['nbnm'].setdefault('live_route_planner',{})
    if args.reference_continuity is not None:
        spec['nbnm'].setdefault('demonstration_route_teacher',{})['source_switch_penalty']=args.reference_continuity
        spec['nbnm']['demonstration_route_teacher']['source_switch_stages']=[int(stage_key)]
    if args.route_bank is not None:
        spec['nbnm'].setdefault('demonstration_route_teacher',{})['manifest']=str(args.route_bank)
    if args.road_end_frame is not None:
        if args.road_end_frame<=capsule['landing']['stage_frame']:
            raise ValueError('Road interval must include the current student position')
        planner.setdefault('road_stage_windows',{})[stage_key]=[0,args.road_end_frame]
    if args.search_beam is not None:
        from fly_touhou.nbnm_live_routes import configured_beam
        planner.setdefault('stage_beams',{})[stage_key]=args.search_beam
        configured_beam(int(stage_key),planner)
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = runtime(args.output)
    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg = environment(spec, evaluation=True)
    cfg['nbnm'] = task_settings(spec, capsule['task'])
    cfg['nbnm']['collect_route_samples'] = True
    options = story_kwargs(cfg, args.output / 'native', 1, 'train', teacher=True)[0]
    options['online_teacher_revision'] = REVISION
    env = None
    begun = time.monotonic()

    def status(phase, **fields):
        write_json(args.output / 'progress.json', dict(
            phase=phase, PID=proc.pid, create_time=proc.create_time(),
            world=args.world, seed=capsule['native_seed'],
            elapsed_seconds=time.monotonic()-begun, network_training=False, **fields))

    try:
        status('restoring_actual_student_world')
        env = factory(options)
        observation, info = env.reset(options={'continuation': capsule})
        write_json(args.output / 'restoration.json', info['native_continuation'])
        prefix = info.get('policy_burn_in') or {}
        features = list(np.asarray(prefix.get('features', []), dtype=np.float32))
        actions = list(map(int, prefix.get('actions', [])))
        if len(features) != len(actions):
            raise ValueError('Incomplete recurrent burn-in history')
        training_start = len(actions)
        start_stage = env.snapshot.stage
        start_frame = env.snapshot.stage_frame
        samples, durations, contacts = [], [], []
        success = False
        ended = False
        last_report = 0.
        for step in range(args.max_decisions):
            label = env._teacher_label
            action = int(label['action'])
            if not 0 <= action < 18:
                raise ValueError('NBNM teacher selected a non-movement action')
            snapshot = env.snapshot
            before_frame = snapshot.frame
            details = env.encoder.last_result['details']
            features.append(np.asarray(observation['features'], np.float32).copy())
            actions.append(action)
            samples.append(info['route_sample'])
            row = dict(frame=before_frame, stage=snapshot.stage,
                       stage_frame=snapshot.stage_frame,
                       position=[snapshot.player.x, snapshot.player.y], action=action,
                       probabilities=label['probabilities'],
                       teacher_weight=label.get('guidance_weight', 1.),
                       margins=np.asarray(details['margins']).tolist(),
                       trusted=list(map(bool, details['trusted'])),
                       route=label.get('nb_nm_route'))
            observation, _, done, truncated, info = env.step(action)
            hit = bool(info.get('nb_nm_contact'))
            durations.append(env.snapshot.frame-before_frame)
            contacts.append(hit)
            row.update(contact=hit, after_frame=env.snapshot.frame,
                       after_phase=info.get('nb_nm_phase'), terminated=bool(done),
                       truncated=bool(truncated), native_end_reason=info.get('nb_nm_reason'))
            append_jsonl(args.output / 'decisions.jsonl', row)
            success = not truncated and reached_stage_end(start_stage, env.snapshot, info, capsule['task'])
            ended = bool(done or truncated or success)
            if time.monotonic()-last_report > 10:
                status('teacher_continuation', decisions=step+1,
                       stage=env.snapshot.stage, stage_frame=env.snapshot.stage_frame)
                last_report = time.monotonic()
            if ended:
                break
        success = bool(success and not any(contacts))
        # Keep failures as diagnostic evidence, never successful training labels.
        archive = 'verified-continuation.npz' if success else 'failed-continuation.npz'
        np.savez_compressed(args.output / archive,
            features=np.asarray(features, np.float32), actions=np.asarray(actions, np.int64),
            native_frames=np.asarray([s['frame'] for s in samples], np.int64),
            native_action_frames=np.asarray(durations, np.int32),
            observed_contacts=np.asarray(contacts, bool),
            position=np.asarray([s['position'] for s in samples], np.float32),
            phase=np.asarray([s['phase'] for s in samples]),
            clock=np.asarray([s['clock'] if s['clock'] is not None else -1 for s in samples]),
            frame=np.asarray([s['frame'] for s in samples]),
            power=np.asarray([s['power'] for s in samples]),
            signature=np.asarray([s['signature'] for s in samples]))
        result = dict(world=args.world, seed=capsule['native_seed'], split='train',
                      source_checkpoint=str(args.checkpoint), source_task=capsule['task'],
                      source_stage=start_stage, source_stage_frame=start_frame,
                      final_stage=env.snapshot.stage, final_stage_frame=env.snapshot.stage_frame,
                      final_phase=info.get('nb_nm_phase'), decisions=len(samples),
                      prefix_decisions=training_start, contacts=sum(contacts), Bombs=0,
                      stage_clear=success, incomplete=bool(truncated or not ended),
                      native_end_reason=info.get('nb_nm_reason'), native_task=info.get('episode_summary'),
                      full_game_NBNM=False, controller='rule_teacher',
                      network_training=False, optimizer_updates=0,
                      elapsed_seconds=time.monotonic()-begun)
        result['teacher_search_settings']=planner
        result['teacher_demonstration_settings']=spec['nbnm'].get('demonstration_route_teacher')
        write_json(args.output / 'report.json', result)
        if success:
            record = dict(file=archive, task=dict(mode='stage', stage=start_stage),
                          task_key=f's{start_stage}_student_continuation',
                          native_seed=capsule['native_seed'], split='train',
                          decisions=len(samples), training_start=training_start,
                          nbnm_success=True, nb_nm_success=True,
                          source_kind='native_verified_teacher_continuation',
                          source_student_checkpoint=str(args.checkpoint), source_world=args.world,
                          initial_memory='zero before complete original student prefix',
                          native_rule_geometry=spec['nbnm'].get('native_rule_geometry'),
                          native_validation=dict(report='report.json', contacts=0, Bombs=0,
                              stage_clear=True, starts_at_source_student_frame=start_frame,
                              ends_at_stage=env.snapshot.stage), full_game_NBNM=False)
            write_json(args.output / 'manifest.json', dict(
                schema='nbnm-successful-history-rehearsal-v1', histories=[record], PPO_rows=0))
        status('complete', **{k:v for k,v in result.items()
                             if k not in ('world','seed','elapsed_seconds','network_training')})
        print(json.dumps(result))
    except BaseException:
        status('failed')
        write_json(args.output / 'failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()
        cleanup_native_temp(temporary)


if __name__ == '__main__':
    main()
