"""Finite V10 conditional-movement reference after the frozen neural tests.

No Torch or optimizer. This does not run until its explicit predecessor has
finished and all original worlds are gone. Native terminal state pairs retain
object identities for later motion audits; they do not alter the game.
"""
from pathlib import Path
from collections import Counter, deque
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def conditional_movement(target):
    p = np.asarray(target, dtype=np.float64)
    if p.shape != (19,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 2e-6:
        raise ValueError('Normalized source19 reference required')
    if p[:18].sum() <= 0:
        raise ValueError('Teacher has no movement support;never replace its Bomb with idle')
    return p[:18] / p[:18].sum()


def verify_predecessor(background, result, *, predecessor_alive, native_worlds):
    if (background.get('status') != 'finished' or background.get('exit_code') != 0
            or predecessor_alive or native_worlds != 0):
        raise RuntimeError('Required frozen neural comparison is not fully exited')
    if (result.get('schema') != 'B1-entity-PPO-frozen-native-comparison-v1'
            or result.get('status') != 'completed' or result.get('native_games') != 140
            or result.get('new_optimizer_updates') != 0
            or len(result.get('paired_full_games', [])) != 20
            or set(result.get('full_story', {})) != {'initial','final'}
            or any(x.get('episodes') != 20 for x in result['full_story'].values())):
        raise ValueError('Complete preregistered140 neural games required')


def teacher_pair_factory(kwargs):
    from fly_touhou.thrl_teacher_probe import TeacherAvoidanceProbeEnv

    class NativePairTeacherProbe(TeacherAvoidanceProbeEnv):
        def step(self, action):
            before = self.snapshot
            observation, reward, terminated, truncated, info = super().step(action)
            if terminated or truncated:
                info = dict(info, teacher_terminal_native_pair=dict(
                    before=before.to_dict(), after=self.snapshot.to_dict(),
                    action=int(action), requested_frame_skip=4,
                    before_frame=before.frame, after_frame=self.snapshot.frame,
                    observed_native_frame_delta=self.snapshot.frame-before.frame,
                    native_object_identity_reuse_possible=True,
                    executed_in_original=True, unexecuted_counterfactual=False))
            return observation, reward, terminated, truncated, info

    return NativePairTeacherProbe(**kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(plan_path):
    import psutil
    from fly_touhou.resource_teacher_v10 import REVISION
    from fly_touhou.thrl_online_reference import current_targets
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou import thrl_vector
    plan = read(plan_path)
    before = read(plan['predecessor_identity'])
    try:
        process = psutil.Process(before['pid'])
        live = process.is_running() and abs(process.create_time()-before['create_time']) < .01
    except psutil.Error:
        live = False
    result = read(plan['predecessor_report'])
    background = read(Path(plan['predecessor_control'])/'status.json')
    worlds = sum((p.info['name'] or '').lower() == 'th14.exe' for p in psutil.process_iter(['name']))
    verify_predecessor(background, result, predecessor_alive=live, native_worlds=worlds)
    if (plan['teacher_revision'] != REVISION or plan['workers'] != 30
            or plan['maximum_total_native_worlds'] != 30 or plan['cap_native_frames'] != 32768
            or len(plan['games']) != 20 or len({x['native_seed'] for x in plan['games']}) != 20
            or any(x['native_seed'] % 10 not in (6,7) for x in plan['games'])):
        raise ValueError('One frozen V10 development cohort,30-world ceiling and32k first-hit cap required')
    if sha(ROOT/'source_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Frozen teacher benchmark source changed')
    for item in read(ROOT/'source_manifest.json')['files']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen source differs: '+item['path'])
    cfg = read(plan['config'])
    for key in ('game_exe','profile'):
        if sha(cfg[key]) != cfg[key+'_sha256']:
            raise ValueError('Qualified original identity changed: '+key)
    reference = read(plan['BC_development_report'])
    if sha(plan['BC_development_report']) != plan['BC_development_report_sha256']:
        raise ValueError('Frozen BC development cohort changed')
    cohort = sorted((x for x in reference['episode_results'] if x['policy']=='trained_greedy'), key=lambda x:x['game'])
    if [x['native_seed'] for x in cohort] != [x['native_seed'] for x in plan['games']]:
        raise ValueError('Reference must use all20 existing BC development seeds')
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    write(output/'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), native_worlds_max=30, neural_model_test=False, optimizer_updates=0))
    write(output/'study_config.json', dict(cfg, display_name='规则教师V10 · 禁Bomb移动参考',
        neural_model_test=False, teacher_reference_only=True))
    write(output/'effective_plan.json', plan)
    configure_native_temp(output)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-V10-movement-reference-'+str(os.getpid())
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    jobs = deque(dict(game=game['game'], native_seed=game['native_seed'],
        environment_seed=game['environment_seed'], action_seed=game['action_seed'], policy=policy)
        for game in plan['games'] for policy in ('teacher_greedy','teacher_sampled'))
    kwargs = [dict(executable=cfg['game_exe'], profile_path=cfg['profile'],
        private_appdata=str(output/'native'/f'worker{i}'/'appdata'), difficulty=3,
        frame_skip=4, policy_action_count=19, input_schema=cfg['input_schema'],
        headless=True, hide_window=True, timeout=25, require_complete_reward_mapping=True,
        training_mode='full-story-v1', full_story_split='development', full_story_seed=383021,
        reward_schema=cfg['reward_schema'], reward_gamma=cfg.get('reward_gamma',.999),
        teacher_revision=REVISION, probe_max_frames=32768) for i in range(30)]
    vector = None; old_factory = thrl_vector.worker_factory; slots = {}; rows = []; decisions = 0
    started = time.monotonic(); last = 0.

    def stop():
        if (output/'STOP').exists():
            raise RuntimeError('Operator STOP;no further teacher games')

    try:
        # Use an importable module name, rather than a __main__ closure, on
        # Windows spawn. The exact callable is qualified without native games.
        from benchmark_b1_V10_movement import teacher_pair_factory as spawnable_teacher_pair_factory
        thrl_vector.worker_factory = spawnable_teacher_pair_factory
        vector = thrl_vector.THRLNativeVector(kwargs, output/'workers', timeout=180)

        def assign(i):
            job = dict(jobs.popleft()); job.update(rng=np.random.default_rng(job['action_seed']),
                action_hash=hashlib.sha256(), abstentions=Counter())
            slots[i] = job
            vector.submit_at([i], 'reset', options=[dict(evaluation_seed=job['environment_seed'])])

        for i in range(30):
            assign(i)
        while vector._inflight:
            stop(); messages = vector.receive_ready(max_wait_ms=2, min_batch=4, check_stop=stop)
            ready = []; actions = []
            for message in messages:
                i = message['index']; job = slots[i]; info = message['info']
                if message['operation'] == 'step':
                    decisions += 1
                if message.get('terminated') or message.get('truncated'):
                    endpoint = info['probe_episode']
                    if endpoint['actual_bombs_used'] != 0 or endpoint['native_seed'] != job['native_seed']:
                        raise ValueError('Teacher conditional movement task changed native seed or used Bomb')
                    pair = info['teacher_terminal_native_pair']
                    pair_path = output/'native_collision_pairs'/f"{job['policy']}_game{job['game']:03d}.json"
                    write(pair_path, dict(pair, teacher_revision=REVISION, game=job['game'],
                        native_seed=job['native_seed'], first_collision_observed=endpoint['first_collision_observed'],
                        right_censored_at_cap=endpoint['right_censored_at_cap']))
                    row = dict(endpoint, **{k:job[k] for k in ('game','policy','environment_seed','action_seed')},
                        action_sha256=job['action_hash'].hexdigest(), reference_abstentions=dict(job['abstentions']),
                        raw_native_pair=str(pair_path), raw_native_pair_sha256=sha(pair_path))
                    rows.append(row)
                    with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row)+'\n')
                    if jobs:
                        assign(i)
                    continue
                if message['operation'] == 'reset':
                    if info.get('game_rng_seed_applied') is not True or info.get('full_story_native_seed') != job['native_seed']:
                        raise ValueError('Actual original development RNG differs')
                target = conditional_movement(current_targets(info))
                label = info['teacher_label']
                if label.get('reference_revision') != REVISION:
                    raise ValueError('Worker used a different reference revision')
                job['abstentions'][label.get('guidance_weight_reason','unknown')] += 1
                action = int(target.argmax()) if job['policy']=='teacher_greedy' else min(17,
                    int(np.searchsorted(np.cumsum(target),job['rng'].random(),side='right')))
                job['action_hash'].update(bytes([action])); ready.append(i); actions.append(action)
            if ready:
                vector.submit_at(ready,'step',actions)
            if time.monotonic()-last >= 5:
                write(output/'progress.json',dict(status='teacher_reference_evaluation',utc=utc_now(),
                    completed=len(rows),target=40,decisions=decisions,neural_model_test=False,
                    optimizer_updates=0,native_instances_max=30))
                last=time.monotonic()
        if len(rows) != 40:
            raise ValueError('Incomplete teacher reference cohort')
        summaries={}
        for policy in ('teacher_greedy','teacher_sampled'):
            group=sorted((row for row in rows if row['policy']==policy),key=lambda row:row['game'])
            if [row['native_seed'] for row in group] != [game['native_seed'] for game in plan['games']]:
                raise ValueError('Teacher cohort completion differs')
            values=np.asarray([row['capped_score_native_frames'] for row in group])
            summaries[policy]=dict(episodes=20,mean_capped_frames=float(values.mean()),
                median_capped_frames=float(np.median(values)),observed_collisions=sum(row['first_collision_observed'] for row in group),
                right_censored=sum(row['right_censored_at_cap'] for row in group),actual_Bombs=0)
        write(output/'report.json',dict(schema='B1-V10-development-noBomb-reference-v1',status='completed',
            utc=utc_now(),teacher_revision=REVISION,policies=summaries,episode_results=rows,
            native_games=40,native_seed_split='development',cap_native_frames=32768,
            source_manifest_sha256=plan['source_manifest_sha256'],neural_model_test=False,
            teacher_reference_conditional_on_source18=True,not_an_optimal_noBomb_teacher_claim=True,
            raw_pairs_are_executed_original_states=True,unexecuted_counterfactuals_measured=False,
            current_training_modified=False,optimizer_updates=0,wall_seconds=time.monotonic()-started))
    except BaseException:
        write(output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),completed=len(rows)))
        raise
    finally:
        thrl_vector.worker_factory=old_factory
        if vector is not None:
            vector.close()
        write(output/'native_cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector else [],
            ownership=read(vector.ownership_path) if vector else None))
        if vector is not None and vector.close_errors:
            raise RuntimeError('Unresolved teacher reference native cleanup')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',required=True,type=Path)
    run(parser.parse_args().plan)
