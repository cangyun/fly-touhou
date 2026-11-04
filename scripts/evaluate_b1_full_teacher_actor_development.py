"""Forty original no-Bomb development games after real full-teacher BC.

Compares greedy BC with its pinned previously trained ancestor on the same
twenty development seeds. No optimizer and no formal test seeds are used.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_real_fit(report,payload):
    if (report.get('schema')!='B1-full-natural-teacher-actor-warmup-result-v1'
            or report.get('status')!='completed' or report.get('CUDA') is not True
            or report.get('teacher_complete_episodes')!=30
            or payload.get('schema')!='B1-full-teacher-temporary-actor-warmup-v1'):
        raise ValueError('A completed real thirty-game CUDA actor fit is required')
    for row in (report,payload):
        if row.get('numerical_qualification_only') or row.get('deployment_forbidden') or row.get('PPO_updates')!=0:
            raise ValueError('No numerical smoke policy or mislabeled PPO policy may enter BC development')
    if any(type(report.get(key)) is not int or report[key]<0 or report[key]!=payload.get(key)
            for key in ('selected_epoch','selected_BC_optimizer_steps','paid_BC_optimizer_steps')):
        raise ValueError('Selected BC identity differs from the completed fitter result')
    if report['selected_BC_optimizer_steps']>report['paid_BC_optimizer_steps']:
        raise ValueError('Selected BC work cannot exceed paid BC work')


def native_development_seeds(seed):
    import numpy as np
    from fly_touhou.thrl_training_mode import draw_native_seed
    result=[]
    for game in range(20):
        environment_seed,_=np.random.SeedSequence([seed,game]).generate_state(2)
        result.append(draw_native_seed(np.random.default_rng(int(environment_seed)),'development'))
    if len(set(result))!=20:raise ValueError('Choose a unique native development cohort before model outcomes exist')
    return result


def run(args):
    import psutil,torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from evaluate_thrl_avoidance_probe import run as probe,checkpoint_copy
    from register_b1_goal_phase import register
    if sha(ROOT/'source_manifest.json')!=args.source_manifest_sha256:
        raise ValueError('Qualified development source changed')
    for entry in read(ROOT/'source_manifest.json')['files']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Frozen development file changed')
    if args.workers!=30:raise ValueError('This declared development uses at most30 owned worlds')
    expected_seeds=native_development_seeds(args.seed)
    status=read(args.actor_control/'status.json')
    if status.get('status')!='finished' or status.get('exit_code')!=0:
        raise RuntimeError('Wait for the actual completed actor fitter;do not run an extra evaluator yet')
    identity=read(args.actor_output/'launch.json')
    try:
        p=psutil.Process(identity['pid'])
        if abs(p.create_time()-identity['create_time'])<.01 and p.is_running():
            raise RuntimeError('Actor fitter has not exited')
    except psutil.NoSuchProcess:pass
    if any(p.info['name'] and p.info['name'].lower()=='th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another original-game pool is still present')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no CPU inference fallback')
    result=read(args.actor_output/'result.json');actor=Path(result['policy'])
    if sha(actor)!=result['policy_sha256']:raise ValueError('Selected actor policy changed')
    payload=torch.load(actor,map_location='cpu',weights_only=False);verify_real_fit(result,payload)
    parent=Path(payload['parent_checkpoint'])
    if sha(parent)!=payload['parent_sha256']:raise ValueError('Actual pre-BC ancestor changed')
    ancestor=torch.load(parent,map_location='cpu',weights_only=False)
    if ancestor['model_config']!=payload['model_config']:raise ValueError('Actor and ancestor network architecture differs')
    config=read(args.actor_output/'config.json')
    config.update(source_runtime=str(ROOT),source_manifest_sha256=args.source_manifest_sha256,
        original_actor_training_source=str(args.actor_output),teacher_pretraining=False,
        initial_baseline_semantics='Previously trained E303 ancestor before full natural teacher BC;BC-only development;not formal test',
        initial_baseline_checkpoint_identity=dict(scope='before-full-natural-teacher-BC-and-PPO',
            sha256=payload['parent_sha256'],optimized_decisions=ancestor.get('candidate_optimized_decisions',0)),
        model_config=payload['model_config'],development_objective='avoidance-first-v1')
    args.output.mkdir(parents=True,exist_ok=False);args.output_created=True
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        CUDA_required=True,native_worlds_max=30,optimizer_updates=0,desktop_input=False))
    write(args.output/'study_config.json',config)
    inputs=args.output/'evaluation_input';inputs.mkdir()
    if checkpoint_copy(parent,inputs/'initial.pt')!=payload['parent_sha256']:raise ValueError('Copied ancestor differs')
    if checkpoint_copy(actor,inputs/'latest.pt')!=result['policy_sha256']:raise ValueError('Copied BC actor differs')
    write(inputs/'config.json',config)
    destination=args.output/'movement40'
    write(args.output/'study_status.json',dict(status='validation',phase='frozen_BC_movement_development',
        current_run=str(destination),updated_utc=utc_now(),current_progress=dict(status='validation',stage='development',
        optimizer_updates=0,device='cuda:0',workers=30)))
    write(args.output/'panel_registration.json',register(args.output,args.job_id,'evaluation',control_root=args.control_root))
    comparison=probe(SimpleNamespace(run=inputs,output=destination,episodes=20,workers=30,
        seed=args.seed,native_split='development',device='cuda',max_frames=32768,wall_seconds=5400.,
        include_greedy=True,include_bullet_ablation=False,conditions=['trained_greedy','initial_greedy'],
        capture_failures=True,capture_history_maps=True),model_factory=model_from_config)
    if read(destination/'cleanup.json')!=dict(closed=True,close_errors=[]):raise RuntimeError('Development native cleanup incomplete')
    if comparison['completed']!=40 or comparison['native_seed_split']!='development':raise ValueError('Incomplete BC development')
    rows=comparison['episode_results']
    groups={name:sorted((r for r in rows if r['policy']==name),key=lambda r:r['game'])
            for name in ('trained_greedy','initial_greedy')}
    for group in groups.values():
        if (len(group)!=20 or len({r['native_seed'] for r in group})!=20
                or any(r['native_seed']%10 not in (6,7) or r['actual_bombs_used']!=0 for r in group)):
            raise ValueError('BC movement development must use20 unique development seeds and zero Bombs')
    if any([r['native_seed'] for r in group]!=expected_seeds for group in groups.values()):
        raise ValueError('Actual matched native development seeds differ')
    report=dict(schema='B1-full-natural-teacher-BC-original-movement-development-v1',status='completed',utc=utc_now(),
        actor_policy=str(actor),actor_sha256=result['policy_sha256'],parent_policy=str(parent),parent_sha256=payload['parent_sha256'],
        native_games=40,episodes_per_policy=20,native_seed_split='development',seed=args.seed,
        actual_native_development_seeds=expected_seeds,
        native_frame_cap=32768,action_selection='greedy;conditional movement0..17',Bomb_allowed=False,
        policies=comparison['policies'],paired=comparison['greedy_paired']['initial_greedy'],
        report=str(destination/'report.json'),neural_optimizer_updates=0,teacher_actions=0,safety_takeovers=0,
        formal_test_seeds_used=False,formal_six_stage_acceptance=False,full_policy_optimization_still_required=True,
        original_goal_achieved=False)
    write(args.output/'report.json',report)
    write(args.output/'study_status.json',dict(status='finished',phase='completed',current_run=str(destination),updated_utc=utc_now()))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('actor-output','actor-control','output','control-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--job-id',required=True)
    p.add_argument('--seed',type=int,default=383021);p.add_argument('--workers',type=int,default=30)
    args=p.parse_args()
    try:run(args)
    except BaseException:
        if getattr(args,'output_created',False):write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_restart=True))
        raise
