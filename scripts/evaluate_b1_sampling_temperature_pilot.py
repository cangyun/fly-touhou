"""One frozen B1,balanced native worlds; categorical temperature1 versus0.5."""
from pathlib import Path
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(plan):
    if (plan.get('workers_each') not in (14,15) or plan.get('episodes_each')!=plan.get('workers_each')
            or plan.get('maximum_original_worlds')!=30 or plan.get('arms')!={'T1':1.,'T05':.5}
            or plan.get('action_selection')!='sampled' or plan.get('policy_actions')!=19
            or plan.get('native_frame_limit','missing') is not None or plan.get('continues')!=0
            or plan.get('native_seed_split')!='development'):
        raise ValueError('Require a balanced14+14 or15+15 frozen-temperature pilot with natural terminals')
    seeds=plan['expected_native_seeds']
    if len(seeds)!=plan['episodes_each'] or len(set(seeds))!=len(seeds) or any(type(s) is not int or s%10 not in (6,7) for s in seeds):
        raise ValueError('Unique declared development seeds required')


def check_world_budget(plan,existing_originals):
    total=2*plan['workers_each']+existing_originals
    if total>plan['maximum_original_worlds']:raise ValueError('Reserve existing original games within the30-world maximum')
    return total


def validate_prior(report,plan):
    if (report.get('schema')!='B1-point-risk-calibration-paired-development40-v1'
            or report.get('status')!='completed' or report.get('native_games')!=40
            or report.get('new_optimizer_updates')!=0 or report.get('formal_test_seeds_used') is not False):
        raise ValueError('Wait for both complete20-game calibrated-point cohorts')
    policies=report['policies']
    for role in ('baseline','candidate'):
        item=policies[role];rows=item.get('episode_results',[])
        if (len(rows)!=20 or item.get('policy_frozen') is not True or item.get('PPO_updates')!=0
                or item.get('teacher_actions')!=0 or item.get('safety_takeovers')!=0 or item.get('continues')!=0
                or [x['native_seed'] for x in rows]!=plan['expected_prior_native_seeds']
                or [x['evaluation_game'] for x in rows]!=list(range(1,21))
                or any(x.get('outcome') not in ('clear','gameover') or x.get('full_story_episode') is not True for x in rows)
                or item.get('clears')!=sum(x['outcome']=='clear' for x in rows)):
            raise ValueError('Prior complete naturally terminated frozen cohorts must be verified from their actual rows')
    if (policies['candidate']['episodes']!=20 or policies['baseline']['episodes']!=20
            or policies['candidate']['checkpoint_sha256']!=plan['checkpoint_sha256']
            or policies['candidate']['clears']<=policies['baseline']['clears']
            or [x['native_seed'] for x in policies['candidate']['episode_results'][:plan['episodes_each']]]!=plan['expected_native_seeds']):
        raise ValueError('Use the verified preferable calibrated checkpoint and its declared initial cohort')
    return policies['candidate']['episode_results'][:plan['episodes_each']]


def qualification(plan,*,require_prior):
    import numpy as np
    import torch
    from fly_touhou.thrl_training_mode import draw_native_seed
    from fly_touhou.thrl_control import alive
    validate_protocol(plan)
    desktop_names=['fly-touhou-job-B1-sampling-'+role+'-'+str(os.getpid()) for role in plan['arms']]
    if any(not re.fullmatch(r'fly-touhou-job-[A-Za-z0-9_-]{1,96}',name) for name in desktop_names):
        raise ValueError('Prospective private desktops must satisfy the native launch contract')
    if sha(ROOT/'source_manifest.json')!=plan['source_manifest_sha256']:raise ValueError('Pilot source manifest changed')
    for row in read(ROOT/'source_manifest.json')['files']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Frozen pilot source changed')
    if sha(plan['environment_config'])!=plan['environment_config_sha256']:raise ValueError('Original environment changed')
    cfg=read(plan['environment_config'])
    if (cfg.get('headless') is not True or cfg.get('reward_schema')!='th14-completion-reward-v3'
            or sha(cfg['game_exe'])!=plan['game_exe_sha256'] or sha(cfg['profile'])!=plan['profile_sha256']):
        raise ValueError('Pinned private original execution and unchanged reward required')
    if sha(plan['checkpoint'])!=plan['checkpoint_sha256']:raise ValueError('Frozen neural weights changed')
    if sha(plan['prior_plan'])!=plan['prior_plan_sha256']:raise ValueError('Prior native comparison protocol changed')
    prior_path=Path(plan['prior_report']);reference=None
    if prior_path.exists():reference=validate_prior(read(prior_path),plan)
    elif require_prior:raise RuntimeError('Prior complete40-game report is not ready')
    if require_prior and alive(read(Path(plan['prior_report']).parent/'launch.json')):
        raise RuntimeError('Prior coordinator is still exiting; do not launch a competing pool')
    seeds=[draw_native_seed(np.random.default_rng(int(np.random.SeedSequence([plan['seed'],i]).generate_state(2)[0])),
                           'development') for i in range(plan['episodes_each'])]
    if seeds!=plan['expected_native_seeds']:raise ValueError('Native seeds differ from the declared paired cohort')
    payload=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA inference required')
    return cfg,payload,reference


def role_run(plan,role):
    import torch,psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_full_story_evaluation import evaluate
    from fly_touhou.thrl_training_mode import episode_protocol
    from fly_touhou.thrl_job_paths import configure_native_temp
    from train_thrl_native import make_env_kwargs,development_summary
    cfg,payload,_=qualification(plan,require_prior=True)
    output=Path(plan['output'])/role;output.mkdir(exist_ok=False);role_run.created_output=output
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        role=role,temperature=plan['arms'][role],native_worlds_max=plan['workers_each'],optimizer_updates=0,desktop_input=False))
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-sampling-'+role+'-'+str(os.getpid())
    configure_native_temp(output)
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True)
    if any(not torch.equal(v.cpu(),payload['model'][k]) for k,v in model.state_dict().items()):raise ValueError('CUDA weight load changed')
    cfg=dict(cfg,workers=plan['workers_each'],PPO_training_active=False,evaluation_checkpoint_sha256=plan['checkpoint_sha256'])
    def stop():
        if (Path(plan['output'])/'STOP').exists():raise RuntimeError('Operator stopped frozen sampling pilot')
    result=evaluate(torch,model,make_env_kwargs(cfg,output,plan['workers_each'],'development'),output/'games',episodes=plan['episodes_each'],
        seed=plan['seed'],check_stop=stop,summary=development_summary,native_split='development',
        episode_protocol=episode_protocol(cfg),action_selection='sampled',action_temperature=plan['arms'][role],
        trace_config=dict(checkpoint_sha256=plan['checkpoint_sha256'],history=128))
    if [x['native_seed'] for x in result['episode_results']]!=plan['expected_native_seeds']:
        raise ValueError('Executed native cohort changed')
    if any(not torch.equal(v.cpu(),payload['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Frozen policy changed')
    if sha(plan['checkpoint'])!=plan['checkpoint_sha256']:raise ValueError('Policy file changed')
    write(output/'report.json',dict(result,checkpoint_sha256=plan['checkpoint_sha256'],role=role,
        formal_test_seeds_used=False,new_optimizer_updates=0))


def optional(path):
    try:return read(path)
    except (FileNotFoundError,ValueError):return {}


def parent_run(plan,qualify_only=None):
    import psutil
    from fly_touhou.thrl_control import alive
    cfg,payload,reference=qualification(plan,require_prior=not bool(qualify_only));del payload
    originals=[p for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='th14.exe']
    reserved=None
    if originals:
        active=optional(Path(plan['managed_demo_control_root'])/'active.json')
        state=optional(Path(active['output'])/'status.json') if active.get('output') else {}
        worker=state.get('worker_identity')
        if len(originals)==1 and alive(worker) and state.get('game',{}).get('pid')==originals[0].pid and originals[0].ppid()==worker['pid']:
            reserved=dict(output=active['output'],worker_identity=worker,native_pid=originals[0].pid,
                          native_create_time=originals[0].create_time(),not_modified=True)
        else:raise RuntimeError('An unrelated original pool is active;leave it untouched')
    check_world_budget(plan,len(originals))
    if qualify_only:
        write(qualify_only,dict(status='passed',utc=utc_now(),source_qualified=True,CUDA_available=True,
            prior_complete40_available=reference is not None,dispatch_still_requires_prior_and_pool_exit=True,
            prospective_owned_originals=2*plan['workers_each'],reserved_managed_demo=reserved,
            prospective_total_originals=2*plan['workers_each']+len(originals),
            native_worlds_started=0,new_optimizer_updates=0));return
    output=Path(plan['output']);output.mkdir(exist_ok=False);parent_run.created_output=output
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        native_worlds_max=2*plan['workers_each'],optimizer_updates=0,desktop_input=False))
    write(output/'plan.json',plan)
    write(output/'reserved_managed_demonstration.json',dict(utc=utc_now(),demo=reserved,not_modified=True))
    write(output/'greedy_reference.json',dict(reused=True,new_native_games=0,formal_test=False,
        source=plan['prior_report'],source_sha256=sha(plan['prior_report']),episode_results=reference))
    workers=plan['workers_each'];episodes=plan['episodes_each']
    write(output/'study_config.json',dict(cfg,workers=2*workers,display_name=f'B1 · 冻结采样温度1 / 0.5 · 各{episodes}局'))
    from register_b1_goal_phase import register
    write(output/'panel_registration.json',register(output,plan['panel_job_id'],'evaluation',
        control_root=Path(plan['control_center_root'])))
    processes={};identities={}
    def publish():
        progress={role:optional(output/role/'games/progress.json') for role in plan['arms']};worlds=[]
        for offset,role in [(0,'T1'),(workers,'T05')]:
            worlds.extend(dict(row,worker=row['worker']+offset,policy=role,temperature=plan['arms'][role])
                          for row in progress[role].get('current_environments',[]))
        state=dict(completed=sum(x.get('completed',0) for x in progress.values()),target=2*episodes,workers=2*workers,
            policy_decisions=sum(x.get('policy_decisions',0) for x in progress.values()),current_environments=worlds,
            by_policy={role:x.get('completed',0) for role,x in progress.items()})
        write(output/'progress.json',dict(status='validation',stage='development',actual_decisions=0,
            target_decisions=0,optimizer_updates=0,evaluation_progress=state,**state))
        write(output/'study_status.json',dict(status='validation',phase='frozen_sampling_temperature',
            current_run=str(output),updated_utc=utc_now(),current_progress=dict(status='validation',stage='development',
            actual_decisions=0,optimizer_updates=0,workers=2*workers,device='cuda:0',evaluation_progress=state)))
    try:
        for role in plan['arms']:
            with (output/(role+'_stdout.log')).open('ab') as stdout,(output/(role+'_stderr.log')).open('ab') as stderr:
                p=subprocess.Popen([plan['python'],'-B',str(Path(__file__)),'--plan',plan['plan_path'],'--role',role],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            processes[role]=p;identities[role]=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time())
        write(output/'dispatched_children.json',identities)
        while any(p.poll() is None for p in processes.values()):
            if (output/'STOP').exists():raise RuntimeError('Operator STOP')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('A frozen sampling arm failed')
            publish();time.sleep(2)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('Frozen role failed')
        reports={role:read(output/role/'report.json') for role in processes}
        if any(x['episodes']!=episodes for x in reports.values()):raise ValueError('Both complete declared sampling cohorts required')
        write(output/'report.json',dict(schema='B1-frozen-categorical-temperature-pilot-v1',status='completed',
            utc=utc_now(),arms=reports,checkpoint_sha256=plan['checkpoint_sha256'],original_games_started_this_job=2*episodes,
            greedy_reference_games_reused=episodes,greedy_reference=reference,matched_initial_native_seeds=True,
            formal_test=False,new_optimizer_updates=0,rewards_model_and_PPO_unchanged=True))
        publish();write(output/'study_status.json',dict(status='finished',phase='completed',current_run=str(output),updated_utc=utc_now()))
    finally:
        pending=[k for k,p in processes.items() if p.poll() is None]
        if pending:
            (output/'STOP').touch();deadline=time.monotonic()+60
            while pending and time.monotonic()<deadline:
                time.sleep(.5);pending=[k for k in pending if processes[k].poll() is None]
            for role in pending:
                actual=optional(output/role/'launch.json')
                if alive(actual):psutil.Process(actual['pid']).terminate()
                if alive(identities[role]):processes[role].terminate()
        write(output/'coordinator_cleanup.json',dict(utc=utc_now(),owned_processes_only=True,no_automatic_restart=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--role',choices=('T1','T05'));parser.add_argument('--qualify-only',type=Path)
    args=parser.parse_args();plan=read(args.plan)
    try:
        if args.role:role_run(plan,args.role)
        else:parent_run(plan,args.qualify_only)
    except BaseException:
        output=getattr(role_run if args.role else parent_run,'created_output',None)
        if output is not None:write(output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_restart=True))
        raise
