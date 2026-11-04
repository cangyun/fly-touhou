"""Two frozen B1 policies,15 owned native worlds each, matched development20."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_full_geometry_data import digest


def validate_protocol(plan):
    if (plan.get('workers_each') not in (14,15) or plan.get('episodes_each')!=20
            or plan.get('maximum_original_worlds')!=30 or plan.get('action_selection')!='greedy'
            or plan.get('policy_actions')!=19 or plan.get('native_frame_limit') is not None
            or plan.get('continues')!=0 or plan.get('native_seed_split')!='development'
            or set(plan.get('policies',{}))!={'baseline','candidate'}):
        raise ValueError('Require a balanced14+14/15+15,19-action,natural-terminal development protocol')
    seeds=plan['expected_native_seeds']
    if len(seeds)!=20 or len(set(seeds))!=20 or any(seed%10 not in (6,7) for seed in seeds):
        raise ValueError('Declared development seeds only; no training or formal-test seeds')


def validate_candidate(torch,baseline,candidate):
    if (candidate.get('schema')!='B1-risk-calibrated-frozen-candidate-v1'
            or baseline['model_config']!=candidate['model_config']
            or baseline['config']!=candidate['config']
            or set(baseline['model'])!=set(candidate['model'])):
        raise ValueError('Candidate architecture,input or reward configuration changed')
    changed=[k for k in baseline['model'] if not torch.equal(baseline['model'][k],candidate['model'][k])]
    if not changed or any(not key.startswith('risk.') for key in changed):
        raise ValueError('Only the calibrated point-risk network may differ')
    return changed


def qualification(plan):
    import torch
    from fly_touhou.thrl_training_mode import draw_native_seed
    import numpy as np
    validate_protocol(plan)
    if digest(ROOT/'source_manifest.json')!=plan['source_manifest_sha256']:raise ValueError('Evaluation source changed')
    for row in read(ROOT/'source_manifest.json')['files']:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Frozen evaluation module changed')
    if digest(plan['environment_config'])!=plan['environment_config_sha256']:raise ValueError('Original environment changed')
    cfg=read(plan['environment_config'])
    if (cfg.get('headless') is not True or cfg.get('reward_schema')!='th14-completion-reward-v3'
            or digest(cfg['game_exe'])!=plan['game_exe_sha256'] or digest(cfg['profile'])!=plan['profile_sha256']):
        raise ValueError('Keep the verified private headless original and fixed reward')
    payloads={}
    for role,item in plan['policies'].items():
        if digest(item['checkpoint'])!=item['sha256']:raise ValueError('Frozen neural checkpoint changed')
        payloads[role]=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
    changed=validate_candidate(torch,payloads['baseline'],payloads['candidate'])
    if payloads['candidate'].get('parent_sha256')!=plan['policies']['baseline']['sha256']:
        raise ValueError('Risk candidate was fitted from a different neural parent')
    seeds=[draw_native_seed(np.random.default_rng(int(np.random.SeedSequence([plan['seed'],i]).generate_state(2)[0])),
        'development') for i in range(20)]
    if seeds!=plan['expected_native_seeds']:raise ValueError('Full original cohort does not match the declared20 seeds')
    return cfg,payloads,changed


def role_run(plan,role):
    import psutil,torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_full_story_evaluation import evaluate
    from fly_touhou.thrl_training_mode import episode_protocol
    from train_thrl_native import make_env_kwargs,development_summary
    cfg,payloads,_=qualification(plan);payload=payloads[role];del payloads
    workers=plan['workers_each'];output=Path(plan['output'])/role;output.mkdir(parents=True,exist_ok=False)
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        role=role,workers=workers,CUDA_required=True,optimizer_updates=0,desktop_input=False))
    if not torch.cuda.is_available():raise RuntimeError('CUDA evaluation required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.3);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-geometry-'+role+'-'+str(os.getpid())
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(output)
    cfg=dict(cfg,workers=workers,PPO_training_active=False,evaluation_checkpoint_sha256=plan['policies'][role]['sha256'])
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True)
    if any(not torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Actual CUDA inference weights differ')
    def stop():
        if (Path(plan['output'])/'STOP').exists() or (output/'STOP').exists():raise RuntimeError('Operator stopped the frozen paired evaluation')
    result=evaluate(torch,model,make_env_kwargs(cfg,output,workers,'development'),output/'games',
        episodes=20,seed=plan['seed'],check_stop=stop,summary=development_summary,native_split='development',
        episode_protocol=episode_protocol(cfg),action_selection='greedy',
        trace_config=dict(checkpoint_sha256=plan['policies'][role]['sha256'],history=128))
    if [r['native_seed'] for r in result['episode_results']]!=plan['expected_native_seeds']:
        raise ValueError('Executed original cohort differs')
    if any(not torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Frozen neural policy changed')
    if digest(plan['policies'][role]['checkpoint'])!=plan['policies'][role]['sha256']:
        raise ValueError('Frozen policy file changed during the original evaluation')
    write(output/'report.json',dict(result,role=role,checkpoint_sha256=plan['policies'][role]['sha256'],
        formal_test_seeds_used=False,new_optimizer_updates=0))


def optional(path):
    try:return read(path)
    except (FileNotFoundError,ValueError):return {}


def parent_run(plan,qualify_only=None):
    import psutil
    from fly_touhou.thrl_control import alive
    cfg,payloads,changed=qualification(plan);del payloads
    originals=[p for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='th14.exe']
    reserved=None
    if originals:
        active=optional(Path(plan.get('managed_demo_control_root',''))/'active.json')
        state=optional(Path(active['output'])/'status.json') if active.get('output') else {}
        worker=state.get('worker_identity')
        if (len(originals)==1 and alive(worker) and state.get('game',{}).get('pid')==originals[0].pid
                and originals[0].ppid()==worker['pid']):
            reserved=dict(native_pid=originals[0].pid,native_create_time=originals[0].create_time(),
                worker_identity=worker,output=active['output'])
        else:raise RuntimeError('An unrelated original pool is active; do not displace it')
    workers=plan['workers_each'];owned_worlds=2*workers
    if owned_worlds+len(originals)>30:raise RuntimeError('Reserve the active demonstration within the30-world budget')
    if qualify_only:
        write(qualify_only,dict(status='passed',utc=utc_now(),changed_tensors=changed,workers_each=workers,
            owned_original_worlds=owned_worlds,reserved_managed_demo=reserved,
            total_original_worlds=owned_worlds+len(originals),native_worlds_started=0,optimizer_updates=0));return
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False);parent_run.created_output=output
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        workers=owned_worlds,optimizer_updates=0,desktop_input=False))
    write(output/'reserved_interactive_demo.json',dict(utc=utc_now(),managed_demo=reserved,not_modified=True))
    write(output/'plan.json',plan);write(output/'study_config.json',dict(cfg,workers=owned_worlds,
        display_name=f'B1 · 原风险{workers}实例 / 尺寸校准{workers}实例 · 完整开发对照'))
    from register_b1_goal_phase import register
    write(output/'panel_registration.json',register(output,plan['panel_job_id'],'evaluation',
        control_root=Path(plan['control_center_root'])))
    processes={};identity={}
    def publish():
        values={role:optional(output/role/'games/progress.json') for role in ('baseline','candidate')}
        worlds=[]
        for offset,role in [(0,'baseline'),(workers,'candidate')]:
            for world in values[role].get('current_environments',[]):
                worlds.append({**world,'worker':world['worker']+offset,'policy':role})
        combined=dict(completed=sum(v.get('completed',0) for v in values.values()),target=40,workers=owned_worlds,
            policy_decisions=sum(v.get('policy_decisions',0) for v in values.values()),
            current_environments=worlds,by_policy={k:v.get('completed',0) for k,v in values.items()})
        write(output/'study_status.json',dict(status='validation',phase='paired_geometry_development',
            updated_utc=utc_now(),current_run=str(output),current_progress=dict(status='validation',stage='development',
            actual_decisions=0,optimizer_updates=0,workers=owned_worlds,device='cuda:0',evaluation_progress=combined)))
        write(output/'progress.json',dict(status='validation',stage='development',actual_decisions=0,
            target_decisions=0,optimizer_updates=0,evaluation_progress=combined,**combined))
    try:
        for role in ('baseline','candidate'):
            stdout=(output/(role+'_stdout.log')).open('ab');stderr=(output/(role+'_stderr.log')).open('ab')
            try:
                process=subprocess.Popen([plan['python'],'-B',str(ROOT/'scripts/evaluate_b1_full_geometry_pair.py'),
                    '--plan',plan['plan_path'],'--role',role],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            finally:stdout.close();stderr.close()
            processes[role]=process;identity[role]=dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time())
        write(output/'dispatched_children.json',identity)
        while any(p.poll() is None for p in processes.values()):
            if (output/'STOP').exists():raise RuntimeError('Operator stopped the comparison')
            if any(p.poll() not in (None,0) for p in processes.values()):raise RuntimeError('One frozen policy evaluation failed; preserve both results')
            publish();time.sleep(2)
        if any(p.returncode!=0 for p in processes.values()):raise RuntimeError('Frozen role exited unsuccessfully')
        reports={role:read(output/role/'report.json') for role in processes}
        if any(row['episodes']!=20 for row in reports.values()):raise ValueError('Wait for both complete20-game cohorts')
        write(output/'report.json',dict(schema='B1-point-risk-calibration-paired-development40-v1',status='completed',
            utc=utc_now(),policies=reports,native_games=40,workers_each=workers,maximum_original_worlds=30,
            formal_test_seeds_used=False,new_optimizer_updates=0,changed_state_tensors=changed,
            reward_input_actor_GRU_critic_and_laser_unchanged=True,initial_native_seeds_matched=True))
        publish();write(output/'study_status.json',dict(status='finished',phase='completed',current_run=str(output),updated_utc=utc_now()))
    finally:
        pending=[k for k,p in processes.items() if p.poll() is None]
        if pending:
            (output/'STOP').touch(exist_ok=True);deadline=time.monotonic()+60
            while pending and time.monotonic()<deadline:
                time.sleep(.5);pending=[k for k in pending if processes[k].poll() is None]
            for role in pending:
                actual=optional(output/role/'launch.json')
                if alive(actual):psutil.Process(actual['pid']).terminate()
                if alive(identity[role]):processes[role].terminate()
        write(output/'coordinator_cleanup.json',dict(utc=utc_now(),owned_processes_only=True,no_automatic_restart=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--role',choices=['baseline','candidate']);p.add_argument('--qualify-only',type=Path);a=p.parse_args()
    plan=read(a.plan)
    try:
        if a.role:role_run(plan,a.role)
        else:parent_run(plan,a.qualify_only)
    except BaseException:
        output=Path(plan['output'])/a.role if a.role else getattr(parent_run,'created_output',None)
        if output is not None and output.is_dir():write(output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),
            new_optimizer_updates=0,no_automatic_restart=True))
        raise
