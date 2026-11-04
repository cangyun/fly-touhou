"""Frozen BC-only B1, full natural19-action games on development seeds.

This is the missing intermediate policy measurement, not more optimization.
The preceding fixed140 round must be complete and its actual process exited.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_full_story_recovery import digest


def validate_closed_predecessor(report,status,identity_live,native_worlds):
    if (status.get('status')!='finished' or status.get('exit_code')!=0
            or report.get('schema')!='B1-entity-PPO-frozen-native-comparison-v1'
            or report.get('status')!='completed' or report.get('native_games')!=140
            or report.get('new_optimizer_updates')!=0
            or any(report.get('full_story',{}).get(k,{}).get('episodes')!=20 for k in ('initial','final'))
            or identity_live or native_worlds):
        raise RuntimeError('Complete fixed140 and exited original pool required before BC-only development')


def run(args):
    import psutil,numpy as np,torch
    from fly_touhou.thrl_control import alive
    from fly_touhou.thrl_actor_transfer import validate_actor_payload
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_training_mode import draw_native_seed,episode_protocol
    from fly_touhou.thrl_full_story_evaluation import evaluate
    from train_thrl_native import make_env_kwargs,development_summary
    plan=read(args.plan)
    if digest(ROOT/'source_manifest.json')!=plan['source_manifest_sha256']:
        raise ValueError('Qualified BC full-game source changed')
    for item in read(ROOT/'source_manifest.json')['files']:
        if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen BC evaluator file changed')
    after=Path(plan['after_output']);status=read(Path(plan['after_control'])/'status.json')
    previous=read(after/'report.json')
    validate_closed_predecessor(previous,status,alive(read(after/'launch.json')),
        sum((p.info['name'] or '').lower()=='th14.exe' for p in psutil.process_iter(['name'])))
    if digest(after/'report.json')!=plan['after_report_sha256']:
        raise ValueError('Completed comparison result changed')
    seeds=[]
    for game in range(20):
        environment_seed,_=np.random.SeedSequence([plan['seed'],game]).generate_state(2)
        seeds.append(draw_native_seed(np.random.default_rng(int(environment_seed)),'development'))
    if seeds!=plan['expected_native_seeds'] or len(set(seeds))!=20 or any(s%10 not in (6,7) for s in seeds):
        raise ValueError('Use only the declared20 whole-game development seeds')
    if digest(plan['checkpoint'])!=plan['checkpoint_sha256']:
        raise ValueError('Selected BC neural policy changed')
    result=read(plan['BC_result']);bc=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    if result['policy_sha256']!=plan['checkpoint_sha256'] or result['teacher_complete_episodes']!=30:
        raise ValueError('The actual selected thirty-game BC policy is required')
    if digest(plan['parent'])!=bc['parent_sha256']:raise ValueError('BC ancestor changed')
    parent=torch.load(plan['parent'],map_location='cpu',weights_only=False)
    scope=validate_actor_payload(torch,bc,result,parent);del parent
    config_path=Path(plan['environment_config'])
    if digest(config_path)!=plan['environment_config_sha256']:raise ValueError('Original game configuration changed')
    cfg=read(config_path);runtime=Path(cfg['source_runtime'])
    for name in ('thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py'):
        if digest(ROOT/'src/fly_touhou'/name)!=digest(runtime/'src/fly_touhou'/name):
            raise ValueError('Qualified native execution changed:'+name)
    if (digest(ROOT/'native/th14_fake_d3d9.dll')!=digest(runtime/'native/th14_fake_d3d9.dll')
            or digest(cfg['game_exe'])!=plan['game_exe_sha256']
            or digest(cfg['profile'])!=plan['profile_sha256']):
        raise ValueError('Original headless executable,profile or COM library changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no CPU inference fallback')
    qualification=dict(status='passed',utc=utc_now(),checkpoint_sha256=plan['checkpoint_sha256'],
        actual_BC_scope=scope,development_native_seeds=seeds,full_story_Bomb_allowed=True,
        native_frame_limit=None,teacher_execution=False,optimizer_updates=0,native_worlds_started=0)
    if args.qualify_only:
        write(args.qualify_only,qualification);print(json.dumps(qualification),flush=True);return
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False);args.created=output
    write(output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        CUDA_required=True,optimizer_updates=0,native_worlds_max=20,desktop_input=False))
    write(output/'qualification.json',qualification);write(output/'plan.json',plan)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-BC-full-dev-'+str(os.getpid())
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(output);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=dict(cfg,evaluation_only_policy_role='selected-BC-only-B1-full-story-development',
        evaluation_checkpoint_sha256=plan['checkpoint_sha256'],workers=20,PPO_training_active=False)
    write(output/'study_config.json',cfg)
    destination=output/'full_story_BC'
    def publish(row):
        write(output/'study_status.json',dict(status='validation',phase='BC_only_full_story_development',
            current_run=str(destination),updated_utc=utc_now(),current_progress=dict(status='validation',
                stage='development',workers=20,device='cuda:0',actual_decisions=0,target_decisions=0,
                optimizer_updates=0,evaluation_progress=row)))
    def stop():
        if (output/'STOP').exists():raise RuntimeError('Operator STOP;preserve all complete BC results')
    publish(dict(completed=0,target=20))
    from register_b1_goal_phase import register
    write(output/'panel_registration.json',register(output,plan['panel_job_id'],'evaluation',
        control_root=Path(plan['control_center_root'])))
    model=model_from_config(**bc['model_config']).cuda();model.load_state_dict(bc['model'],strict=True)
    if any(not torch.equal(v.detach().cpu(),bc['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Actual BC native-inference weights differ')
    results=evaluate(torch,model,make_env_kwargs(cfg,destination,20,'development'),destination,
        episodes=20,seed=plan['seed'],check_stop=stop,summary=development_summary,progress=publish,
        native_split='development',episode_protocol=episode_protocol(cfg),action_selection='greedy',
        trace_config=dict(checkpoint_sha256=plan['checkpoint_sha256'],history=128))
    if [r['native_seed'] for r in results['episode_results']]!=seeds:
        raise ValueError('Executed BC development cohort differs')
    if any(not torch.equal(v.detach().cpu(),bc['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Frozen BC policy changed during evaluation')
    report=dict(results,schema='B1-BC-only-frozen-full-story-development-v1',status='completed',utc=utc_now(),
        neural_model='single B1 CNN+object/spatial attention+GRU256',checkpoint_sha256=plan['checkpoint_sha256'],
        selected_BC_epoch=result['selected_epoch'],paid_BC_Adam=result['paid_BC_optimizer_steps'],
        native_games=20,new_optimizer_updates=0,formal_test_seeds_used=False,not_teacher_demonstration=True)
    write(output/'report.json',report)
    write(output/'study_status.json',dict(status='finished',phase='completed',current_run=str(destination),
        updated_utc=utc_now(),optimizer_updates=0))
    print(json.dumps({k:report[k] for k in ('episodes','clears','mean_misses','mean_bombs_used')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--qualify-only',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException:
        if getattr(a,'created',None):write(a.created/'failure.json',dict(traceback=traceback.format_exc(),
            optimizer_updates=0,no_automatic_restart=True))
        raise
