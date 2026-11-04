"""Frozen B1 full-game comparison, after the existing teacher pool is closed.

Uses the qualified complete-game evaluator:19 autonomous actions, native lives,
no continues and no frame horizon. No optimizer, teacher or desktop input.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def dependency_finished(control,output):
    try:status=read(control/'status.json')
    except FileNotFoundError:return False
    if status.get('status') in ('failed','launcher-error'):raise RuntimeError('Required teacher evaluation failed')
    if status.get('status')!='finished':return False
    if status.get('exit_code')!=0:raise RuntimeError('Teacher evaluation exit is not successful')
    result=read(output/'report.json');cleanup=read(output/'cleanup.json')
    if result.get('status')!='completed' or cleanup!=dict(closed=True,close_errors=[]):
        raise RuntimeError('Teacher evaluation has no sealed successful result and cleanup')
    return True


def closed_pooling_dependency(plan,*,is_alive=None):
    """Wait for the exact finite predecessor, not a recycled launcher PID."""
    if is_alive is None:
        from fly_touhou.thrl_control import alive
        is_alive=alive
    output=Path(plan['after_output']);status_path=Path(plan['after_control'])/'status.json'
    status=read(status_path) if status_path.exists() else {}
    if status.get('status') in ('failed','launcher-error') or (output/'failure.json').exists():
        raise RuntimeError('Required pooling evaluation failed;no automatic restart')
    if status.get('status')!='finished':return False
    if status.get('exit_code')!=0:raise RuntimeError('Required pooling evaluation exited unsuccessfully')
    result=read(output/'report.json')
    if (result.get('schema')!='B1-known-margin-pooling-frozen-native-effect-v1' or result.get('status')!='completed'
            or result.get('newly_executed_native_games')!=20 or result.get('optimizer_updates')!=0
            or Path(result.get('new_report','')).resolve()!=(output/'full_story_pooling_v2/report.json').resolve()
            or read(output/'full_story_pooling_v2/native_cleanup.json').get('close_errors')!=[]):
        raise RuntimeError('Pooling predecessor has no sealed complete20-game result and closed pool')
    return not is_alive(plan['after_identity'])


def reused_pooling_baseline(plan):
    """Seal completed predecessor data after exit; never reuse partial games."""
    from run_b1_pooling_native_effect import baseline_report,sha
    path=Path(plan['after_output'])/'full_story_pooling_v2/report.json'
    receipt=dict(baseline_report=str(path),baseline_report_sha256=sha(path),
        original_checkpoint_sha256=plan['baseline_checkpoint_sha256'],expected_native_seeds=plan['expected_native_seeds'])
    report=baseline_report(receipt)
    if report.get('action_selection','sampled')!='sampled':raise ValueError('Expected sampled predecessor baseline')
    return report,receipt


def model_digest(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((str(x.dtype),tuple(x.shape))).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def policy_identity(state,configuration,action_selection='sampled'):
    """Equal tensors do not imply equal inference when risk operators differ."""
    if action_selection not in ('sampled','greedy'):raise ValueError('Unknown full-game action selection')
    encoded=json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    payload=model_digest(state).encode('ascii')+b'\n'+encoded
    # Keep existing sampled policy identities stable for sealed reports.
    if action_selection!='sampled':payload+=b'\naction_selection='+action_selection.encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def paired_summary(reports,episodes,baseline='parent'):
    import numpy as np
    base=reports[baseline]['episode_results'];answer={}
    if len(base)!=episodes:raise ValueError('Incomplete parent games')
    for name,report in reports.items():
        rows=report['episode_results']
        if len(rows)!=episodes or any((x['evaluation_game'],x['native_seed'])!=(y['evaluation_game'],y['native_seed']) for x,y in zip(rows,base)):
            raise ValueError('Full-game policies require complete matched native seeds')
        stage=np.array([max(x['stage_history']) for x in rows]);prior=np.array([max(x['stage_history']) for x in base])
        answer[name]=dict(episodes=episodes,clears=sum(x['outcome']=='clear' for x in rows),
            reach_stage_counts={str(k):int((stage>=k).sum()) for k in range(1,7)},
            mean_native_frames=float(np.mean([x['native_frames'] for x in rows])),
            mean_Bombs=float(np.mean([x['bombs_used'] for x in rows])),
            stage_longer_than_parent=int((stage>prior).sum()),stage_equal=int((stage==prior).sum()),stage_lower=int((stage<prior).sum()),
            aliases_existing_policy=report.get('aliases_existing_policy'),
            full_game_lifetime_is_not_pure_avoidance=True)
    return answer


def run(a, *, model_factory=None):
    import psutil
    from fly_touhou.thrl_job_paths import configure_native_temp
    plan=read(a.plan);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);configure_native_temp(out)
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time())
    write(out/'identity.json',identity);write(out/'launch.json',identity)
    def publish(phase,**kw):
        write(out/'progress.json',dict(phase=phase,utc=utc_now(),**kw))
        if plan.get('panel_job_id'):
            write(out/'study_status.json',dict(status='finished' if phase=='completed' else 'validation',phase=phase,
                current_run=str(out/kw['policy']) if 'policy' in kw else str(out),updated_utc=utc_now(),
                current_progress=dict(status='validation',stage='validation',workers=plan.get('workers',20),device='cuda:0',
                    actual_decisions=0,target_decisions=0,optimizer_updates=0,evaluation_progress=kw.get('evaluation'))))
    def stop():
        if (out/'STOP').exists():raise RuntimeError('Operator requested full-game comparison STOP')
    try:
        if plan.get('source_runtime') and Path(plan['source_runtime']).resolve()!=ROOT.resolve():
            raise ValueError('Frozen evaluator runtime differs from the declared plan')
        if plan.get('source_manifest_sha256') and hashlib.sha256((ROOT/'source_manifest.json').read_bytes()).hexdigest()!=plan['source_manifest_sha256']:
            raise ValueError('Frozen evaluator source manifest changed')
        closed_pooling=plan.get('after_kind')=='closed-pooling-effect-v1';wait_started=time.monotonic()
        if plan.get('after_kind','teacher') not in ('teacher','closed-pooling-effect-v1'):raise ValueError('Unknown finite predecessor kind')
        while not (closed_pooling_dependency(plan) if closed_pooling else dependency_finished(Path(plan['after_control']),Path(plan['after_output']))):
            stop();publish('waiting_for_predecessor_cleanup',native_instances=0)
            if time.monotonic()-wait_started>4*3600:raise RuntimeError('Finite predecessor wait expired;no restart')
            time.sleep(10)
        stop()
        if closed_pooling and any('th14' in (p.info['name'] or '').lower() for p in psutil.process_iter(['name'])):
            raise RuntimeError('Other original worlds still live after predecessor closure;no new pool started')
        import torch
        from fly_touhou.thrl_pair_policy import model_from_config
        factory=model_factory or model_from_config
        from fly_touhou.thrl_full_story_evaluation import evaluate
        from fly_touhou.thrl_training_mode import episode_protocol
        from train_thrl_native import make_env_kwargs,development_summary
        from evaluate_thrl_avoidance_probe import checkpoint_copy
        if not torch.cuda.is_available():raise RuntimeError('CUDA inference required; no CPU fallback')
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA reserve unavailable before native launch')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-b1-fullgames-'+str(os.getpid())
        if plan.get('environment_config'):
            cfg_path=Path(plan['environment_config'])
            if hashlib.sha256(cfg_path.read_bytes()).hexdigest()!=plan['environment_config_sha256']:
                raise ValueError('Pinned original environment configuration changed')
        else:cfg_path=Path(plan['training_output'])/'trial_000_both/config.json'
        cfg=read(cfg_path)
        runtime=Path(cfg['source_runtime'])
        for name in ['thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py']:
            if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
                raise ValueError('Qualified native execution changed: '+name)
        if (ROOT/'native/th14_fake_d3d9.dll').read_bytes()!=(runtime/'native/th14_fake_d3d9.dll').read_bytes():
            raise ValueError('Qualified native COM library changed')
        episodes=plan.get('episodes',20);workers=min(plan.get('workers',30),episodes)
        if not 2<=episodes<=20 or not 1<=workers<=30:raise ValueError('Bounded comparison required')
        reports={};pins={};same_weights={};native_games=0
        (out/'pins').mkdir()
        if plan.get('reuse_pooling_baseline'):
            if not closed_pooling:raise ValueError('Pooling baseline reuse requires its exact closed predecessor')
            reports['parent'],receipt=reused_pooling_baseline(plan)
            write(out/'reused_baseline_receipt.json',dict(receipt,reused_native_games=episodes,newly_executed_native_games=0))
        write(out/'study_config.json',dict(cfg,workers=workers,diagnostic_only=True,PPO_training_active=False))
        if plan.get('panel_job_id'):
            from register_b1_goal_phase import register
            register(out,plan['panel_job_id'],'evaluation',control_root=Path(plan['control_center_root']))
        for job in plan['policies']:
            stop();name=job['name'];target=out/'pins'/(name+'.pt')
            if name in reports:raise ValueError('Duplicate comparison policy name')
            checkpoint_sha=checkpoint_copy(Path(job['checkpoint']),target)
            if job.get('checkpoint_sha256') is not None and checkpoint_sha!=job['checkpoint_sha256']:
                raise ValueError('Pinned frozen checkpoint changed')
            if plan.get('reuse_pooling_baseline') and checkpoint_sha!=plan['baseline_checkpoint_sha256']:
                raise ValueError('Action-selection comparison changed checkpoint weights or configuration')
            p=torch.load(target,map_location='cpu',weights_only=False)
            if p.get('optimizer_transaction_complete') is False or p['model_config']['input_schema']!=cfg['input_schema']:
                raise ValueError('Partial or incompatible frozen policy')
            sha=model_digest(p['model']);pins[name]=dict(source=job['checkpoint'],checkpoint_sha256=checkpoint_sha,model_state_sha256=sha,
                candidate_optimized_decisions=p.get('candidate_optimized_decisions'),inherited_decisions=p.get('inherited_decisions',p.get('config',{}).get('inherited_decisions',0)))
            selection=job.get('action_selection',plan.get('action_selection','sampled'))
            semantic=policy_identity(p['model'],p['model_config'],selection);pins[name]['policy_identity_sha256']=semantic
            pins[name]['action_selection']=selection
            if semantic in same_weights:
                alias=same_weights[semantic];reports[name]=dict(reports[alias],aliases_existing_policy=alias)
                write(out/(name+'_alias.json'),dict(identical_frozen_parameters=True,source=alias,native_games_repeated=0));del p;continue
            model=factory(**p['model_config']).cuda();model.load_state_dict(p['model']);model.eval();del p
            destination=out/name;kwargs=make_env_kwargs(cfg,destination,workers,'test')
            if any('online_teacher_revision' in k or k.get('wall_time_limit_seconds') is not None for k in kwargs):
                raise ValueError('Full-game evaluation must have no teacher or external game horizon')
            publish('evaluating_full_games',policy=name,episodes=episodes,workers=workers)
            protocol={**episode_protocol(cfg), 'evaluation_role':'frozen normal19 original full-game comparison',
                'evaluation_native_frame_horizon':None,'evaluation_Bomb_allowed':True}
            trace_config=dict(checkpoint_sha256=checkpoint_sha,history=128) if plan.get('capture_failure_traces') else None
            reports[name]=evaluate(torch,model,kwargs,destination,episodes=episodes,seed=plan['seed'],check_stop=stop,
                summary=development_summary,native_split='test',episode_protocol=protocol,
                progress=lambda row:publish('evaluating_full_games',policy=name,evaluation=row),action_selection=selection,
                trace_config=trace_config)
            if model_digest(model.state_dict())!=sha:raise RuntimeError('Frozen policy weights changed during full-game evaluation')
            if read(destination/'native_cleanup.json')['close_errors']:raise RuntimeError('Full-game pool did not close cleanly')
            same_weights[semantic]=name;native_games+=episodes;del model;torch.cuda.empty_cache()
        summary=paired_summary(reports,episodes,plan.get('comparison_baseline','parent'))
        result=dict(status='completed',utc=utc_now(),summary=summary,pins=pins,native_games=native_games,
            native_pools_sequential=True,optimizer_updates=0,teacher_actions=0,continues=0,
            policy_actions=19,frame_horizon=None,native_seed_split='test',seed=plan['seed'],
            original_game_RNG_matched=True,formal_full_six_stage_acceptance=False,
            reused_native_games=episodes if plan.get('reuse_pooling_baseline') else 0,
            reused_baseline_report=str(Path(plan['after_output'])/'full_story_pooling_v2/report.json') if plan.get('reuse_pooling_baseline') else None,
            reports={k:(str(Path(plan['after_output'])/'full_story_pooling_v2/report.json') if k=='parent' and plan.get('reuse_pooling_baseline')
                else str(out/k/'report.json') if not r.get('aliases_existing_policy') else str(out/r['aliases_existing_policy']/'report.json')) for k,r in reports.items()})
        write(out/'report.json',result);publish('completed',native_instances=0,native_games=native_games);print(json.dumps(summary))
    except BaseException:
        write(out/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),partial_results_are_not_acceptance=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
