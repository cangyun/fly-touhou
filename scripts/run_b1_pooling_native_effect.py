"""One finite frozen operator-effect test after the existing140 games close.

Reuse the pinned completed20-game legacy baseline; execute20 new matched-seed
games with identical learned tensors and explicitly versioned risk pooling.
No training, teacher, stage starts, external frame horizon, or desktop input.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_report(plan):
    path=Path(plan['baseline_report'])
    if sha(path)!=plan['baseline_report_sha256']:raise ValueError('Pinned complete baseline report changed')
    report=read(path);rows=report['episode_results']
    if (report.get('episodes')!=20 or len(rows)!=20 or report.get('original_game_rng_controlled') is not True
            or report.get('native_seed_split')!='test' or report.get('training_mode')!='full-story-v1'
            or report.get('full_game_trace_capture',{}).get('model_checkpoint_sha256')!=plan['original_checkpoint_sha256']
            or any(r.get('outcome') not in ('clear','gameover') for r in rows)
            or [r.get('native_seed') for r in rows]!=plan['expected_native_seeds']
            or read(path.parent/'native_cleanup.json').get('close_errors')!=[]):
        raise ValueError('Complete closed20-game native baseline with exact seed provenance required')
    return report


def run(a):
    import psutil
    from replay_b1_native_traces import replay_dependency
    plan=read(a.plan);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now())
    write(out/'launch.json',identity)
    def stop():
        if (out/'STOP').exists():raise RuntimeError('Operator requested finite pooling evaluation STOP')
    def publish(phase,current=None,detail=None):
        write(out/'study_status.json',dict(status='finished' if phase=='completed' else 'validation',phase=phase,
            current_run=str(current or out),updated_utc=utc_now(),current_progress=dict(status='validation',
                stage='validation',workers=20,device='cuda:0',actual_decisions=0,target_decisions=0,
                evaluation_progress=detail,optimizer_updates=0),native_instances_max=0 if phase.startswith('waiting') else 20))
    phase='waiting_for_existing140_and_cleanup';started=time.monotonic()
    try:
        baseline=baseline_report(plan)
        while True:
            stop();state=replay_dependency(Path(plan['after_control']),Path(plan['after_output']),plan['after_identity'])
            if state['phase']=='failed':raise RuntimeError(state['reason'])
            if state['phase']=='ready':break
            if time.monotonic()-started>4*3600:raise RuntimeError('Finite predecessor wait expired;no restart')
            publish(phase,detail=state);time.sleep(10)
        # Verify both predecessor natural-game pools explicitly closed.
        for label in ('initial','final'):
            if read(Path(plan['after_output'])/('full_story_'+label)/'native_cleanup.json').get('close_errors')!=[]:
                raise RuntimeError('Predecessor natural pool did not close')
        while any('th14' in (p.info['name'] or '').lower() for p in psutil.process_iter(['name'])):
            stop();publish('waiting_for_no_other_original_worlds',detail=dict(native_worlds_started=0))
            if time.monotonic()-started>4*3600:raise RuntimeError('Other original worlds remain live;no restart')
            time.sleep(10)
        import torch
        from fly_touhou.thrl_pair_policy import model_from_config
        from fly_touhou.thrl_job_paths import configure_native_temp
        from evaluate_b1_full_story_comparison import model_digest,policy_identity,paired_summary
        from evaluate_thrl_avoidance_probe import checkpoint_copy
        from train_thrl_native import evaluate
        from register_b1_goal_phase import register
        configure_native_temp(out)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:
            raise RuntimeError('CUDA evaluation reserve unavailable;no new original worlds started')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-pooling-v2-'+str(os.getpid())
        if sha(plan['checkpoint'])!=plan['checkpoint_sha256'] or sha(plan['original_checkpoint'])!=plan['original_checkpoint_sha256']:
            raise ValueError('Frozen checkpoint pin differs')
        checkpoint_copy(Path(plan['checkpoint']),out/'policy.pt')
        p=torch.load(out/'policy.pt',map_location='cpu',weights_only=False)
        original=torch.load(plan['original_checkpoint'],map_location='cpu',weights_only=False)
        tensors=model_digest(p['model'])
        if tensors!=model_digest(original['model']):raise ValueError('Pooling comparison changed learned tensors')
        semantics=policy_identity(p['model'],p['model_config'])
        if semantics==policy_identity(original['model'],original['model_config']):raise ValueError('No declared operator difference')
        wrapper=p['model_config'].get('pair_risk_wrapper',{})
        if wrapper.get('aggregation')!='worst-known-margin-v2':raise ValueError('Wrong declared pooling operator')
        model=model_from_config(**p['model_config']).cuda();model.load_state_dict(p['model'],strict=True);model.eval()
        if sha(plan['environment_config'])!=plan['environment_config_sha256']:raise ValueError('Pinned native environment configuration changed')
        config=read(plan['environment_config']);runtime=Path(config['source_runtime'])
        for name in ('thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py'):
            if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
                raise ValueError('Qualified native execution changed: '+name)
        if (ROOT/'native/th14_fake_d3d9.dll').read_bytes()!=(runtime/'native/th14_fake_d3d9.dll').read_bytes():
            raise ValueError('Qualified software COM DLL changed')
        write(out/'study_config.json',dict(config,workers=20,diagnostic_only=True,PPO_training_active=False))
        config.update(full_story_evaluation_backend='full-story-ready-parallel-v1',full_story_evaluation_workers=20,
            full_story_trace=dict(checkpoint_sha256=plan['checkpoint_sha256'],history=128))
        write(out/'comparison_inputs.json',dict(original_checkpoint_sha256=plan['original_checkpoint_sha256'],
            modified_checkpoint_sha256=plan['checkpoint_sha256'],model_tensors_sha256=tensors,
            model_config=p['model_config'],policy_identity_sha256=semantics,
            baseline_report=plan['baseline_report'],baseline_report_sha256=plan['baseline_report_sha256'],
            input_reward_native_backend_unchanged=True,optimizer_updates=0))
        del p,original
        register(out,plan['panel_job_id'],'evaluation',control_root=Path(plan['control_center_root']))
        phase='full_story_pooling_v2';dest=out/phase;publish(phase,dest)
        result=evaluate(torch,model,config,dest,episodes=20,seed=plan['seed'],native_split='test',check_stop=stop,
            progress=lambda row:publish(phase,dest,row))
        if (model_digest(model.state_dict())!=tensors or [r['native_seed'] for r in result['episode_results']]!=plan['expected_native_seeds']):
            raise ValueError('Frozen weights or original RNG cohort changed')
        if read(dest/'native_cleanup.json')['close_errors']!=[]:raise RuntimeError('Pooling test pool not closed')
        summary=paired_summary({'parent':baseline,'pooling_v2':result},20)
        write(out/'report.json',dict(schema='B1-known-margin-pooling-frozen-native-effect-v1',status='completed',utc=utc_now(),
            summary=summary,source_baseline_report=plan['baseline_report'],new_report=str(dest/'report.json'),
            newly_executed_native_games=20,reused_pinned_baseline_games=20,optimizer_updates=0,teacher_actions=0,
            original_initial_RNG_matched=True,same_future_hazards_guaranteed=False,
            matched_existing_cohort_is_engineering_test_not_new_final_acceptance=True,
            full_game_trace_capture=result.get('full_game_trace_capture'),goal_complete=False))
        phase='completed';publish(phase)
    except BaseException:
        write(out/'failure.json',dict(utc=utc_now(),phase=phase,traceback=traceback.format_exc(),no_automatic_restart=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
