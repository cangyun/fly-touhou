"""Frozen original Lunatic six-stage evaluation of the existing compact SNN."""
from pathlib import Path
import argparse,os,sys,time,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','connectome','environment-config','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--selection-modes',nargs='+',choices=('joint_mode','family_mode','sample'))
    p.add_argument('--evaluation-seed',type=int,default=2027432926)
    p.add_argument('--split',choices=('development','test'),default='development')
    p.add_argument('--policy-seed',type=int,default=2027432927)
    p.add_argument('--sensation-audit-rollout',type=Path,
        help='Read-only task-signal/DN probe before native evaluation;use a matching behavior checkpoint')
    p.add_argument('--risk-protection-transition',action='store_true',
        help='Restore finite neural risk costs as observed invulnerability approaches expiry')
    p.add_argument('--forecast-native-arming',action='store_true',
        help='Include observed pending native bullet arming in the existing risk hints;keep uncertainty flags')
    p.add_argument('--protected-exit-forecast',action='store_true',
        help='Use the opt-in measured post-protection sensory semantics;inherit checkpoint when omitted')
    p.add_argument('--primary-attack-target',action='store_true',
        help='Use the observed primary Boss identity for existing attack-target features')
    p.add_argument('--idle-prior-scale',type=float,
        help='Movement-only fixed-idle preference scale[0,1];preserve Bomb probability;inherit checkpoint by default')
    p.add_argument('--compare-idle-prior-scales',type=float,nargs=2,
        help='Sequential paired development games on one checkpoint;first scale must match the checkpoint')
    p.add_argument('--risk-base-input-soft-clip',type=float,
        help='Bound only out-of-calibration DN extrapolation in the existing linear risk readout')
    p.add_argument('--bomb-urgency-gain',type=float,
        help='Frozen readout ablation;keep the learned Bomb head and all brain weights')
    p.add_argument('--neural-risk-scale',type=float,
        help='Frozen neural movement-risk composition ablation')
    p.add_argument('--allow-owned-game-controller',nargs=2,action='append',default=[],
        metavar=('PID','CREATE_TIME'),help='Coexist with this owned training job;never stop its games')
    p.add_argument('--record-source-hashes',action='store_true',
        help='Optional legacy hashes;disabled for routine native evaluations')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    import math
    if a.bomb_urgency_gain is not None and (not math.isfinite(a.bomb_urgency_gain) or a.bomb_urgency_gain<0):p.error('Finite nonnegative urgency gain required')
    if a.neural_risk_scale is not None and not 1<=a.neural_risk_scale<=16:p.error('Neural risk scale[1,16] required')
    if a.risk_base_input_soft_clip is not None and not 1<=a.risk_base_input_soft_clip<=1000:p.error('Risk baseline soft limit[1,1000] required')
    if a.selection_modes and len(set(a.selection_modes))!=len(a.selection_modes):p.error('Selection modes must be unique')
    if a.compare_idle_prior_scales:
        if a.idle_prior_scale is not None:p.error('Use either one idle-prior override or a paired contrast')
        if a.split!='development':p.error('Prior selection uses development episodes only')
        if a.selection_modes is not None and len(a.selection_modes)!=1:p.error('Paired prior comparison requires one decoder')
        from fly_touhou.compact_story_idle_prior import checked_idle_scale
        try:
            scales=[checked_idle_scale(x) for x in a.compare_idle_prior_scales]
            if scales[0]==scales[1]:raise ValueError('Two distinct prior scales required')
        except ValueError as error:p.error(str(error))
    if not a.execute:print(dict(status='prepared',full_game=True,episodes_per_mode=a.episodes,
        selection_modes=a.selection_modes,compare_idle_prior_scales=a.compare_idle_prior_scales,
        checkpoint_scale_verified=False));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cfg=read(a.environment_config);q=psutil.Process();cpus=worker_cpu_pool(q.cpu_affinity())
    from train_compact_story_ppo import check_other_native_games
    check_other_native_games(psutil,Path(cfg['game_exe']).name,a.allow_owned_game_controller)
    q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-full-SNN-'+str(q.pid))
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    def digest(path):
        if not a.record_source_hashes:return None
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
        return h.hexdigest()
    launch=dict(schema='candidate-SNN-full-story-baseline-v1',PID=q.pid,create_time=q.create_time(),argv=sys.argv,
        checkpoint=str(a.checkpoint),checkpoint_SHA256=digest(a.checkpoint),CUDA_started=False,
        selection_modes=a.selection_modes,episodes_per_mode=a.episodes,training_updates=0,evaluation_split=a.split,
        compare_idle_prior_scales=a.compare_idle_prior_scales)
    write_json(a.output/'launch.json',launch)
    write_json(a.output/'source-pins.json',{str(n):digest(n) for n in (
        Path(__file__),ROOT/'src/fly_touhou/compact_full_story.py',ROOT/'src/fly_touhou/candidate_spiking_model.py',
        ROOT/'src/fly_touhou/compact_story_model.py',ROOT/'src/fly_touhou/compact_story_action_selection.py',
        ROOT/'src/fly_touhou/compact_story_sensation_audit.py',ROOT/'src/fly_touhou/compact_story_contact_evidence.py',
        ROOT/'src/fly_touhou/compact_spiking_sensation.py',ROOT/'src/fly_touhou/compact_story_diagnostics.py',
        ROOT/'src/fly_touhou/compact_growth_forecast.py',ROOT/'src/fly_touhou/dynamic_planner.py',
        ROOT/'src/fly_touhou/compact_protected_exit_candidate.py',ROOT/'src/fly_touhou/compact_protection_forecast.py',
        ROOT/'src/fly_touhou/compact_story_idle_prior.py',
        ROOT/'src/fly_touhou/compact_story_paired_evaluation.py',
        ROOT/'src/fly_touhou/dn_residual_risk.py',ROOT/'src/fly_touhou/compact_story_risk_composition.py',
        ROOT/'src/fly_touhou/compact_story_sensation.py',ROOT/'src/fly_touhou/th14_completion_v3.py')})
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>14400:raise TimeoutError('Finite full-story evaluation wall budget')
    native_temp=None;native_closed=False
    try:
        import torch
        from fly_touhou.candidate_spiking_model import build_compact_model,restore_parameters
        from fly_touhou.compact_full_story import evaluate,apply_sensor_patches
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        native_temp=configure_native_temp(a.output)
        point=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if point['schema']=='full-connectome-compact129-spiking-story-PPO-v1':
            from fly_touhou.compact_story_model import from_story_checkpoint
            model=from_story_checkpoint(a.connectome,point).eval()
            bomb_policy='learned source19 Bomb;frozen full-story policy'
        else:model=restore_parameters(build_compact_model(a.connectome,seed=point['seed']),point).eval()
        if point['schema']!='full-connectome-compact129-spiking-story-PPO-v1':bomb_policy='existing frozen NoBomb;baseline only'
        del point
        launch['frozen_readout_overrides']={}
        if a.bomb_urgency_gain is not None:
            launch['frozen_readout_overrides']['bomb_urgency_gain']=dict(
                before=float(model.bomb_urgency_gain.detach()),after=a.bomb_urgency_gain)
            with torch.no_grad():model.bomb_urgency_gain.fill_(a.bomb_urgency_gain)
        if a.neural_risk_scale is not None:
            launch['frozen_readout_overrides']['neural_risk_scale']=dict(
                before=model.neural_risk_scale,after=a.neural_risk_scale)
            model.neural_risk_scale=a.neural_risk_scale
        # A patched environment is an explicit request to apply its sensor
        # semantics even when evaluating an older checkpoint. Otherwise the
        # frozen model's stored flags are preserved as before.
        launch['sensor_patches']=apply_sensor_patches(model,cfg)
        if a.risk_protection_transition:
            if not (getattr(model,'phase_valid_control',False) and getattr(model,'soft_risk_composition',False)):
                raise ValueError('Protection transition requires the existing finite-risk full-story policy')
            model.protection_aware_risk=True
        if a.forecast_native_arming:
            if getattr(model,'input_schema',None)!='connectome-compact129-full-story-v1':raise ValueError('Compact129 sensor required')
            model.anticipate_point_arming=True
        if a.primary_attack_target:model.primary_attack_target=True
        if a.protected_exit_forecast:model.protected_exit_forecast=True
        if a.idle_prior_scale is not None:
            from fly_touhou.compact_story_idle_prior import checked_idle_scale
            model.idle_prior_scale=checked_idle_scale(a.idle_prior_scale)
        if a.risk_base_input_soft_clip is not None:model.risk.base_input_soft_clip=a.risk_base_input_soft_clip
        if a.selection_modes is None:a.selection_modes=[getattr(model,'full_story_action_selection','joint_mode')]
        from fly_touhou.compact_story_paired_evaluation import variants,paired_report
        evaluation_variants=variants(a.selection_modes,getattr(model,'idle_prior_scale',1.),a.compare_idle_prior_scales)
        launch['evaluation_variants']=evaluation_variants
        launch['selection_modes']=a.selection_modes
        cfg['compact_forecast_native_arming']=getattr(model,'anticipate_point_arming',False)
        cfg['compact_forecast_growth_motion']=getattr(model,'forecast_growth_motion',False)
        cfg['compact_native_motion_forecast']=getattr(model,'native_motion_forecast',False)
        cfg['compact_native_focus_gate']=getattr(model,'native_focus_gate',False)
        cfg['compact_unknown_laser_player_motion']=getattr(model,'unknown_laser_player_motion',False)
        cfg['compact_native_curve_laser_geometry']=getattr(model,'native_curve_laser_geometry',False)
        cfg['compact_protected_exit_forecast']=getattr(model,'protected_exit_forecast',False)
        cfg['compact_primary_attack_target']=getattr(model,'primary_attack_target',False)
        write_json(a.output/'environment-config.json',cfg)
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),
            observation_features=129 if getattr(model,'input_schema',None)=='connectome-compact129-full-story-v1' else 111,
            neural_input_per_candidate=47 if getattr(model,'input_schema',None)=='connectome-compact129-full-story-v1' else 8,
            full_story=True,Bomb_policy=bomb_policy,protection_aware_risk=getattr(model,'protection_aware_risk',False),
            anticipate_point_arming=getattr(model,'anticipate_point_arming',False),
            forecast_growth_motion=getattr(model,'forecast_growth_motion',False),
            native_motion_forecast=getattr(model,'native_motion_forecast',False),
            protected_exit_forecast=getattr(model,'protected_exit_forecast',False),
            primary_attack_target=getattr(model,'primary_attack_target',False),
            neural_risk_scale=getattr(model,'neural_risk_scale',1.),
            idle_prior_scale=getattr(model,'idle_prior_scale',1.),
            risk_readout_kind=getattr(model.risk,'readout_kind','legacy'),
            risk_base_input_soft_clip=getattr(model.risk,'base_input_soft_clip',None))
        write_json(a.output/'launch.json',launch)
        if a.sensation_audit_rollout:
            from fly_touhou.compact_story_sensation_audit import audit
            tape=torch.load(a.sensation_audit_rollout,map_location='cpu',weights_only=False)
            result=audit(model,tape,scenes=16,capture=lambda trace:torch.save(trace,a.output/'sensation-DN-signals.pt'))
            result['source_rollout']=str(a.sensation_audit_rollout)
            write_json(a.output/'sensation-audit.json',result)
            del tape
            import gc
            gc.collect();torch.cuda.empty_cache()
        def progress(info):write_json(a.output/'progress.json',dict(phase='frozen_full_story',variant=current['label'],evaluation=info))
        reports={}
        for current in evaluation_variants:
            model.idle_prior_scale=current['idle_prior_scale']
            reports[current['label']]=evaluate(model,cfg,a.output/current['folder'],episodes=a.episodes,worker_cpus=cpus,guard=guard,progress=progress,
                seed=a.evaluation_seed,policy_seed=a.policy_seed,action_selection=current['mode'],evaluation_split=a.split)
            write_json(a.output/'completed-modes.json',dict(completed=list(reports),planned=[v['label'] for v in evaluation_variants]))
        if a.compare_idle_prior_scales:
            comparison=paired_report(*(reports[v['label']] for v in evaluation_variants))
            comparison.update(checkpoint=str(a.checkpoint),checkpoint_SHA256=launch['checkpoint_SHA256'],variants=evaluation_variants)
            write_json(a.output/'paired-idle-comparison.json',comparison)
        summary=dict(status='completed_full_story_decoding_evaluation',episodes_per_mode=a.episodes,
            modes={k:dict(clears=v['clears'],stage_counts=v['stage_counts'],episodes=v['episodes']) for k,v in reports.items()},
            Bomb_policy=bomb_policy,goal_complete_claimed=False,optimizer_updates=0,evaluation_split=a.split,
            scope='Frozen original full games on the declared seed partition;not changing-policy training statistics')
        if len(reports)==1:
            result=next(iter(reports.values()));summary.update(clears=result['clears'],episodes=a.episodes,stage_counts=result['stage_counts'])
        write_json(a.output/'result.json',summary);native_closed=True
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise
    finally:
        if native_closed and native_temp is not None:
            from fly_touhou.thrl_job_paths import cleanup_native_temp
            write_json(a.output/'native-temp-cleanup.json',cleanup_native_temp(native_temp))


if __name__=='__main__':main()
