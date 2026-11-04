"""One declared B1 CUDA experiment through the existing native PPO learner.

It transfers either sealed BC weights with fresh PPO Adam or an explicitly
pinned parent PPO model and optimizer for a new correction experiment.
Native full-story episodes,19 autonomous actions and v3 reward are preserved.
"""
from pathlib import Path
import argparse, json, os, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(a,*,model_factory=None):
    from dataclasses import asdict
    from train_thrl_branch import verify_sources
    from train_thrl_native import train_trial,StopRequested,validate_evaluation_schedule
    from fly_touhou.thrl_branching import reuse_teacher_initialization,candidate_trial_config
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import learning_overrides
    import psutil,torch
    cfg=read(a.config)
    validate_evaluation_schedule(cfg)
    if cfg.get('branch')!='B1' or cfg.get('workers')!=30:
        raise ValueError('This experiment is exclusively B1 with30 original-game workers')
    parent=cfg.get('parent_PPO_policy');resume=cfg.get('baseline_resume');reused=False
    if resume:
        if cfg.get('teacher_pretraining') or cfg.get('reset_Adam_after_BC'):
            raise ValueError('PPO recovery cannot repeat BC or reset Adam')
        payload=torch.load(resume['checkpoint'],map_location='cpu',weights_only=False)
        if (payload.get('optimizer_transaction_complete') is not True
                or payload.get('candidate_optimized_decisions')!=resume['optimized_decisions']
                or payload['model_config']['input_schema']!=cfg['input_schema']):
            raise ValueError('Recovery needs a sealed complete optimizer transaction and the same candidate/input')
        from train_thrl_native import digest
        if digest(resume['checkpoint'])!=resume['sha256']:raise ValueError('Recovery checkpoint hash differs')
        cfg.update(warm_start_checkpoint=resume['checkpoint'],warm_start_sha256=resume['sha256'],
            initial_avoidance_development=False,initial_development_not_repeated_during_recovery=True)
        del payload
    elif parent:
        if cfg.get('teacher_pretraining') or cfg.get('reset_Adam_after_BC') or cfg.get('baseline_resume'):
            raise ValueError('A declared new on-policy correction experiment must preserve its parent PPO optimizer')
        payload=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
        if (payload.get('optimizer_transaction_complete') is False
                or payload.get('candidate_optimized_decisions')!=parent['optimized_decisions']
                or payload['model_config']['input_schema']!=cfg['input_schema']):
            raise ValueError('Parent PPO identity, transaction or input differs')
        from train_thrl_native import digest
        if digest(parent['checkpoint'])!=parent['sha256']:raise ValueError('Parent PPO checkpoint hash differs')
        cfg.update(warm_start_checkpoint=parent['checkpoint'],warm_start_sha256=parent['sha256'])
        del payload
    else:
        cfg,reused=reuse_teacher_initialization(cfg)
        if not reused or not cfg.get('reset_Adam_after_BC'):
            raise ValueError('A sealed BC transfer and an explicit fresh PPO optimizer are required')
    verify_sources(cfg)
    from fly_touhou.thrl_native_read_qualification import verify as verify_item_reader
    reading=verify_item_reader(cfg,ROOT)
    if reading is not None:cfg['qualified_native_item_reading']=reading
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
    if cfg['trial_decisions']%cfg['rollout_total_decisions'] or any(n%cfg['rollout_total_decisions'] for n in cfg['development_decisions']):
        raise ValueError('All decision nodes must be complete per-world rollouts')
    parameters={**asdict(PPOConfig()),**cfg['parameters'],**learning_overrides(cfg)}
    PPOConfig(**parameters)
    torch.set_num_threads(cfg.get('torch_threads',2))
    a.output.mkdir(parents=True,exist_ok=False)
    configure_native_temp(a.output)
    write(a.output/'study_config.json',cfg)
    write(a.output/'effective_study_config.json',cfg)
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now()))
    if cfg.get('goal_panel_registration'):
        from register_b1_goal_phase import register
        panel=cfg['goal_panel_registration']
        registration=register(a.output,panel['job_id'],'training',control_root=Path(panel['control_root']))
        write(a.output/'panel_registration.json',registration)
    started=time.monotonic()
    def progress(row):
        current=a.output/('initial_avoidance_development' if row.get('stage')=='initial_avoidance_development' else 'trial_000_both')
        write(a.output/'study_status.json',dict(status=row.get('status','training'),
            current_run=str(current),current_progress=row,
            updated_utc=utc_now(),wall_seconds=time.monotonic()-started,planned_trials=1,
            device='cuda:0',gpu=torch.cuda.get_device_name(0),teacher_pretraining_reused=reused,
            optimizer_initialization=('recovered PPO Adam and RNG; same declared correction objective' if resume else
                'inherited PPO Adam; new declared correction objective' if parent else 'fresh PPO Adam, BC weights retained')))
    try:
        if cfg.get('initial_avoidance_development'):
            from train_thrl_native import make_env_kwargs
            from fly_touhou.thrl_model import MOActorCritic
            from fly_touhou.thrl_avoidance_development import evaluate as initial_evaluate
            p=torch.load(cfg['warm_start_checkpoint'],map_location='cpu',weights_only=False)
            model=(model_factory or MOActorCritic)(**p['model_config']).cuda();model.load_state_dict(p['model']);del p
            quota=cfg['development_episodes'];destination=a.output/'initial_avoidance_development'
            count=min(quota,cfg.get('avoidance_development_workers',6),30)
            kwargs=make_env_kwargs(cfg,destination,count,'development')
            for row in kwargs:row['probe_max_frames']=cfg.get('avoidance_development_max_frames',4096)
            def check_initial():
                if (a.output/'STOP').exists():raise StopRequested('Operator STOP before correction training')
            initial_evaluate(torch,model,kwargs,destination,episodes=quota,
                seed=cfg['avoidance_development_seed'],check_stop=check_initial,
                progress=lambda row:progress(dict(status='initial_avoidance_development',stage='initial_avoidance_development',evaluation_progress=row)))
            from train_thrl_native import digest
            cfg['sealed_avoidance_baseline']=dict(format='B1-no-Bomb-first-collision-development-v1',
                report=str(destination/'report.json'),report_sha256=digest(destination/'report.json'),
                checkpoint=cfg['warm_start_checkpoint'],checkpoint_sha256=cfg['warm_start_sha256'],
                input_schema=cfg['input_schema'],episodes=quota,seed=cfg['avoidance_development_seed'],
                max_frames=cfg.get('avoidance_development_max_frames',4096))
            write(a.output/'effective_study_config.json',cfg)
            del model;torch.cuda.empty_cache()
        trial_config=candidate_trial_config(cfg,0)
        result=train_trial(torch,trial_config,a.output,a.output/'trial_000_both',parameters,
            trial=None,study_started=started,on_progress=progress,attention='both',model_factory=model_factory)
        write(a.output/'study_result.json',dict(status='finished',experiment=cfg.get('experiment','B1-avoidance-PPO-v1'),
            training=result,formal_acceptance=False,BC_reused=reused,PPO_optimizer_fresh=not bool(parent or resume)))
        print(json.dumps(dict(status='finished',decisions=result['decisions'],output=str(a.output))))
    except StopRequested as e:
        write(a.output/'study_result.json',dict(status='operator_or_budget_stop',reason=str(e)))
    except BaseException:
        write(a.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);run(p.parse_args())
