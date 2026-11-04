"""Full-policy CUDA PPO after complete teacher-trajectory actor warmup.

Starts only after the prior original pool has exited. Uses the unchanged
qualified trainer, full Story episodes, nineteen autonomous actions, reward,
and policy. The new artifact is an explicit transfer with fresh PPO Adam.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_transfer(torch,cfg,output):
    from fly_touhou.thrl_actor_transfer import build_transfer
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import learning_overrides
    origin=cfg['full_teacher_actor_transfer'];report_path=Path(origin['result'])
    if sha(report_path)!=origin['result_sha256']:raise ValueError('Actor fit result changed')
    result=read(report_path);checkpoint=Path(result['policy'])
    if sha(checkpoint)!=result['policy_sha256']:raise ValueError('Restored best actor checkpoint changed')
    warmup=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if Path(warmup['parent_checkpoint']).resolve()!=Path(origin['parent_checkpoint']).resolve():
        raise ValueError('Explicit ancestral policy path differs')
    if warmup['parent_sha256']!=origin['parent_sha256'] or sha(origin['parent_checkpoint'])!=origin['parent_sha256']:
        raise ValueError('Explicit ancestral policy hash differs')
    dataset=FullTeacherDataset(Path(warmup['teacher_manifest']).parent)
    if (dataset.manifest_sha256!=warmup['teacher_manifest_sha256']
            or dataset.manifest_sha256!=warmup['config']['teacher_dataset_sha256']
            or dataset.rows!=result['original_teacher_rows'] or len(dataset.records)!=30
            or result['teacher_complete_episodes']!=30
            or dataset.manifest.get('numerical_contract_fixture_only')):
        raise ValueError('Thirty complete real original-game demonstrations required')
    source=Path(warmup['config']['source_runtime'])
    if sha(source/'source_manifest.json')!=warmup['config']['source_manifest_sha256']:
        raise ValueError('Pinned actor-fit source provenance changed')
    parent=torch.load(origin['parent_checkpoint'],map_location='cpu',weights_only=False)
    model=model_from_config(**warmup['model_config'])
    parameters=PPOConfig(**{**asdict(PPOConfig()),**cfg['parameters'],**learning_overrides(cfg)})
    transfer=build_transfer(torch,model,warmup,result,parent,parameters,seed=cfg['seed'])
    target=output/'actor_to_full_policy_transfer.pt'
    temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:torch.save(transfer,temporary);os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    audit=dict(transfer['BC_to_PPO_transfer_audit'],transfer_checkpoint=str(target),
        transfer_sha256=sha(target),actor_checkpoint=str(checkpoint),actor_sha256=result['policy_sha256'],
        actor_result=str(report_path),actor_result_sha256=origin['result_sha256'],
        teacher_manifest_sha256=dataset.manifest_sha256,
        original_teacher_rows=dataset.rows,complete_native_teacher_games=30,native_worlds_started=0)
    write(output/'actor_transfer_audit.json',audit)
    configured={**cfg,'policy_model_config':warmup['model_config'],'warm_start_checkpoint':str(target),
        'warm_start_sha256':sha(target),'teacher_pretraining':True,'reset_Adam_after_BC':True,
        'teacher_pretraining_result':dict(schema='B1-full-natural-actor-transfer-for-full-PPO-v1',
            checkpoint=str(target),checkpoint_sha256=sha(target),audit=audit),
        'initialization_semantics':'Complete natural teacher actor weights;fresh full-policy PPO Adam and declared RNG;fresh original openings'}
    return configured


def run(args):
    import psutil,torch
    from train_thrl_branch import verify_sources
    from train_thrl_native import train_trial,StopRequested,validate_evaluation_schedule
    from fly_touhou.thrl_branching import candidate_trial_config
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_ppo import PPOConfig
    from fly_touhou.th14_completion_protocol import learning_overrides
    cfg=read(args.config)
    if (cfg.get('branch')!='B1' or cfg.get('workers')!=30 or not cfg.get('headless')
            or cfg.get('baseline_resume') or cfg.get('parent_PPO_policy')
            or not cfg.get('full_teacher_actor_transfer')
            or cfg.get('evaluation_after_training_shutdown') is not True
            or cfg.get('initial_avoidance_development')):
        raise ValueError('One B1 transfer,30 headless worlds,one uninterrupted run and final development only')
    validate_evaluation_schedule(cfg)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required;no CPU training fallback')
    if any(p.info['name'] and p.info['name'].lower()=='th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Prior original worlds still present;no concurrent pool launched')
    teacher_folder=Path(read(cfg['full_teacher_actor_transfer']['result'])['policy']).parent
    prior=read(teacher_folder/'launch.json')
    try:
        p=psutil.Process(prior['pid'])
        if abs(p.create_time()-prior['create_time'])<.01 and p.is_running():
            raise RuntimeError('Actor fitter still exiting;wait for its sealed result')
    except psutil.NoSuchProcess:pass
    args.output.mkdir(parents=True,exist_ok=False);args.output_created=True
    write(args.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        utc=utc_now(),CUDA_required=True,native_worlds_max=30,desktop_input=False))
    cfg=prepare_transfer(torch,cfg,args.output)
    verify_sources(cfg)
    from fly_touhou.thrl_native_read_qualification import verify as verify_item_reader
    reading=verify_item_reader(cfg,ROOT)
    if reading is not None:cfg['qualified_native_item_reading']=reading
    configure_native_temp(args.output)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-full-teacher-PPO-'+str(os.getpid())
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(cfg.get('torch_threads',2))
    parameters={**asdict(PPOConfig()),**cfg['parameters'],**learning_overrides(cfg)}
    write(args.output/'study_config.json',cfg);write(args.output/'effective_study_config.json',cfg)
    if cfg.get('goal_panel_registration'):
        from register_b1_goal_phase import register
        panel=cfg['goal_panel_registration']
        write(args.output/'panel_registration.json',register(args.output,panel['job_id'],'training',control_root=Path(panel['control_root'])))
    started=time.monotonic()
    def progress(row):
        write(args.output/'study_status.json',dict(status=row.get('status','training'),
            current_run=str(args.output/'trial_000_both'),current_progress=row,
            updated_utc=utc_now(),wall_seconds=time.monotonic()-started,planned_trials=1,
            device='cuda:0',gpu=torch.cuda.get_device_name(0),
            optimizer_initialization='fresh full-policy PPO Adam after restricted actor BC;not numerical PPO recovery'))
    try:
        result=train_trial(torch,candidate_trial_config(cfg,0),args.output,args.output/'trial_000_both',parameters,
            trial=None,study_started=started,on_progress=progress,attention='both',model_factory=model_from_config)
        write(args.output/'study_result.json',dict(status='finished',training=result,
            formal_acceptance=False,BC_reused=True,PPO_optimizer_fresh=True,full_shared_policy_optimized=True))
    except StopRequested as error:
        write(args.output/'study_result.json',dict(status='operator_or_budget_stop',reason=str(error)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except BaseException:
        if getattr(args,'output_created',False):write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),no_automatic_restart=True))
        raise
