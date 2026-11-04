"""Prepare a clean proposed finite experiment;freeze only an approved overlay list.

No training, CUDA allocation, original EXE, demo/control changes or restart.
"""
from pathlib import Path
import argparse
import copy
from dataclasses import asdict
import difflib
import json
import shutil
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_parameter_transfer import (
    EXPERIMENT_SCHEMA,INPUT_SCHEMA,file_sha,read_evaluation_source,source_historical_metadata,
    validate_finite_config,validate_target_parameters)
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from train_b1_parameter_transfer import STRICT_FP32_PRECISION

D95=Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')
D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
CALIBRATION_RESULT=D95.parent/'result.json'
CALIBRATION_RESULT_SHA='ed0b7a6025f825493f79f6de0fedb180723c2ff82e51804a25f413f4f7f098ef'
BASE=Path('./outputs/source_snapshots/20261025-B1-full-geometry-paired-development-v2')
BASE_SHA='7678b3db16d0f54d0a1d0a7f765d8a253a73e00f7e0b5a483e1f8fe086c30a37'
REQUIRED_OVERLAYS={
    'src/fly_touhou/thrl_policy_actions.py','src/fly_touhou/thrl_ppo.py',
    'src/fly_touhou/thrl_ready_collection.py','src/fly_touhou/thrl_deathbomb_learning.py',
    'src/fly_touhou/thrl_full_story_evaluation.py','scripts/train_thrl_native.py',
    'src/fly_touhou/thrl_native_env.py','src/fly_touhou/thrl_point_motion_audit.py',
    'src/fly_touhou/thrl_point_motion_history.py',
    'src/fly_touhou/thrl_branching.py','src/fly_touhou/thrl_parameter_transfer.py',
    'scripts/train_b1_parameter_transfer.py'}
FORBIDDEN_OVERLAYS={
    'src/fly_touhou/thrl_movement_teacher.py','src/fly_touhou/thrl_model.py',
    'src/fly_touhou/thrl_pair_policy.py','src/fly_touhou/thrl_pair_risk.py',
    'src/fly_touhou/thrl_laser_risk.py','src/fly_touhou/thrl_rewards.py',
    'src/fly_touhou/th14_completion_v3.py','src/fly_touhou/th14_completion_protocol.py'}


def calibration_cost_evidence(source):
    if file_sha(CALIBRATION_RESULT)!=CALIBRATION_RESULT_SHA:
        raise ValueError('Pinned source calibration cost evidence changed')
    result=read(CALIBRATION_RESULT)
    if (result['checkpoint_sha256']!=D95_SHA or result['paid_risk_fit_Adam_steps']!=8000
            or result['selected_risk_fit_Adam_steps']!=source['risk_fit_Adam_steps']
            or result['selected_risk_fit_Adam_steps']!=7800 or result['selected_epoch']!=39):
        raise ValueError('Source paid and selected calibration counters differ from their pinned evidence')
    return dict(result=str(CALIBRATION_RESULT),result_sha256=CALIBRATION_RESULT_SHA,
        paid_risk_fit_Adam_steps=8000,selected_risk_fit_Adam_steps=7800,selected_epoch=39,
        epochs_executed=result['epochs_executed'],source_checkpoint_sha256=D95_SHA,
        scope='risk-readout calibration only;not full upstream PPO or lifetime costs',
        inherited_as_new_run_progress=False)


def proposed_config(source,runtime,seed,budget):
    old=source['config']
    parameters=copy.deepcopy(old['parameters']);parameters['action_temperature']=.25
    validate_target_parameters(old['parameters'],parameters)
    config=dict(schema=EXPERIMENT_SCHEMA,branch='B1',display_name='D95参数迁移 · 新Adam · T0.25有限实验',
        input_schema=INPUT_SCHEMA,policy_model_config=copy.deepcopy(source['model_config']),
        parameter_transfer=dict(source_checkpoint=str(D95),source_sha256=D95_SHA),
        game_exe=old['game_exe'],game_exe_sha256=old['game_exe_sha256'],
        profile=old['profile'],profile_sha256=old['profile_sha256'],
        source_runtime=str(runtime),source_manifest_sha256=None,seed=seed,
        workers=30,maximum_concurrent_native_worlds=30,headless=True,collector_mode='ready',
        point_motion_telemetry=True,
        ready_shared_observations=False,policy_action_count=19,native_frame_horizon=None,
        finite_native_lives=True,autonomous_bomb=True,bombs_enabled=True,continues_allowed=False,
        training_mode='full-story-v1',first_hit_terminal=False,attention='both',
        training_recipe='th14-completion-v3',reward_schema='th14-completion-reward-v3',parameters=parameters,
        precision=copy.deepcopy(STRICT_FP32_PRECISION),
        learning_revision=old['learning_revision'],movement_guidance_revision=old['movement_guidance_revision'],
        teacher_revision=old['teacher_revision'],movement_guidance=copy.deepcopy(old['movement_guidance']),
        deathbomb_auxiliary_coefficient=old['deathbomb_auxiliary_coefficient'],
        deathbomb_auxiliary_min_remaining_frames=old['deathbomb_auxiliary_min_remaining_frames'],
        full_story_burn_in=128,full_story_history_storage='raw-v1',
        rollout_total_decisions=7680,trial_decisions=budget,development_decisions=[budget],
        development_episodes=20,validation_episodes_per_policy=0,checkpoint_every_updates=2,
        full_story_evaluation_backend='full-story-ready-parallel-v1',full_story_evaluation_workers=20,
        full_story_evaluation_action_selection='greedy',evaluation_after_training_shutdown=True,
        optimization_trials=1,first_trial_protected=True,torch_threads=2,api_port=0,
        trial_wall_cap_seconds=None,study_wall_cap_seconds=None,
        formal_acceptance=False,no_automatic_restart=True,
        native_item_read_qualification=old['native_item_read_qualification'],
        initialization_semantics='evaluation model parameters only;fresh empty Adam;fresh seed;candidate axis starts0',
        temperature_experiment=dict(source=1.,target=.25,optimality_claimed=False,old_rollout_or_logp_reused=False),
        source_historical_work_is_not_new_candidate_progress=True,
        source_historical_work=source_historical_metadata(source),
        source_calibration_cost_evidence=calibration_cost_evidence(source),
        teacher_uncertainty_source='S486 unchanged;current undeployed player-aware repair excluded',
        RAM_guard=dict(start_available_GiB=16,running_available_GiB=3,stop='existing StopRequested cleanup;no restart'),
        preparation_status='proposed_not_frozen_not_started')
    validate_finite_config(config)
    return config


def freeze_runtime(runtime,overlay_manifest,output):
    if file_sha(BASE/'source_manifest.json')!=BASE_SHA:
        raise ValueError('Pinned S486 baseline changed')
    specification=read(overlay_manifest)
    if specification.get('schema')!='B1-D95-next-training-approved-overlays-v1':
        raise ValueError('An explicit final approved overlay manifest is required')
    entries=specification['files'];names={item['path'] for item in entries}
    if len(names)!=len(entries) or not REQUIRED_OVERLAYS.issubset(names) or names&FORBIDDEN_OVERLAYS:
        raise ValueError('Missing final temperature/transfer dependencies or an undeclared teacher/reward/model change')
    for item in entries:
        path=(ROOT/item['path']).resolve()
        if (not path.is_relative_to(ROOT) or not item.get('semantic_role')
                or 'demo' in item['path'].lower() or file_sha(path)!=item['sha256']):
            raise ValueError('Final overlay identity/role differs;do not freeze an intermediate revision')
    runtime.mkdir(parents=True,exist_ok=False)
    baseline=read(BASE/'source_manifest.json')
    for item in baseline['files']:
        source=(BASE/item['path']).resolve()
        if not source.is_relative_to(BASE.resolve()) or file_sha(source)!=item['sha256']:
            raise ValueError('S486 baseline file identity differs')
        if item['path'] in names:continue
        target=runtime/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    differences=[]
    for item in entries:
        target=runtime/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/item['path'],target)
        if file_sha(target)!=item['sha256']:raise ValueError('Frozen overlay bytes changed')
        old=(BASE/item['path']).read_text(encoding='utf-8').splitlines(True) if (BASE/item['path']).is_file() else []
        new=target.read_text(encoding='utf-8').splitlines(True)
        differences.extend(difflib.unified_diff(old,new,fromfile='S486/'+item['path'],tofile='prepared/'+item['path']))
    files=[dict(path=path.relative_to(runtime).as_posix(),sha256=file_sha(path),bytes=path.stat().st_size)
           for path in sorted(runtime.rglob('*')) if path.is_file()]
    write(runtime/'source_manifest.json',dict(schema='B1-D95-fresh-Adam-temperature-runtime-v1',utc=utc_now(),
        baseline=str(BASE),baseline_manifest_sha256=BASE_SHA,approved_overlay_manifest=str(overlay_manifest),
        approved_overlay_manifest_sha256=file_sha(overlay_manifest),semantic_overlays=entries,files=files,
        precision=copy.deepcopy(STRICT_FP32_PRECISION),
        reward_gamma_and_loss_coefficients_unchanged=True,teacher_uncertainty_unchanged_from_S486=True,
        old_demo_runtimes_modified=False,native_worlds_started=0,optimizer_updates=0))
    (output/'runtime-vs-S486.patch').write_text(''.join(differences),encoding='utf-8')
    return file_sha(runtime/'source_manifest.json')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--runtime',type=Path,default=Path('./outputs/source_snapshots/20261026-B1-D95-T025-fresh-Adam-prepared-v3'))
    parser.add_argument('--seed',type=int,default=2026102601)
    parser.add_argument('--new-decisions',type=int,default=1228800)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--overlay-manifest',type=Path)
    args=parser.parse_args()
    if args.freeze and args.overlay_manifest is None:parser.error('--freeze requires the final --overlay-manifest')
    import psutil,torch
    torch.set_num_threads(2)
    source=read_evaluation_source(torch,D95,D95_SHA)
    config=proposed_config(source,args.runtime,args.seed,args.new_decisions)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.freeze:
        config['source_manifest_sha256']=freeze_runtime(args.runtime,args.overlay_manifest,args.output)
        config['preparation_status']='frozen_prepared_not_started'
    write(args.output/('config.prepared.json' if args.freeze else 'config.proposed.json'),config)
    maps=7680*16*92*96*4;history=30*128*16*92*96*4
    resource=dict(schema='B1-D95-next-training-resource-evidence-v1',utc=utc_now(),
        available_RAM_bytes=psutil.virtual_memory().available,physical_RAM_bytes=psutil.virtual_memory().total,
        J_free_disk_bytes=shutil.disk_usage('J:/').free,rollout_map_bytes=maps,raw128_history_map_bytes=history,
        numeric_maps_only_lower_bound_bytes=maps+history,includes_native30_processes_or_temporary_copies=False,
        rollout_object_bytes=7680*96*29*4,model_state_bytes=sum(v.numel()*v.element_size() for v in source['model'].values()),
        source_checkpoint_bytes=D95.stat().st_size,new_decisions=args.new_decisions,
        full_rollouts=args.new_decisions//7680,worlds=30,decisions_per_world_per_rollout=256,
        startup_available_RAM_min_GiB=16,running_available_RAM_min_GiB=3,
        concurrent_label_export_modified=False,new_native_worlds=0,new_optimizer_updates=0,CUDA_initialized=torch.cuda.is_initialized())
    write(args.output/'resource-evidence.json',resource)
    write(args.output/'preparation.json',dict(status=config['preparation_status'],utc=utc_now(),
        source_checkpoint=str(D95),source_sha256=D95_SHA,source_checkpoint_schema=source['schema'],
        no_source_optimizer=True,no_source_RNG=True,source_temperature=1.,target_temperature=.25,
        precision=config['precision'],
        source_historical_work=source_historical_metadata(source),
        source_calibration_cost_evidence=calibration_cost_evidence(source),
        new_candidate_axis_starts0=True,old_demo_or_current_control_modified=False,
        native_worlds_started=0,optimizer_updates=0,training_started=False))
    print(json.dumps(dict(status=config['preparation_status'],output=str(args.output),runtime_frozen=args.freeze)))


if __name__=='__main__':main()
