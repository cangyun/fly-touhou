"""Publish the verified object-allocation input for future manual job launches.

Existing jobs retain their immutable requests and frozen runtimes. No stop,
launch, desktop, native game or GPU operation is performed by this script.
Old input checkpoints cannot silently resume under the new schema.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
from fly_touhou.thrl_inputs import (SEPARATED_TIMERS_INPUT,SEPARATED_OBJECT_PRIORITY_INPUT,
                                   ORIGINAL_TIMERS_INPUT)
from deploy_rescue_chain_isolation import freeze_sources,GROUPS

REVISION='object-priority-v3'


def versioned_template(old,engine,initial=None):
    config=deepcopy(old)
    schema=old['input_schema']
    if schema==SEPARATED_TIMERS_INPUT:
        if not initial:raise ValueError('Separated input needs a verified zero-step initialization')
        config.update(input_schema=SEPARATED_OBJECT_PRIORITY_INPUT,
            model_overrides={**config['model_overrides'],'input_schema':SEPARATED_OBJECT_PRIORITY_INPUT},
            warm_start_checkpoint=initial['path'],warm_start_sha256=initial['sha256'],
            object_selection_revision='nonitem-first-fill-v2')
    elif schema!=ORIGINAL_TIMERS_INPUT:
        raise ValueError('Unexpected source input schema: '+str(schema))
    runtime=Path(engine['runtime'])
    config.update(source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
        profile=str(runtime/'profiles/th14_1.00b_original.json'),
        profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'))
    allowed={'source_runtime','source_manifest_sha256','profile','profile_sha256'}
    if schema==SEPARATED_TIMERS_INPUT:
        allowed|={'input_schema','model_overrides','warm_start_checkpoint','warm_start_sha256',
                  'object_selection_revision'}
    changed={key for key in set(old)|set(config) if old.get(key)!=config.get(key)}
    if changed-allowed:raise ValueError('Unexpected template changes: '+str(changed-allowed))
    return config


def clone_zero_step_initial(old,output):
    """Copy the same untrained tensors/RNG/empty Adam; version metadata only."""
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_branching import warm_start
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    source=Path(old['warm_start_checkpoint'])
    if digest(source)!=old['warm_start_sha256']:raise ValueError('Initialization identity changed')
    payload=torch.load(source,map_location='cpu',weights_only=False)
    counters=('decisions','rollout_updates','adam_minibatch_steps','branch_decisions',
        'branch_rollout_updates','branch_adam_minibatch_steps','lifetime_decisions',
        'lifetime_rollout_updates','lifetime_adam_minibatch_steps','candidate_decisions',
        'candidate_optimized_decisions','candidate_paid_decisions_lower_bound')
    if (payload.get('schema')!='th14-thrl-training-checkpoint-v1'
            or any(type(payload.get(k)) is not int or payload[k]!=0 for k in counters)
            or payload['optimizer']['state']
            or payload.get('model_config',{}).get('input_schema')!=SEPARATED_TIMERS_INPUT
            or payload.get('optimizer_transaction_complete') is not True):
        raise ValueError('Only a complete zero-step initialization may be re-versioned')
    rewritten={**payload,'model_config':{**payload['model_config'],
        'input_schema':SEPARATED_OBJECT_PRIORITY_INPUT},
        'observation_upgrade':dict(revision=REVISION,source=str(source),source_sha256=digest(source),
            zero_training_steps=True,weights_changed=False,optimizer_changed=False,
            age_normalization_changed=False,not_a_trained_checkpoint_migration=True)}
    if output.exists():raise FileExistsError(output)
    torch.save(rewritten,output)
    target=MOActorCritic(**rewritten['model_config'])
    audit=warm_start(torch,target,make_optimizer(target,PPOConfig()),output,digest(output),exact_optimizer=True)
    if not all(torch.equal(value,target.state_dict()[name]) for name,value in payload['model'].items()):
        raise ValueError('Untrained initialization tensors changed')
    return {'path':str(output.resolve()),'sha256':digest(output),'source':str(source),
        'source_sha256':old['warm_start_sha256'],'model_tensors_unchanged':True,
        'zero_training_steps':True,'strict_loader':audit}


def verify_saved_snapshots(paths,output):
    import json
    import numpy as np
    from fly_touhou.state import Snapshot
    from fly_touhou.thrl_timer_observation import THRLTimerObservationEncoder
    from fly_touhou.thrl_object_selection_candidate import OfflineObjectSelectionCandidateEncoder
    from fly_touhou.thrl_observation_audit import audit_snapshot,aggregate_audits
    old=THRLTimerObservationEncoder(SEPARATED_TIMERS_INPUT,require_events=False)
    new=THRLTimerObservationEncoder(SEPARATED_OBJECT_PRIORITY_INPUT,require_events=False)
    candidate=OfflineObjectSelectionCandidateEncoder(require_events=False)
    before=[];after=[];examples=[];changed=0
    for path in paths:
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                if index>=1200:raise ValueError('Saved snapshot budget exceeded')
                state=Snapshot.from_dict(json.loads(line)['snapshot'])
                original,actual,expected=old.encode(state),new.encode(state),candidate.encode(state)
                for field in ('features','maps'):
                    if not np.array_equal(original[field],actual[field]):
                        raise ValueError('Allocation repair changed '+field)
                if not all(np.array_equal(actual[k],expected[k]) for k in actual):
                    raise ValueError('Registered encoder differs from verified offline candidate')
                changed+=int(not np.array_equal(actual['objects'],original['objects']))
                a=audit_snapshot(state,input_schema=SEPARATED_TIMERS_INPUT)
                b=audit_snapshot(state,input_schema=SEPARATED_OBJECT_PRIORITY_INPUT)
                before.append(a);after.append(b)
                if state.stage==1 and state.stage_frame==1602:
                    examples.append({'stage':state.stage,'stage_frame':state.stage_frame,
                        'before':a['selection']['groups'],'after':b['selection']['groups']})
    if not before:raise ValueError('Real saved snapshot qualification required')
    baseline=aggregate_audits(before);fixed=aggregate_audits(after)
    if (fixed['selection_groups']['bullet']['selected_danger_tokens']
            <baseline['selection_groups']['bullet']['selected_danger_tokens']):
        raise ValueError('Saved-sample dangerous bullet coverage regressed')
    result=dict(status='passed',snapshots=len(before),selection_changed_snapshots=changed,
        schema=SEPARATED_OBJECT_PRIORITY_INPUT,production_matches_offline_candidate=True,
        globals_maps_normalization_unchanged=True,before=baseline,after=fixed,examples=examples,
        sources=[{'path':str(p),'sha256':digest(p)} for p in paths],
        scope='saved biased snapshots; geometry screening is not learning or collision performance',
        missing_events_are_not_measured_zero=True)
    write(output,result)
    return result


def publish_future_configuration(center,before,after,preserved):
    """Switch only future launch templates; never mutate an active job."""
    center=Path(center)
    with controller_lock(center):
        if read(center/'configuration.json')!=before:raise ValueError('Control configuration changed')
        if any((center/name).read_bytes()!=data for name,data in preserved.items()):
            raise ValueError('Operator settings or job registry changed; retry publication')
        write(center/'configuration.json',after)


def deploy(*,center,runtime,output,tests,captures,root=ROOT):
    root,center,runtime,output,tests=map(lambda p:Path(p).resolve(),(root,center,runtime,output,tests))
    captures=[Path(p).resolve() for p in captures]
    if runtime.exists() or output.exists():raise FileExistsError('Use fresh release paths')
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    count=sum(int(row.get('tests',0)) for row in suites)
    if count<60 or any(int(row.get(k,0)) for row in suites for k in ('failures','errors','skipped')):
        raise ValueError('Current input/action/control qualification must pass')
    ctl=ControlCenter(center);before=deepcopy(ctl.config);registry=ctl.registry()
    if set(before['template_configs'])!=set(GROUPS):raise ValueError('Six template groups required')
    preserved={name:(center/name).read_bytes() for name in ('registry.json','settings.json')
        if (center/name).is_file()}
    old_runtime=Path(before['engine']['runtime'])
    if digest(old_runtime/'source_manifest.json')!=before['engine']['manifest_sha256']:
        raise ValueError('Prior runtime identity changed')
    qualification=before['engine']['qualification']
    native=deepcopy(qualification['inherited-native-focus-proof']['verified_unchanged_sources'])
    # Execution, PPO and reward implementations retain their existing evidence.
    for name in ('thrl_rewards.py','thrl_ppo.py','thrl_native_env.py'):
        rel='src/fly_touhou/'+name
        if digest(root/rel)!=digest(old_runtime/rel):raise ValueError('Unexpected training/runtime change: '+rel)
    for rel,sha in native.items():
        if digest(root/rel)!=sha:raise ValueError('Native execution path changed: '+rel)
    output.mkdir(parents=True);(output/'before').mkdir()
    write(output/'before/configuration.json',before)
    for name,data in preserved.items():(output/'before'/name).write_bytes(data)
    snapshot_proof=verify_saved_snapshots(captures,output/'production-snapshot-qualification.json')
    old={key:read(item['path']) for key,item in before['template_configs'].items()}
    for key,item in before['template_configs'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Prior template changed: '+key)
    initials={}
    for key in ('B0','M0'):initials[key]=clone_zero_step_initial(old[key],output/f'initial_{key}.pt')
    release_qualification={
        'CPU-input-actions-control':dict(path=str(tests),sha256=digest(tests),tests=count),
        'production-snapshot-input':dict(path=str(output/'production-snapshot-qualification.json'),
            sha256=digest(output/'production-snapshot-qualification.json'),snapshots=snapshot_proof['snapshots']),
        'inherited-native-focus-proof':deepcopy(qualification['inherited-native-focus-proof']),
        'inherited-CUDA-training-core':dict(source_runtime=str(old_runtime),
            scope='PPO/reward/native path unchanged; current release uses offline CPU input/model/backward checks',
            proof=deepcopy(qualification.get('offline-CUDA')))}
    engine=freeze_sources(root,runtime,release_qualification,native)
    manifest=read(runtime/'source_manifest.json')
    manifest.update(learning_revision=before.get('learning_revision'),training_mode='full-story-v1',
        input_release=REVISION,
        engineering_change='Versioned non-item-first object allocation; manual future launches only; old input resume blocked; active jobs preserved')
    write(runtime/'source_manifest.json',manifest);engine['manifest_sha256']=digest(runtime/'source_manifest.json')
    after=deepcopy(before);after.update(engine=engine,updated_utc=utc_now(),input_release=REVISION)
    for key in GROUPS:
        initial=initials.get(key[0]+'0')
        config=versioned_template(old[key],engine,initial)
        path=output/f'branch_{key}_config.json';write(path,config)
        item=before['template_configs'][key]
        label=item['label']
        if config['input_schema']==SEPARATED_OBJECT_PRIORITY_INPUT:label+=' · 弹幕名额修正v3'
        after['template_configs'][key]={**item,'path':str(path),'sha256':digest(path),
            'input_schema':config['input_schema'],'label':label}
    after['object_selection_qualification']=dict(report=str(output/'deployment_report.json'),
        input_schema=SEPARATED_OBJECT_PRIORITY_INPUT,revision='nonitem-first-fill-v2',
        runtime_for_future_manual_starts=True,active_jobs_modified=False,automatic_launch=False,
        old_input_resume='blocked; select new training; historical checkpoints preserved')
    for row in manifest['files']:
        if digest(runtime/row['path'])!=row['sha256']:raise ValueError('Frozen release changed')
    publish_future_configuration(center,before,after,preserved)
    report=dict(status='passed',utc=utc_now(),engine=engine,initials=initials,
        templates=after['template_configs'],input_schema=SEPARATED_OBJECT_PRIORITY_INPUT,
        qualification=release_qualification,training_jobs_started=0,original_game_processes_started=0,
        registry_settings_unchanged=all((center/name).read_bytes()==data for name,data in preserved.items()),
        old_runtime_preserved=True,old_logs_checkpoints_preserved=True,
        active_groups_at_preparation={key:ctl.branch_activity(key,registry)['running'] for key in GROUPS},
        configuration_sha256=digest(center/'configuration.json'))
    write(output/'deployment_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    for field in ('runtime','output','tests'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    parser.add_argument('--capture',dest='captures',type=Path,action='append',required=True)
    args=parser.parse_args();result=deploy(**vars(args))
    print(str(args.output/'deployment_report.json'))


if __name__=='__main__':main()
