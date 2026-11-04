"""Publish reviewed completion-v3 for a later manual start; never launch games.

Requires final CPU XML and the sealed offline CUDA qualification. Every old
template/runtime/checkpoint, operator setting and registry entry is preserved.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest,alive,read_optional
from fly_touhou.th14_completion_protocol import (baseline_parameters,exploration_parameters,
    learning_overrides,search_space)
from fly_touhou.th14_completion_v3 import RECIPE,REWARD_SCHEMA
from fly_touhou.thrl_inputs import ORIGINAL_TIMERS_INPUT,SEPARATED_TIMERS_INPUT,observation_shapes
from fly_touhou.thrl_teacher_training import TeacherPretrainingConfig
from fly_touhou.thrl_branching import validate_parallel_config
from prepare_thrl_control_center import freeze

GROUPS=('A0','A1','B0','B1','M0','M1')
BASE=ROOT/'artifacts/engineering/learning-revision-20261023'
DEFAULT_CUDA=Path('./outputs/engineering/learning-revision-20261023/offline-CUDA-v1/report.json')
DEFAULT_RUNTIME=Path('./outputs/source_snapshots/20261023-completion-v3')
DEFAULT_EVIDENCE=Path('./outputs/engineering/headless-20261022')


def require_cpu_xml(path):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    count=sum(int(row.get('tests',0)) for row in suites)
    if count<340 or any(int(row.get('failures',0))+int(row.get('errors',0)) for row in suites):
        raise ValueError('Final full CPU qualification must pass before deployment')
    return dict(path=str(path.resolve()),sha256=digest(path),tests=count)


def qualified_initials(path):
    data=read(path)
    if (data.get('status')!='passed' or data.get('training_recipe')!=RECIPE
            or data.get('synthetic_only') is not True
            or data.get('training_jobs_started')!=0 or data.get('original_game_processes_started')!=0
            or set(data.get('groups',{}))!={'A0','B0','M0'}):
        raise ValueError('Three fresh offline-CUDA model qualifications are required')
    for key,row in data['groups'].items():
        expected=ORIGINAL_TIMERS_INPUT if key=='A0' else SEPARATED_TIMERS_INPUT
        model=row['model_overrides'];init=row['initial_loader']
        if (model.get('input_schema')!=expected
                or model.get('feature_dim')!=observation_shapes(expected)['features'][0]
                or model.get('temporal_core','gru')!=('mlp' if key=='M0' else 'gru')
                or row.get('initialization_unchanged_by_test') is not True
                or init.get('common_weights_exact') is not True
                or any(init.get(field)!=0 for field in ('inherited_decisions','inherited_rollout_updates',
                                                        'inherited_adam_minibatch_steps','common_Adam_states'))
                or digest(row['initial_checkpoint'])!=row['initial_sha256']
                or row.get('synthetic_CUDA',{}).get('optimizer_updates',0)<1):
            raise ValueError('Fresh model/schema/checkpoint qualification differs: '+key)
    return data


def stopped_evidence(center,registry):
    evidence={key:center.branch_activity(key,registry) for key in GROUPS}
    if any(row['running'] for row in evidence.values()):
        raise ValueError('All groups must be stopped before publishing completion-v3')
    # Check historical jobs too: changing the active pointer cannot hide a
    # still-live launcher. PID alone is never treated as process identity.
    identities=[]
    for job in registry.get('jobs',[]):
        identities.append(dict(job=job['id'],kind='dispatcher',identity=job.get('dispatch_identity')))
        status=read_optional(Path(job['output'])/'comparison_status.json')
        identities.extend(dict(job=job['id'],kind='branch:'+key,identity=row.get('launch_identity'))
                          for key,row in status.get('branches',{}).items())
    for item in identities:
        item['alive_by_pid_and_creation']=alive(item['identity'])
        if item['alive_by_pid_and_creation']:
            raise ValueError('A recorded job process is still live: '+item['job']+'/'+item['kind'])
    return dict(groups=evidence,recorded_processes=identities)


def make_template(old,key,initial,engine):
    cfg=copy.deepcopy(old)
    schema=ORIGINAL_TIMERS_INPUT if key.startswith('A') else SEPARATED_TIMERS_INPUT
    runtime=Path(engine['runtime'])
    # No accidental continuation or old teacher origin in a fresh experiment.
    obsolete=('baseline_resume','previous_study_output','teacher_pretraining_result',
        'teacher_pretraining_result_path','teacher_pretraining_result_sha256','teacher_pretraining_reused',
        'teacher_resume_audit','candidate_initial_checkpoint','candidate_initial_sha256',
        'candidate_resume_applied','candidate_decision_offset','candidate_paid_decision_offset',
        'candidate_trial_decisions','candidate_development_decisions','inherited_unoptimized_sampling_lower_bound',
        'unknown_inflight_decisions','curriculum_inherited_pools','curriculum_overlay_path',
        'operator_stop_path','control_request','parameters','model_config')
    for field in obsolete:cfg.pop(field,None)
    for field in tuple(cfg):
        if field.startswith('inherited_'):cfg.pop(field,None)
    cfg.update(branch=key,input_schema=schema,training_recipe=RECIPE,reward_schema=REWARD_SCHEMA,
        model_overrides=copy.deepcopy(initial['model_overrides']),parameter_counts=copy.deepcopy(initial['parameter_counts']),
        warm_start_checkpoint=initial['initial_checkpoint'],warm_start_sha256=initial['initial_sha256'],
        source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
        profile=str(runtime/'profiles/th14_1.00b_original.json'),
        profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'),
        entropy_decay_decisions=1_000_000,curriculum_eval_backend='ready-parallel-v1',
        curriculum_eval_workers=16,curriculum_eval_batch_wait_ms=2.0,curriculum_eval_min_batch_size=4,
        new_initialization=True,old_checkpoints_used=False,manual_start_only=True,
        trial_wall_cap_seconds=None,study_wall_cap_seconds=None,
        user_authorization='Preparation only; user will manually start the next training job',
        experiment_comparison='Completion-v3: each input/core pair differs only by fresh validated teacher BC; pre-BC initial checkpoint identical',
        comparison_stop_path='',study_name='TH14-completion-v3-'+key,api_port=0)
    cfg['ppo_baseline']=baseline_parameters(cfg)
    cfg['ppo_learning_overrides']=learning_overrides(cfg)
    cfg['ppo_search_space']=search_space(cfg)
    cfg['ppo_exploration_schedule']=exploration_parameters(cfg,cfg['ppo_baseline']['entropy_coeff_start'])
    teacher=TeacherPretrainingConfig(samples=8192,workers=max(n for n in (1,2,4) if n<=cfg['workers']),
        epochs=20,minimum_epochs=4,patience=4,validation_enabled=True,
        validation_fraction=.25,validation_seed=11,minimum_kl_improvement=.005,validation_max_kl=.5)
    cfg['teacher_pretraining']=asdict(teacher) if key.endswith('1') else None
    cfg['capacity']=validate_parallel_config(cfg)
    return json.loads(json.dumps(cfg,ensure_ascii=False,allow_nan=False))


def deploy(*,tests,center_path,runtime,output,qualification,evidence,active_path,freezer=None):
    """Transactional publication; parameters permit isolated offline fixtures."""
    paths=[tests,center_path,runtime,output,qualification,evidence,active_path]
    tests,center_path,runtime,output,qualification,evidence,active_path=[Path(p).resolve() for p in paths]
    if runtime.exists() or output.exists():raise FileExistsError('Keep all existing frozen runtimes and prepared templates')
    cpu=require_cpu_xml(tests);cuda=qualified_initials(qualification)
    c=ControlCenter(center_path);before=copy.deepcopy(c.config);registry=c.registry()
    if set(before['template_configs'])!=set(GROUPS):raise ValueError('All six group templates are required')
    originals={name:(center_path/name).read_bytes() for name in ('configuration.json','registry.json','settings.json')}
    # _publish writes this exact representation. Refuse noncanonical bytes
    # before touching the panel rather than silently reformat its registry.
    canonical=json.dumps(registry,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8')
    if originals['registry.json']!=canonical:
        raise ValueError('Registry bytes are not the control writer format; deployment preserves them unchanged')
    stopped=stopped_evidence(c,registry)
    old_templates={}
    for key,item in before['template_configs'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Old template identity changed: '+key)
        old_templates[key]=read(item['path'])
    output.mkdir(parents=True,exist_ok=False);audit=output/'deployment';audit.mkdir()
    for name,raw in originals.items():(audit/(name+'.before')).write_bytes(raw)
    if active_path.exists():(audit/'active_thrl_study.json.before').write_bytes(active_path.read_bytes())
    write(audit/'preflight.json',dict(cpu=cpu,offline_CUDA=dict(path=str(qualification),sha256=digest(qualification)),
        stopped=stopped,training_jobs_started=0,original_game_processes_started=0))
    engine=(freeze if freezer is None else freezer)(runtime,tests,evidence,
        extra_qualification={'completion-v3-offline-CUDA':qualification},scientific_training_core_changed=True,
        engineering_change='Versioned measured protection/state inputs; bounded phase HP reward and resource-aware Bomb/miss costs; full-rollout GAE, 256-observation PPO minibatches and whole-epoch KL budget; independent native-seed teacher holdout; bounded ready-worker evaluation microbatching. Prepared for manual start; no original-game or training launch.')
    after=copy.deepcopy(before);after.update(engine=engine,updated_utc=utc_now(),standby_groups=list(GROUPS))
    prepared={}
    for key,old in old_templates.items():
        initial=cuda['groups'][key[0]+'0']
        cfg=make_template(old,key,initial,engine)
        for field in ('curriculum_protocol','curriculum_options','curriculum_manifest','curriculum_manifest_sha256'):
            if cfg.get(field)!=old.get(field):raise ValueError('Deployment changed curriculum thresholds or sampling')
        path=output/f'branch_{key}_config.json';write(path,cfg)
        if read(path)!=cfg or digest(cfg['warm_start_checkpoint'])!=cfg['warm_start_sha256']:
            raise ValueError('Prepared template or initial checkpoint changed')
        prepared[key]=cfg
        after['template_configs'][key]=dict(before['template_configs'][key],path=str(path),sha256=digest(path),
            input_schema=cfg['input_schema'],label=cfg['display_name']+' · 保护计时与资源v3')
    for base in ('A','B','M'):
        left,right=prepared[base+'0'],prepared[base+'1']
        if (left['warm_start_sha256']!=right['warm_start_sha256']
                or left['model_overrides']!=right['model_overrides']):
            raise ValueError('Teacher pair initial conditions differ')
    manifest=read(runtime/'source_manifest.json')
    if digest(runtime/'source_manifest.json')!=engine['manifest_sha256']:
        raise ValueError('New runtime manifest identity changed')
    for row in manifest['files']:
        path=(runtime/row['path']).resolve()
        if not path.is_relative_to(runtime) or digest(path)!=row['sha256']:
            raise ValueError('New frozen runtime file differs: '+row['path'])
    with controller_lock(center_path):
        if any((center_path/name).read_bytes()!=raw for name,raw in originals.items()):
            raise ValueError('Operator state changed during preparation; publication cancelled')
        final_stopped=stopped_evidence(ControlCenter(center_path),registry)
        for key,item in before['template_configs'].items():
            if digest(item['path'])!=item['sha256']:raise ValueError('Old template changed before publication: '+key)
        write(center_path/'configuration.json',after)
        ControlCenter(center_path)._publish(registry)
        if (center_path/'registry.json').read_bytes()!=originals['registry.json'] or (center_path/'settings.json').read_bytes()!=originals['settings.json']:
            raise RuntimeError('Operator registry/settings byte preservation failed')
        write(active_path,dict(schema='thrl-active-study-v1',status='ready_manual',comparison_output=str(center_path),
            training_recipe=RECIPE,reward_schema=REWARD_SCHEMA,training_jobs_started=0,
            note='下一轮v3已准备，尚未开启。请手动点击新训练；旧v2不可跨协议继续。新增保护计时、资源奖励、完整GAE与KL预算、教师独立种子留出验证；复核最多16实例、2ms小批合并。旧模型和日志保留，原设置及注册表未改。'))
    verified=read(center_path/'configuration.json')
    if verified!=after:raise ValueError('Published configuration readback differs')
    for key,item in verified['template_configs'].items():
        if digest(item['path'])!=item['sha256'] or read(item['path'])!=prepared[key]:
            raise ValueError('Published template readback differs: '+key)
    result=dict(status='passed',created_utc=utc_now(),engine=engine,cpu=cpu,
        offline_CUDA=dict(path=str(qualification),sha256=digest(qualification)),
        reward_schema=REWARD_SCHEMA,training_recipe=RECIPE,
        prepared_templates=verified['template_configs'],teacher_pairs_identical_pre_BC=True,
        curriculum_thresholds_and_fractions_unchanged=True,old_templates_preserved=True,
        old_runtime_and_checkpoints_preserved=True,settings_preserved_byte_for_byte=True,
        registry_preserved_byte_for_byte=True,manual_start_only=True,training_jobs_started=0,
        original_game_processes_started=0,published_configuration_sha256=digest(center_path/'configuration.json'),
        stopped_process_evidence=final_stopped)
    write(audit/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--tests',type=Path,required=True,help='Final full-suite CPU JUnit XML')
    parser.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    parser.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--output',type=Path,default=BASE/'prepared-v1')
    parser.add_argument('--offline-cuda',type=Path,default=DEFAULT_CUDA)
    parser.add_argument('--evidence',type=Path,default=DEFAULT_EVIDENCE)
    args=parser.parse_args()
    deploy(tests=args.tests,center_path=args.center,runtime=args.runtime,output=args.output,
        qualification=args.offline_cuda,evidence=args.evidence,
        active_path=ROOT/'artifacts/engineering/active_thrl_study.json')
    print(str(args.output/'deployment/report.json'))


if __name__=='__main__':main()
