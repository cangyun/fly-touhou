"""Publish qualified movement guidance for future manual B/M job starts only.

No stop, restart, training launch, GUI launch or original-game operation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest,prepare_execution
from fly_touhou.thrl_movement_learning import template as guided_template,contract,REVISION
from fly_touhou.th14_completion_protocol import learning_overrides
from fly_touhou.resource_teacher_v7 import REVISION as TEACHER_REVISION
from deploy_rescue_chain_isolation import freeze_sources,GROUPS

GUIDED=('B0','B1','M0','M1')


def versioned_template(old,engine,key):
    result=guided_template(old) if key in GUIDED else deepcopy(old)
    if key in ('B1','M1'):result['teacher_revision']=TEACHER_REVISION
    result.update(source_runtime=engine['runtime'],source_manifest_sha256=engine['manifest_sha256'],
        profile=str(Path(engine['runtime'])/'profiles/th14_1.00b_original.json'),
        profile_sha256=digest(Path(engine['runtime'])/'profiles/th14_1.00b_original.json'))
    result['ppo_learning_overrides']=learning_overrides(result)
    contract(result)
    permitted={'source_runtime','source_manifest_sha256','profile','profile_sha256','ppo_learning_overrides'}
    if key in GUIDED:permitted|={'movement_guidance_revision','movement_guidance','authorized_movement_upgrade'}
    if key in ('B1','M1'):permitted.add('teacher_revision')
    changed={k for k in set(old)|set(result) if old.get(k)!=result.get(k)}
    if changed-permitted:raise ValueError('Unexpected template changes: '+str(changed-permitted))
    return result


def qualification(tests,cuda,shadow):
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    n=sum(int(s.get('tests',0)) for s in suites)
    if n<100 or any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped')):
        raise ValueError('Current movement/core/control regression evidence required')
    gpu=read(cuda);native=read(shadow/'report.json');cleanup=read(shadow/'cleanup.json')
    if (gpu.get('status')!='passed' or {c['core'] for c in gpu['cases']}!={'mlp','gru'}
            or any(not c.get('all_gradients_finite') for c in gpu['cases'])):
        raise ValueError('Both full-size CUDA cores must pass movement-gradient checks')
    calibration=native.get('calibration') or {}
    if (native.get('status')!='completed' or native.get('episodes')!=20
            or not calibration.get('adequate_smoke_coverage') or calibration.get('false_safe_rate',1)>.01
            or native.get('teacher_actions_executed')!=0 or native.get('optimizer_updates')!=0
            or cleanup!={'closed':True,'close_errors':[]}):
        raise ValueError('Bounded native shadow qualification or cleanup failed')
    if gpu['native_state_bundle_sha256']!=digest(shadow/'states.npz'):
        raise ValueError('CUDA checks used another native state bundle')
    return dict(CPU={'path':str(tests),'sha256':digest(tests),'tests':n},
        offline_CUDA={'path':str(cuda),'sha256':digest(cuda)},
        native_shadow={'path':str(shadow/'report.json'),'sha256':digest(shadow/'report.json'),
            'scope':'20 first-collision probes; observed actions only; no learning-effect claim'})


def deploy(*,center,runtime,output,tests,cuda,shadow,root=ROOT):
    root,center,runtime,output,tests,cuda,shadow=map(lambda p:Path(p).resolve(),
        (root,center,runtime,output,tests,cuda,shadow))
    if runtime.exists() or output.exists():raise FileExistsError('Use fresh release paths')
    evidence=qualification(tests,cuda,shadow)
    before=read(center/'configuration.json');ctl=ControlCenter(center)
    registry=ctl.registry();old_runtime=Path(before['engine']['runtime'])
    if digest(old_runtime/'source_manifest.json')!=before['engine']['manifest_sha256']:
        raise ValueError('Prior frozen engine identity changed')
    old_manifest=read(old_runtime/'source_manifest.json')
    native=old_manifest['qualification']['inherited-native-focus-proof']['verified_unchanged_sources']
    for relative,sha in native.items():
        if digest(root/relative)!=sha:raise ValueError('Qualified no-focus backend changed: '+relative)
    evidence['inherited-native-focus-proof']=dict(
        verified_unchanged_sources=native,source_runtime=str(old_runtime),
        scope='Unchanged private desktop/controller/hooks/software COM only; learner newly qualified separately')
    preserved={p.name:p.read_bytes() for p in center.iterdir() if p.is_file()
        and (p.name in ('settings.json','registry.json') or p.name.startswith('comparison') and p.suffix=='.json')}
    output.mkdir(parents=True);(output/'before').mkdir()
    (output/'before'/'configuration.json').write_bytes((center/'configuration.json').read_bytes())
    for name,data in preserved.items():(output/'before'/name).write_bytes(data)
    engine=freeze_sources(root,runtime,evidence,native)
    manifest=read(runtime/'source_manifest.json')
    manifest.update(learning_revision=before.get('learning_revision'),training_mode='full-story-v1',
        movement_guidance_revision=REVISION,
        engineering_change='Feasible movement-set teacher, measured native-collision auxiliary, V7 BC teacher, episode evidence. Future manual starts only.')
    write(runtime/'source_manifest.json',manifest);engine['manifest_sha256']=digest(runtime/'source_manifest.json')
    after=deepcopy(before);after.update(engine=engine,updated_utc=utc_now())
    changes={}
    for key in GROUPS:
        item=before['template_configs'][key]
        if digest(item['path'])!=item['sha256']:raise ValueError('Prior template identity changed')
        old=read(item['path']);new=versioned_template(old,engine,key)
        if digest(new['warm_start_checkpoint'])!=new['warm_start_sha256']:
            raise ValueError('Common untrained initializer changed')
        path=output/f'branch_{key}_config.json';write(path,new)
        changes[key]=sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k))
        label=item['label'].replace('无教师 ·','无教师预训练 ·').replace('有教师 ·','有教师预训练 ·')
        suffix=(' · 移动指导'+('＋新教师V7' if key in ('B1','M1') else '') if key in GUIDED else '')
        if suffix and suffix not in label:label+=suffix
        after['template_configs'][key]={**item,'path':str(path),'sha256':digest(path),'label':label}
    after['movement_guidance_qualification']=dict(report=str(output/'deployment_report.json'),
        guided_groups=list(GUIDED),BC_teacher_V7_groups=['B1','M1'],automatic_launch=False)
    # Exercise the real preparation path with fresh isolated folders only.
    # prepare_execution writes plans/configs; it does not dispatch processes.
    preparation=output/'offline-control-preparation';preparation.mkdir()
    request=dict(engine=engine,control=str(preparation),output=str(preparation/'output'),
        job_id='offline-movement-preparation',keys=list(GROUPS),mode='new',previous={},
        settings=read(center/'settings.json'),template_configs=after['template_configs'],
        python=before['python'])
    prepared=prepare_execution(request)
    for key,row in prepared['branches'].items():
        config=read(row['config'])
        if key in GUIDED and config['ppo_learning_overrides'].get('movement_teacher_coefficient')!=.02:
            raise ValueError('Real control preparation lost movement guidance')
    for item in manifest['files']:
        if digest(runtime/item['path'])!=item['sha256']:raise ValueError('Frozen runtime changed')
    with controller_lock(center):
        if read(center/'configuration.json')!=before:raise ValueError('Operator changed configuration during release')
        if any((center/k).read_bytes()!=v for k,v in preserved.items()):
            raise ValueError('Operator changed job registry/settings during release')
        write(center/'configuration.json',after)
    result=dict(status='passed',utc=utc_now(),engine=engine,qualification=evidence,changes=changes,
        future_manual_starts_only=True,training_jobs_started=0,original_games_started=0,
        active_jobs_modified=False,settings_registry_comparison_preserved=True,
        offline_prepared_groups=list(prepared['branches']),
        active_groups_at_preparation={k:ctl.branch_activity(k,registry)['running'] for k in GROUPS},
        templates=after['template_configs'],configuration_sha256=digest(center/'configuration.json'))
    write(output/'deployment_report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for field in ('runtime','output','tests','cuda','shadow'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    args=p.parse_args();deploy(**vars(args));print(str(args.output/'deployment_report.json'))
