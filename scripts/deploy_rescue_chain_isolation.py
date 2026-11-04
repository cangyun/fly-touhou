"""Publish qualified isolated native execution and explicit rescue/full-chain v1.

Never launches training/games, resumes a checkpoint, or refreshes the GUI.
Old reports, templates, initial weights and operator state remain untouched.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
from fly_touhou.thrl_learning_revision import TARGET,AUTHORIZATION,upgrade_record
from fly_touhou.th14_completion_protocol import learning_overrides

GROUPS=('A0','A1','B0','B1','M0','M1')
FOCUS_CRITICAL=(
    'src/fly_touhou/isolated_desktop_process.py',
    'src/fly_touhou/responsive_controller.py',
    'src/fly_touhou/thrl_native_events.py',
    'src/fly_touhou/hooks/th14_responsive.js',
    'src/fly_touhou/hooks/th14_headless.js',
    'native/th14_fake_d3d9.dll',
    'native/th14_fake_d3d9.dll.sha256',
)
PATH_FIELDS={'source_runtime','source_manifest_sha256','profile','profile_sha256'}
ALLOWED_TEMPLATE_CHANGES=set(TARGET)|{'authorized_resume_upgrade','ppo_learning_overrides'}|PATH_FIELDS


def checked_source(root,relative):
    path=(root/relative).resolve()
    if path==root or not path.is_relative_to(root):raise ValueError('Source path escapes root: '+str(relative))
    return path


def qualifications(root,tests,focus,cuda):
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    count=sum(int(row.get('tests',0)) for row in suites)
    if count<=0 or any(int(row.get(k,0)) for row in suites for k in ('failures','errors','skipped')):
        raise ValueError('Current CPU XML must have positive tests and no failure/error/skip')
    proof=read(focus);cuda_data=read(cuda)
    if proof.get('status')!='passed':raise ValueError('Current real native focus proof must pass')
    if proof.get('render_backend')!='fake-com-v1':
        raise ValueError('New software COM runtime requires its own qualification, not historical NULLREF proof')
    if cuda_data.get('status')!='passed':raise ValueError('Current offline CUDA proof must pass')
    hashes=proof.get('source_sha256') or {}
    if not set(FOCUS_CRITICAL).issubset(hashes):raise ValueError('Native proof lacks critical source hashes')
    for relative,expected in hashes.items():
        if digest(checked_source(root,relative))!=expected:
            raise ValueError('Native proof source changed: '+relative)
    return {'CPU':{'path':str(tests),'sha256':digest(tests),'tests':count},
        'native-private-desktop-focus':{'path':str(focus),'sha256':digest(focus)},
        'offline-CUDA-rescue-chain':{'path':str(cuda),'sha256':digest(cuda)}},hashes


def freeze_sources(root,runtime,qualification,proof_hashes):
    if runtime.exists():raise FileExistsError('Frozen runtime must be new')
    paths=[]
    for folder in ('src','scripts','profiles','native'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file()
                     and not {'__pycache__','.git','target'}.intersection(p.relative_to(root/folder).parts)
                     and p.suffix not in ('.pyc','.pyo'))
    paths.extend(root/name for name in ('COPYING','README.md','pyproject.toml','data/connectome/model.py'))
    runtime.mkdir(parents=True,exist_ok=False);records=[]
    for source in sorted(paths):
        relative=source.relative_to(root).as_posix();checked_source(root,relative)
        data=source.read_bytes()
        if source.suffix=='.py':compile(data,str(source),'exec')
        target=runtime/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(path=relative,bytes=len(data),sha256=digest(target)))
    for relative,expected in proof_hashes.items():
        if digest(checked_source(runtime,relative))!=expected:
            raise ValueError('Frozen source differs from qualified focus proof: '+relative)
    write(runtime/'source_manifest.json',dict(schema='thrl-control-engine-v1',origin=str(root),
        created_utc=utc_now(),qualification=qualification,files=records,
        scientific_training_core_changed=True,learning_revision=TARGET['learning_revision'],
        engineering_change='Private non-input desktop startup with frozen software D3D9 COM DLL and source; rare-state Bomb supervision; native-terminal training chains. No automatic job launch.',
        reward_sha256=digest(root/'src/fly_touhou/thrl_rewards.py')))
    return dict(runtime=str(runtime),manifest_sha256=digest(runtime/'source_manifest.json'),qualification=qualification)


def template(old,engine):
    new=deepcopy(old);new.update(TARGET,authorized_resume_upgrade=AUTHORIZATION)
    upgrade_record(old,new)  # Enforces unchanged model/reward/learning-rate contracts.
    runtime=Path(engine['runtime'])
    new.update(source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
        profile=str(runtime/'profiles/th14_1.00b_original.json'),
        profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'))
    new['ppo_learning_overrides']=learning_overrides(new)
    changes={key:{'before':old.get(key),'after':new.get(key)} for key in set(old)|set(new) if old.get(key)!=new.get(key)}
    if set(changes)-ALLOWED_TEMPLATE_CHANGES:raise ValueError('Unexpected template changes: '+str(set(changes)-ALLOWED_TEMPLATE_CHANGES))
    return new,changes


def stopped(center):
    c=ControlCenter(center);registry=c.registry()
    activity={key:c.branch_activity(key,registry) for key in GROUPS}
    if any(row['running'] for row in activity.values()):raise ValueError('Stop existing groups before publication')
    return activity


def deploy(*,tests,focus_proof,offline_cuda,center,runtime,output,root=ROOT):
    root,tests,focus_proof,offline_cuda,center,runtime,output=[Path(x).resolve() for x in
        (root,tests,focus_proof,offline_cuda,center,runtime,output)]
    if runtime.exists() or output.exists():raise FileExistsError('Preserve old runtime and publication evidence')
    qualification,hashes=qualifications(root,tests,focus_proof,offline_cuda)
    names={'configuration.json','settings.json','registry.json'}|{p.name for p in center.glob('comparison*.json')}
    originals={name:(center/name).read_bytes() for name in names}
    before=read(center/'configuration.json')
    if set(before['template_configs'])!=set(GROUPS):raise ValueError('Exactly six existing templates required')
    stopped(center)
    old={}
    for key,item in before['template_configs'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Old template hash changed: '+key)
        old[key]=read(item['path'])
        if old[key].get('warm_start_checkpoint') and digest(old[key]['warm_start_checkpoint'])!=old[key]['warm_start_sha256']:
            raise ValueError('Old initialization changed: '+key)
    output.mkdir(parents=True,exist_ok=False);backup=output/'before';backup.mkdir()
    for name,data in originals.items():(backup/name).write_bytes(data)
    engine=freeze_sources(root,runtime,qualification,hashes)
    after=deepcopy(before);after.update(engine=engine,updated_utc=utc_now())
    differences={};prepared={}
    for key in GROUPS:
        cfg,differences[key]=template(old[key],engine)
        path=output/f'branch_{key}_config.json';write(path,cfg);prepared[key]=cfg
        after['template_configs'][key]={**before['template_configs'][key],'path':str(path),'sha256':digest(path)}
    after.pop('maintenance',None)
    after['focus_fix_candidate']={'native_focus_acceptance':'passed','proof':str(focus_proof),
        'proof_sha256':digest(focus_proof),'automatic_restart':False,'published_runtime':str(runtime)}
    with controller_lock(center):
        if {p.name for p in center.glob('comparison*.json')}!={n for n in names if n.startswith('comparison')}:
            raise ValueError('Operator comparison files changed during preparation')
        if any((center/name).read_bytes()!=data for name,data in originals.items()):
            raise ValueError('Operator state changed during preparation')
        activity=stopped(center)
        qualifications(root,tests,focus_proof,offline_cuda)
        for label,row in qualification.items():
            if digest(row['path'])!=row['sha256']:raise ValueError('Qualification evidence changed: '+label)
        for key,item in before['template_configs'].items():
            if digest(item['path'])!=item['sha256']:raise ValueError('Old template changed before publication: '+key)
        for row in read(runtime/'source_manifest.json')['files']:
            if digest(checked_source(runtime,row['path']))!=row['sha256']:
                raise ValueError('Frozen runtime changed before publication: '+row['path'])
        if digest(runtime/'source_manifest.json')!=engine['manifest_sha256']:
            raise ValueError('Frozen manifest changed before publication')
        for key,item in after['template_configs'].items():
            if digest(item['path'])!=item['sha256'] or read(item['path'])!=prepared[key]:
                raise ValueError('New template changed before publication: '+key)
        write(center/'configuration.json',after)
        # Do not call _publish: registry/comparison are operator history bytes.
        if any((center/name).read_bytes()!=data for name,data in originals.items() if name!='configuration.json'):
            raise RuntimeError('Operator history/settings byte preservation failed')
    if read(center/'configuration.json')!=after:raise RuntimeError('Publication readback differs')
    report=dict(status='passed',created_utc=utc_now(),engine=engine,qualification=qualification,
        changed_scientific_protocol_fields=TARGET,template_differences=differences,
        prepared_templates=after['template_configs'],initial_checkpoints_preserved=True,
        operator_files_preserved={n:digest(center/n) for n in names if n!='configuration.json'},
        maintenance_released=True,old_templates_and_results_preserved=True,stopped_groups=activity,
        training_jobs_started=0,original_game_processes_started=0,GUI_refreshed=False,
        publication_configuration_sha256=digest(center/'configuration.json'))
    write(output/'deployment_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('tests','focus-proof','offline-cuda','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    a=p.parse_args();deploy(**vars(a));print(str(a.output/'deployment_report.json'))


if __name__=='__main__':main()
