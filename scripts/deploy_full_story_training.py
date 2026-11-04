"""Publish single-policy full-story templates; never start a job or native EXE."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
from fly_touhou.thrl_learning_revision import full_story_template,upgrade_record,FULL_STORY_REVISION
from fly_touhou.thrl_training_mode import validate_training_mode,episode_protocol
from fly_touhou.th14_completion_protocol import learning_overrides
from deploy_rescue_chain_isolation import freeze_sources,FOCUS_CRITICAL,GROUPS,stopped


def deploy(*,tests,focus_proof,offline_cuda,center,runtime,output,root=ROOT):
    root,tests,focus_proof,offline_cuda,center,runtime,output=map(lambda x:Path(x).resolve(),
        (root,tests,focus_proof,offline_cuda,center,runtime,output))
    if runtime.exists() or output.exists():raise FileExistsError('Use fresh publication paths')
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    count=sum(int(row.get('tests',0)) for row in suites)
    if count<663 or any(int(row.get(k,0)) for row in suites for k in ('failures','errors','skipped')):
        raise ValueError('Required full regression checks must pass without skips')
    proof,cuda=read(focus_proof),read(offline_cuda)
    if proof.get('status')!='passed' or proof.get('render_backend')!='fake-com-v1':
        raise ValueError('Qualified isolated desktop backend evidence required')
    if cuda.get('status')!='passed':raise ValueError('Current offline CUDA proof required')
    # This reuses only the unchanged execution path's native proof. It does not
    # relabel changed learner sources as having been tested in an old game run.
    original_hashes=proof['source_sha256']
    native_paths=set(FOCUS_CRITICAL)|{p for p in original_hashes if p.startswith('native/')}
    native_paths|={p for p in original_hashes if p.startswith('src/fly_touhou/hooks/')}
    native_paths|={'src/fly_touhou/native.py','src/fly_touhou/frida_controller.py',
                   'src/fly_touhou/thrl_native_env.py','src/fly_touhou/responsive_campaign.py'}
    hashes={p:original_hashes[p] for p in sorted(native_paths)}
    for rel,sha in hashes.items():
        if digest(root/rel)!=sha:raise ValueError('Qualified native path changed: '+rel)
    qualification={'CPU':{'path':str(tests),'sha256':digest(tests),'tests':count},
        'offline-CUDA':{'path':str(offline_cuda),'sha256':digest(offline_cuda)},
        'inherited-native-focus-proof':{'path':str(focus_proof),'sha256':digest(focus_proof),
            'scope':'unchanged private desktop, controller, hooks and software COM; no new game run',
            'verified_unchanged_sources':hashes}}
    stopped(center)
    before=read(center/'configuration.json')
    if set(before['template_configs'])!=set(GROUPS):raise ValueError('Six template groups required')
    originals={p.name:p.read_bytes() for p in center.iterdir()
        if p.is_file() and (p.name in ('configuration.json','registry.json','settings.json')
                          or p.name.startswith('comparison') and p.suffix=='.json')}
    output.mkdir(parents=True);(output/'before').mkdir()
    for name,data in originals.items():(output/'before'/name).write_bytes(data)
    engine=freeze_sources(root,runtime,qualification,hashes)
    manifest=read(runtime/'source_manifest.json')
    manifest.update(learning_revision=FULL_STORY_REVISION,training_mode='full-story-v1',
        engineering_change='Single shared policy, fresh complete Story episodes; no course sampling or gates; recurrent continuity and explicit score migration. No automatic launch.')
    write(runtime/'source_manifest.json',manifest);engine['manifest_sha256']=digest(runtime/'source_manifest.json')
    after=deepcopy(before);after.update(engine=engine,training_mode='full-story-v1',
        learning_revision=FULL_STORY_REVISION,updated_utc=utc_now())
    migrations={}
    for key in GROUPS:
        item=before['template_configs'][key]
        if digest(item['path'])!=item['sha256']:raise ValueError('Old template identity changed: '+key)
        old=read(item['path']);cfg=full_story_template(old)
        migrations[key]=upgrade_record(old,cfg)
        cfg.update(source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
            profile=str(runtime/'profiles/th14_1.00b_original.json'),
            profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'),
            manual_start_only=True,user_authorization='Correct and publish only; operator starts using the control panel',
            experiment_comparison='One policy per input/core/teacher experiment; all native episodes start from stage1 and end only at native Game Over or clear')
        cfg['ppo_learning_overrides']=learning_overrides(cfg)
        cfg['episode_protocol']=episode_protocol(cfg)
        validate_training_mode(cfg)
        if digest(cfg['warm_start_checkpoint'])!=cfg['warm_start_sha256']:
            raise ValueError('Initialization changed: '+key)
        path=output/f'branch_{key}_config.json';write(path,cfg)
        after['template_configs'][key]={**item,'path':str(path),'sha256':digest(path),
            'label':cfg['display_name']+' · 单策略全程连打'}
    prior_course=after.pop('curriculum_qualification',None)
    if prior_course:after.setdefault('historical_qualifications',{})['retired_curriculum']=prior_course
    after['full_story_qualification']={'report':str(output/'deployment_report.json'),
        'training_mode':'full-story-v1','native_start':'fresh stage1',
        'end':'native Game Over or six-stage clear','reverse_curriculum':False,
        'expert_routing':False,'automatic_launch':False}
    after.pop('maintenance',None)
    with controller_lock(center):
        activity=stopped(center)
        if any((center/name).read_bytes()!=data for name,data in originals.items()):
            raise ValueError('Operator state changed during publication')
        for row in manifest['files']:
            if digest(runtime/row['path'])!=row['sha256']:raise ValueError('Frozen runtime changed')
        for item in after['template_configs'].values():
            if digest(item['path'])!=item['sha256']:raise ValueError('Prepared template changed')
        write(center/'configuration.json',after)
    report=dict(status='passed',utc=utc_now(),engine=engine,qualification=qualification,
        prepared_templates=after['template_configs'],migration_contracts=migrations,
        original_game_processes_started=0,training_jobs_started=0,stopped_groups=activity,
        old_logs_checkpoints_and_history_preserved=True,
        operator_files_preserved={n:digest(center/n) for n in originals if n!='configuration.json'},
        configuration_sha256=digest(center/'configuration.json'))
    write(output/'deployment_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('tests','focus-proof','offline-cuda','runtime','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    a=p.parse_args();deploy(**vars(a));print(str(a.output/'deployment_report.json'))


if __name__=='__main__':main()
