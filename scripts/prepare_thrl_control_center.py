"""Freeze the reviewed engine and connect the native panel to existing jobs."""
import argparse
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json,utc_now
from fly_touhou.thrl_control import ControlCenter,DEFAULT_SETTINGS,ALL_GROUPS,digest
from fly_touhou.thrl_training_mode import is_full_story,validate_training_mode


def template_descriptors(folder,groups=ALL_GROUPS):
    """Load six prepared groups; this never launches games or changes settings."""
    templates={}
    for key in groups:
        path=(folder/f'branch_{key}_config.json').resolve();source=read_shared_json(path)
        validate_training_mode(source)
        templates[key]={'path':str(path),'sha256':digest(path),'input_schema':source['input_schema'],
            'label':source['display_name']}
    return templates


def imported_registry(manifest,templates,study,control):
    """Missing historical M groups become standby instead of fabricated jobs."""
    keys=[key for key in templates if key in manifest['branches']]
    job_id='imported-'+study.name
    registry={'jobs':[{'id':job_id,'keys':keys,'output':str(study.resolve()),
        'control':str(control.resolve()),'mode':'imported','submitted_unix':0,
        'submitted_utc':manifest.get('created_utc')}],'active':{}}
    for key in keys:
        row=manifest['branches'][key]
        registry['active'][key]={'job_id':job_id,**{name:row[name] for name in
            ('output','label','attention','workers','teacher_pretraining','input_schema')}}
    return registry


def freeze(runtime,tests,evidence,*,extra_qualification=None,scientific_training_core_changed=False,engineering_change=None):
    if runtime.exists():raise FileExistsError('Frozen runtime must be a new directory')
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if sum(int(row.get('tests',0)) for row in suites)<340 or any(int(row.get('failures',0))+int(row.get('errors',0)) for row in suites):
        raise ValueError('Current required CPU checks have not passed')
    qualification={'CPU':{'path':str(tests.resolve()),'sha256':digest(tests)}}
    for name in ('no-flash-offscreen-v1','offscreen-trajectory-v1','teacher-stage2-bridge-v2'):
        path=evidence/name/'report.json'
        if read_shared_json(path).get('status')!='passed':raise ValueError('Native qualification missing: '+name)
        qualification[name]={'path':str(path.resolve()),'sha256':digest(path)}
    if digest(ROOT/'src/fly_touhou/thrl_rewards.py')!='5fd82a547c1cd7119ff69a4a56aacdf1cf4a4ae6f64f71388469c5a9e791fc9e':
        raise ValueError('THRL reward core changed')
    for label,path in (extra_qualification or {}).items():
        path=Path(path)
        if read_shared_json(path).get('status')!='passed':raise ValueError('Additional qualification missing: '+label)
        qualification[label]={'path':str(path.resolve()),'sha256':digest(path)}
    runtime.mkdir(parents=True,exist_ok=False);records=[]
    for folder in ('src','scripts','profiles'):
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.js','.cjs','.json'):continue
            content=path.read_bytes()
            if path.suffix=='.py':compile(content,str(path),'exec')
            target=runtime/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            records.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(content),'sha256':digest(target)})
    for relative in ('COPYING','README.md','pyproject.toml','data/connectome/model.py'):
        target=runtime/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
        records.append({'path':relative,'bytes':target.stat().st_size,'sha256':digest(target)})
    write(runtime/'source_manifest.json',{'schema':'thrl-control-engine-v1','origin':str(ROOT),
        'created_utc':utc_now(),'qualification':qualification,'files':records,
        'reward_sha256':digest(ROOT/'src/fly_touhou/thrl_rewards.py'),
        'scientific_training_core_changed':scientific_training_core_changed,
        'engineering_change':engineering_change or 'stage loading bridge, hidden startup, operator controls and independent primary validation'})
    return {'runtime':str(runtime),'manifest_sha256':digest(runtime/'source_manifest.json'),'qualification':qualification}


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('runtime','center','templates','import-study','import-control','tests','evidence','output-root'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    if args.center.exists():raise FileExistsError('Preserve the existing control registry/settings')
    templates=template_descriptors(args.templates)
    if any(is_full_story(read_shared_json(row['path'])) for row in templates.values()):
        raise ValueError('Full-story templates require the current qualified deployment; this legacy freeze '
            'has no private-desktop/software-D3D9 qualification. Use the qualified full-story deployment entry point.')
    engine=freeze(args.runtime.resolve(),args.tests,args.evidence)
    config={'schema':'thrl-control-center-v1','created_utc':utc_now(),'engine':engine,
        'python':sys.executable,'output_root':str(args.output_root.resolve()),'template_configs':templates,
        'standby_groups':list(templates)}
    args.center.mkdir(parents=True,exist_ok=False)
    write(args.center/'configuration.json',config);write(args.center/'settings.json',DEFAULT_SETTINGS)
    manifest=read_shared_json(args.import_study/'comparison_manifest.json')
    registry=imported_registry(manifest,templates,args.import_study,args.import_control)
    controller=ControlCenter(args.center);controller._publish(registry)
    write(ROOT/'artifacts/engineering/active_thrl_study.json',{'schema':'thrl-active-study-v1',
        'comparison_output':str(args.center.resolve()),
        'note':'本地控制面板：A0/A1 原数值＋地图，B0/B1 整理输入＋GRU，M0/M1 整理输入＋MLP；1 为教师组。'
            '默认每组8实例、每候选500万决策；完整采样对齐后实际5,001,216。修改设置只用于下一次启动。'
            '历史作业及检查点保留；面板会按实际进程身份显示状态。'})
    cmd='@echo off\r\nstart "" "'+str(Path(sys.executable).with_name('pythonw.exe'))+'" -B "'+str(ROOT/'scripts/monitor_thrl_comparison.py')+'" --active-study "'+str(ROOT/'artifacts/engineering/active_thrl_study.json')+'" --control-root "'+str(args.center.resolve())+'" --ui-state "'+str((args.center/'window.json').resolve())+'"\r\nexit /b\r\n'
    (ROOT/'打开训练监控.cmd').write_bytes(cmd.encode('ascii'))
    print(str(args.center.resolve()))


if __name__=='__main__':main()
