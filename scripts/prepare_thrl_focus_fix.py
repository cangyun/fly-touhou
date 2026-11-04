"""Prepare a focus-fix candidate; only explicit --publish changes next-launch config.

Never starts/stops games, refreshes the panel, rewrites settings/registry, or
patches a running/frozen engine. Historical visibility tests do not qualify the
new runtime for foreground-focus safety.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_control import controller_lock,digest
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now

REPLACEMENTS=frozenset((
    'src/fly_touhou/hooks/th14_headless.js',
    'src/fly_touhou/hooks/th14_responsive.js',
    'src/fly_touhou/responsive_controller.py',
    'src/fly_touhou/thrl_native_env.py',
    'scripts/monitor_thrl_comparison.py',
))
ENVIRONMENT='src/fly_touhou/thrl_native_env.py'
DIAGNOSTIC_METHODS=frozenset(('_reset_episode_reward_diagnostics','_accumulate_reward_diagnostics'))
PROTECTED_CONTROL=('settings.json','registry.json','comparison_manifest.json')
TEMPLATE_PATH_FIELDS=frozenset(('source_runtime','source_manifest_sha256','profile'))


def sha(content):
    return hashlib.sha256(content).hexdigest()


def confined(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError('Manifest path escapes runtime: '+str(relative))
    return path


class _WithoutDiagnostics(ast.NodeTransformer):
    current_method=None

    def visit_FunctionDef(self,node):
        if node.name in DIAGNOSTIC_METHODS:return None
        previous=self.current_method;self.current_method=node.name
        try:return self.generic_visit(node)
        finally:self.current_method=previous

    def visit_Dict(self,node):
        if self.current_method in ('_info','_endpoint'):
            allowed={ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False)
                for expression in ('self.snapshot.extra.get("native_focus")',
                    'self.snapshot.extra.get("native_focus") if self.snapshot else None')}
            entries=[]
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value=='native_focus':
                    if ast.dump(value,include_attributes=False) not in allowed:
                        raise ValueError('Unexpected native_focus value outside read-only diagnostics')
                else:entries.append((key,value))
            node.keys=[key for key,_ in entries];node.values=[value for _,value in entries]
        return self.generic_visit(node)

    def visit_Expr(self,node):
        call=node.value
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and isinstance(call.func.value,ast.Name) and call.func.value.id=='self'
                and call.func.attr=='_accumulate_reward_diagnostics'):
            return None
        return self.generic_visit(node)


def verify_diagnostics_only(previous,current):
    """Gameplay code must be AST-identical outside two reviewed diagnostic methods."""
    trees=[ast.parse(raw) for raw in (previous,current)]
    for tree in trees:
        found={node.name for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
               and node.name in DIAGNOSTIC_METHODS}
        if found!=DIAGNOSTIC_METHODS:raise ValueError('Expected diagnostic method boundary absent')
    normalized=[ast.dump(_WithoutDiagnostics().visit(tree),include_attributes=False) for tree in trees]
    if normalized[0]!=normalized[1]:
        raise ValueError('Native environment changed outside diagnostic counters')
    return {'outside_diagnostic_methods_AST_identical':True,
            'reviewed_methods':sorted(DIAGNOSTIC_METHODS),
            'read_only_native_focus_output_fields':['THRLNativeEnv._info','THRLNativeEnv._endpoint'],
            'note':'Diagnostic method contents are separately reviewed/tested; this is not game parity proof.'}


def verify_tests(paths):
    reports=[]
    for source in paths:
        source=Path(source).resolve();root=ET.parse(source).getroot()
        suites=list(root.iter('testsuite'))
        total=sum(int(row.get('tests',0)) for row in suites)
        failures=sum(int(row.get('failures',0))+int(row.get('errors',0)) for row in suites)
        skipped=sum(int(row.get('skipped',0)) for row in suites)
        if not total or failures or skipped:
            raise ValueError('Offline tests must pass without skips: '+str(source))
        reports.append({'path':str(source),'sha256':digest(source),'tests':total,'skipped':skipped})
    if not reports:raise ValueError('At least one current offline JUnit report is required')
    return reports


def verify_manifest(runtime,expected_hash):
    runtime=Path(runtime).resolve();manifest_path=runtime/'source_manifest.json'
    if digest(manifest_path)!=expected_hash:raise ValueError('Frozen manifest identity changed')
    manifest=read(manifest_path);seen=set()
    for row in manifest['files']:
        if row['path'] in seen:raise ValueError('Duplicate manifest path')
        seen.add(row['path']);path=confined(runtime,row['path'])
        if digest(path)!=row['sha256']:raise ValueError('Frozen file identity changed: '+row['path'])
    return manifest


def prepare(*,source_root,center,runtime,candidate,tests):
    source_root,center,runtime,candidate=[Path(x).resolve() for x in (source_root,center,runtime,candidate)]
    if runtime.exists() or candidate.exists():raise FileExistsError('Candidate destinations must both be new')
    if runtime.is_relative_to(candidate) or candidate.is_relative_to(runtime):
        raise ValueError('Candidate audit and runtime must be separate directories')
    offline=verify_tests(tests)
    with controller_lock(center):
        before_raw=(center/'configuration.json').read_bytes()
        before=json.loads(before_raw)
        protected={name:(center/name).read_bytes() for name in PROTECTED_CONTROL if (center/name).exists()}
    old_runtime=Path(before['engine']['runtime']).resolve()
    if runtime.is_relative_to(old_runtime) or candidate.is_relative_to(old_runtime):
        raise ValueError('Never create artifacts inside the old frozen runtime')
    manifest=verify_manifest(old_runtime,before['engine']['manifest_sha256'])
    available={row['path'] for row in manifest['files']}
    if not REPLACEMENTS<=available:raise ValueError('Base runtime lacks a reviewed replacement file')
    replacement_bytes={name:confined(source_root,name).read_bytes() for name in REPLACEMENTS}
    environment=verify_diagnostics_only((old_runtime/ENVIRONMENT).read_bytes(),replacement_bytes[ENVIRONMENT])
    # All existing src files outside the explicit engineering allowlist must
    # match the active scientific engine, not merely a hand-picked short list.
    unchanged_sources={}
    for row in manifest['files']:
        name=row['path']
        if name.startswith('src/') and name not in REPLACEMENTS:
            if digest(confined(source_root,name))!=row['sha256']:
                raise ValueError('Unexpected scientific source change: '+name)
            unchanged_sources[name]=row['sha256']
    templates={}
    for key,item in before['template_configs'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Base template identity changed: '+key)
        cfg=read(item['path'])
        if cfg.get('training_recipe')!='th14-completion-v3':raise ValueError('Only current v3 templates supported')
        templates[key]=cfg
    candidate.mkdir(parents=True,exist_ok=False);runtime.mkdir(parents=True,exist_ok=False)
    (candidate/'configuration.before.json').write_bytes(before_raw)
    for name,raw in protected.items():(candidate/(name+'.before')).write_bytes(raw)
    records=[];changed=[]
    for row in manifest['files']:
        name=row['path'];data=replacement_bytes.get(name)
        if data is None:data=confined(old_runtime,name).read_bytes()
        if name not in REPLACEMENTS and sha(data)!=row['sha256']:
            raise ValueError('Base frozen source changed while copying: '+name)
        if name.endswith('.py'):compile(data,name,'exec')
        target=confined(runtime,name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        new_hash=sha(data)
        records.append({'path':name,'bytes':len(data),'sha256':new_hash})
        if new_hash!=row['sha256']:changed.append({'path':name,'before':row['sha256'],'after':new_hash})
    qualification={'offline_CPU':{'status':'passed','reports':offline},
        'original_game_focus':{'status':'not_run','accepted':False,
            'note':'No new-runtime original-game foreground event qualification has been run.'}}
    new_manifest={'schema':'thrl-control-engine-v1','origin':str(source_root),'created_utc':utc_now(),
        'files':records,'qualification':qualification,'scientific_training_core_changed':False,
        'reward_sha256':manifest.get('reward_sha256'),'engineering_change':
            'Process-local D3D no-window-changes, cursor suppression, foreground auditing and deathbomb opportunity diagnostics only; no new original-game qualification.',
        'parent_runtime':copy.deepcopy(before['engine']),
        'historical_qualification':{'scope':'Prior runtime only; not new focus-safety evidence',
            'records':copy.deepcopy(manifest.get('qualification',{}))},
        'native_environment_diagnostic_verification':environment,'changed_files':changed}
    write(runtime/'source_manifest.json',new_manifest)
    engine={'runtime':str(runtime),'manifest_sha256':digest(runtime/'source_manifest.json'),
            'qualification':qualification}
    after=copy.deepcopy(before);after.update(engine=engine,updated_utc=utc_now())
    after['focus_fix_candidate']={'native_focus_acceptance':'not_run','automatic_restart':False,
                                 'prepared_candidate':str(candidate)}
    prepared={}
    for key,old in templates.items():
        cfg=copy.deepcopy(old)
        cfg.update(source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
                   profile=str(runtime/'profiles/th14_1.00b_original.json'))
        if cfg['profile_sha256']!=digest(cfg['profile']):raise ValueError('Profile must stay byte-identical')
        differences={name for name in set(cfg)|set(old) if cfg.get(name)!=old.get(name)}
        if not differences<=TEMPLATE_PATH_FIELDS:raise ValueError('Scientific template fields changed')
        path=candidate/('branch_'+key+'_config.json');write(path,cfg)
        prepared[key]={'path':str(path),'sha256':digest(path),'path_fields_changed':sorted(differences)}
        after['template_configs'][key]={**before['template_configs'][key],
            'path':str(path),'sha256':digest(path)}
    write(candidate/'configuration.prepared.json',after)
    # Retain the original reports and their role, but never inherit native pass
    # labels as this candidate's qualifications.
    report={'schema':'thrl-focus-fix-candidate-v1','status':'prepared','created_utc':utc_now(),
        'center':str(center),'base_configuration_sha256':sha(before_raw),
        'prepared_configuration_sha256':digest(candidate/'configuration.prepared.json'),
        'engine':engine,'base_engine':copy.deepcopy(before['engine']),
        'unchanged_scientific_source_hashes':unchanged_sources,
        'changed_files':changed,'native_environment_diagnostic_verification':environment,
        'templates':prepared,'all_scientific_template_values_unchanged':True,
        'offline_CPU':offline,'native_focus_acceptance':'not_run','published':False,
        'historical_visibility_evidence_is_not_focus_qualification':True,
        'protected_control_at_prepare':{name:sha(raw) for name,raw in protected.items()},
        'training_jobs_started':0,'original_game_processes_started':0,'running_jobs_modified':False}
    verify_manifest(runtime,engine['manifest_sha256'])
    with controller_lock(center):
        if (center/'configuration.json').read_bytes()!=before_raw:
            raise ValueError('Control configuration changed during preparation; candidate not published')
        if any((center/name).read_bytes()!=raw for name,raw in protected.items()):
            raise ValueError('Operator control state changed during preparation; candidate not published')
    write(candidate/'report.json',report)
    return report


def publish(*,center,candidate):
    """Only replace future-launch configuration; preserve live job ownership/state."""
    center,candidate=Path(center).resolve(),Path(candidate).resolve()
    report=read(candidate/'report.json')
    if report.get('schema')!='thrl-focus-fix-candidate-v1' or report.get('status')!='prepared':
        raise ValueError('Expected an unmodified prepared candidate')
    if center!=Path(report['center']).resolve():raise ValueError('Candidate belongs to another control center')
    config_path=candidate/'configuration.prepared.json'
    if digest(config_path)!=report['prepared_configuration_sha256']:
        raise ValueError('Prepared configuration identity changed')
    after=read(config_path)
    if after['engine']!=report['engine']:raise ValueError('Prepared engine identity changed')
    verify_manifest(after['engine']['runtime'],after['engine']['manifest_sha256'])
    verify_manifest(report['base_engine']['runtime'],report['base_engine']['manifest_sha256'])
    before=read(candidate/'configuration.before.json')
    for key,item in before['template_configs'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Base template identity changed: '+key)
    for key,item in after['template_configs'].items():
        if digest(item['path'])!=item['sha256'] or item['sha256']!=report['templates'][key]['sha256']:
            raise ValueError('Prepared template identity changed: '+key)
    published_path=candidate/'publication.json'
    if published_path.exists():raise FileExistsError('Candidate has already been published')
    with controller_lock(center):
        if digest(center/'configuration.json')!=report['base_configuration_sha256']:
            raise ValueError('Control configuration changed since preparation; prepare a fresh candidate')
        # User settings and job records may legitimately change after prepare.
        # Preserve the current bytes, without restoring old snapshots.
        protected={name:(center/name).read_bytes() for name in PROTECTED_CONTROL if (center/name).exists()}
        write(center/'configuration.json',after)
        if read(center/'configuration.json')!=after:raise RuntimeError('Published configuration readback differs')
        if any((center/name).read_bytes()!=raw for name,raw in protected.items()):
            raise RuntimeError('Protected operator state changed during publication')
    result={'status':'published_candidate','created_utc':utc_now(),'native_focus_acceptance':'not_run',
        'configuration_sha256':digest(center/'configuration.json'),
        'protected_control_byte_hashes':{name:sha(raw) for name,raw in protected.items()},
        'training_jobs_started':0,'original_game_processes_started':0,
        'running_jobs_modified':False,'old_frozen_runtimes_preserved':True,
        'applies_to':'future new/resume submissions only; running processes retain old runtime'}
    write(published_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--center',type=Path,default=ROOT/'artifacts/engineering/training-control-center-v1')
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,help='New immutable directory; required for preparation')
    parser.add_argument('--tests',type=Path,nargs='+',help='Current passing offline JUnit XML reports')
    parser.add_argument('--publish',action='store_true',help='Explicitly publish an already prepared candidate')
    args=parser.parse_args()
    if args.publish:
        if args.runtime or args.tests:parser.error('--publish uses the existing candidate; omit --runtime/--tests')
        result=publish(center=args.center,candidate=args.candidate)
    else:
        if args.runtime is None or not args.tests:parser.error('Preparation requires --runtime and --tests')
        result=prepare(source_root=ROOT,center=args.center,runtime=args.runtime,
            candidate=args.candidate,tests=args.tests)
    print(json.dumps({'status':result['status'],'candidate':str(args.candidate.resolve()),
        'native_focus_acceptance':'not_run','training_started':False},ensure_ascii=False))


if __name__=='__main__':main()
