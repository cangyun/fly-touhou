"""Publish bounded parallel gates; never modify a running frozen engine."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
from prepare_thrl_control_center import freeze


def main():
    directory=ROOT/'artifacts/engineering/parallel-gates-20261023';audit=directory/'deployment-v1'
    audit.mkdir(exist_ok=False)
    center=ROOT/'artifacts/engineering/training-control-center-v1';c=ControlCenter(center)
    before=copy.deepcopy(c.config);registry=c.registry();settings=(center/'settings.json').read_bytes()
    write(audit/'configuration.before.json',before);write(audit/'registry.before.json',registry)
    (audit/'settings.before.json').write_bytes(settings)
    qualification=Path('./outputs/engineering/parallel-gates-20261023/native-matched-v1/report.json')
    q=read(qualification)
    if q['status']!='passed' or not q['matched_actions_and_outcomes'] or q['measured_speedup']<=1:
        raise ValueError('Matched native execution/parity/speed qualification has not passed')
    previous=Path(before['engine']['runtime'])
    unchanged={}
    for file in ('thrl_ppo.py','thrl_model.py','thrl_curriculum.py','thrl_rewards.py','th14_completion_protocol.py','thrl_teacher.py','thrl_teacher_training.py','thrl_vector.py'):
        path=ROOT/'src/fly_touhou'/file
        if digest(path)!=digest(previous/'src/fly_touhou'/file):raise ValueError('Unexpected scientific/native change: '+file)
        unchanged[file]=digest(path)
    runtime=Path('./outputs/source_snapshots/20261023-parallel-gates-v1')
    engine=freeze(runtime,directory/'cpu-v4.xml',Path('./outputs/engineering/headless-20261022'),
        extra_qualification={'matched-native-parallel-gates':qualification},scientific_training_core_changed=False,
        engineering_change='8-worker ready-driven frozen course gates with batched policy inference and no reset barrier; an interrupted gate is retried from its recovered optimized checkpoint before further training. Native input interval, model, PPO, reward, curriculum fractions, probe counts and thresholds unchanged. Per-episode evaluation RNG protocol recorded; legacy serial bitwise equivalence not claimed.')
    after=copy.deepcopy(before);after.update(engine=engine,updated_utc=utc_now())
    for key,t in before['template_configs'].items():
        if digest(t['path'])!=t['sha256']:raise ValueError('Operator changed a template')
        cfg=read(t['path']);cfg.update(curriculum_eval_backend='ready-parallel-v1',curriculum_eval_workers=8,
            source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
            profile=str(runtime/'profiles/th14_1.00b_original.json'))
        cfg['profile_sha256']=digest(cfg['profile'])
        path=directory/f'branch_{key}_config.json';write(path,cfg)
        after['template_configs'][key]={**t,'path':str(path),'sha256':digest(path)}
    with controller_lock(center):
        if read(center/'configuration.json')!=before or c.registry()!=registry or (center/'settings.json').read_bytes()!=settings:
            raise ValueError('Operator state changed; preserve it and retry publication')
        write(center/'configuration.json',after);ControlCenter(center)._publish(registry)
    assert c.registry()==registry and (center/'settings.json').read_bytes()==settings
    write(audit/'report.json',dict(status='passed',created_utc=utc_now(),engine=engine,
        unchanged_sources=unchanged,settings_preserved=True,registry_preserved=True,
        native_benchmark=str(qualification),measured_small_benchmark_speedup=q['measured_speedup'],
        evaluation_workers=8,probe_counts_and_thresholds_unchanged=True,
        training_not_restarted_by_deployment=True,old_frozen_runtime_preserved=True))
    print(str(audit/'report.json'))


if __name__=='__main__':main()
