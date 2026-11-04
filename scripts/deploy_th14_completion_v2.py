"""Publish a reviewed v2 recipe and preserve every old job/template/checkpoint."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json,utc_now
from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
from fly_touhou.th14_completion_protocol import RECIPE,REWARD_SCHEMA,baseline_parameters
from fly_touhou.thrl_curriculum import configuration,SCHEMA_V2
from prepare_thrl_control_center import freeze


def main():
    directory=ROOT/'artifacts/engineering/completion-v2-20261023'
    audit=directory/'deployment-v1';audit.mkdir(exist_ok=False)
    center=ROOT/'artifacts/engineering/training-control-center-v1'
    c=ControlCenter(center);before=copy.deepcopy(c.config);registry=c.registry()
    settings=read_shared_json(center/'settings.json')
    write(audit/'configuration.before.json',before);write(audit/'registry.before.json',registry)
    write(audit/'settings.before.json',settings)
    qualification=directory/'offline-CUDA-v1/report.json'
    if read_shared_json(qualification)['synthetic_CUDA_PPO_Adam_updates']!=6:raise ValueError('CUDA check required')
    runtime=Path('./outputs/source_snapshots/20261023-completion-v2')
    engine=freeze(runtime,directory/'cpu-v2.xml',Path('./outputs/engineering/headless-20261022'),
        extra_qualification={'completion-v2-offline-CUDA':qualification},scientific_training_core_changed=True,
        engineering_change='Versioned completion reward, 50/50 opening and 40/20/40 later curriculum, intermediate chain probes, resource reserve gate, decision-budget entropy decay and gamma .999. PPO loss and native input/physics unchanged. Native functionality to be observed in authorized actual training; no separate stability test.')
    after=copy.deepcopy(before);after.update(engine=engine,updated_utc=utc_now())
    for key,entry in before['template_configs'].items():
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Template changed: '+key)
        cfg=read_shared_json(entry['path'])
        cfg.update(training_recipe=RECIPE,reward_schema=REWARD_SCHEMA,ppo_baseline=baseline_parameters({'training_recipe':RECIPE}),
            entropy_decay_decisions=3000000,curriculum_protocol=SCHEMA_V2,curriculum_options=configuration({'version':2}),
            source_runtime=str(runtime),source_manifest_sha256=engine['manifest_sha256'],
            profile=str(runtime/'profiles/th14_1.00b_original.json'),profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'),
            user_authorization='User authorized v2 curriculum/reward/exploration and actual CUDA training on 2026-10-23',
            experiment_comparison='Completion-v2; M0/M1 differ only by teacher BC; pre-BC initial checkpoint identical')
        for field in ('baseline_resume','previous_study_output','teacher_pretraining_result'):cfg.pop(field,None)
        path=directory/f'branch_{key}_config.json';write(path,cfg)
        after['template_configs'][key]={**entry,'path':str(path),'sha256':digest(path),'label':cfg['display_name']+' · 生存推进v2'}
    # Fewer but meaningful frozen full-opening development nodes. Stage gates retain 20+20 probes.
    updated_settings={**settings,'development_interval':250000,'development_episodes':20}
    with controller_lock(center):
        if read_shared_json(center/'configuration.json')!=before or c.registry()!=registry or read_shared_json(center/'settings.json')!=settings:
            raise ValueError('Operator changed the panel during preparation; preserve their changes')
        write(center/'configuration.json',after);write(center/'settings.json',updated_settings)
        ControlCenter(center)._publish(registry)
        write(ROOT/'artifacts/engineering/active_thrl_study.json',dict(schema='thrl-active-study-v1',comparison_output=str(center),
            note='生存推进v2：第一段当前50%/连打50%，后续40%/20%/40%；新版奖励、γ=.999、熵按决策衰减。M0/M1为MLP无/有教师主对照。有限残机、自主Bomb、不续关。旧日志与检查点保留，禁止跨版本静默续训。'))
    assert c.registry()==registry
    for entry in before['template_configs'].values():assert digest(entry['path'])==entry['sha256']
    a=read_shared_json(after['template_configs']['M0']['path']);b=read_shared_json(after['template_configs']['M1']['path'])
    assert a['warm_start_sha256']==b['warm_start_sha256'] and a['model_overrides']==b['model_overrides']
    assert a['teacher_pretraining'] is None and b['teacher_pretraining']['samples']==8192
    write(audit/'report.json',dict(status='passed',created_utc=utc_now(),engine=engine,
        reward_schema=REWARD_SCHEMA,curriculum_protocol=SCHEMA_V2,old_templates_preserved=True,registry_preserved=True,
        settings=updated_settings,training_jobs_started=0,original_game_processes_started=0,
        M0_M1_initialization_identical=True,actual_training_requires_separate_authorized_dispatch=True))
    print(str(audit/'report.json'))


if __name__=='__main__':main()
