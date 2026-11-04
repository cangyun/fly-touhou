"""Add an idle M0 or M1 panel group without launching training or altering old experiments."""
from pathlib import Path
import argparse
import copy
import hashlib
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('center','qualification','tests','runtime','audit'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--group',choices=('M0','M1'),default='M0')
    p.add_argument('--teacher-tests',type=Path)
    a=p.parse_args()
    group=a.group
    from fly_touhou.thrl_control import ControlCenter,controller_lock,digest
    from fly_touhou.thrl_study_io import read_shared_json,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from prepare_thrl_control_center import freeze
    q=read_shared_json(a.qualification)
    if q['status']!='passed' or q['offline_CUDA']['synthetic_Adam_updates']!=6:
        raise ValueError('Offline qualification has not passed')
    c=ControlCenter(a.center);before=copy.deepcopy(c.config);registry=c.registry()
    settings=(a.center/'settings.json').read_bytes()
    if group in c.config['template_configs']:raise ValueError(group+' already exists; do not overwrite')
    teacher=None;extra={'matched-MLP-offline-CUDA':a.qualification}
    if group=='M1':
        if not a.teacher_tests:raise ValueError('M1 requires the real MLP BC/checkpoint regression checks')
        suites=list(ET.parse(a.teacher_tests).getroot().iter('testsuite'))
        cases=[case.get('name','') for case in ET.parse(a.teacher_tests).getroot().iter('testcase')]
        if (not any('test_soft_BC_leaves_critic_and_PPO_Adam_unchanged_and_checkpoint_resumes_lifetime[mlp]' in case for case in cases)
            or not any('test_mlp_groups_can_be_published_and_prepared_without_launching_a_game[M1]' in case for case in cases)
            or any(int(row.get('failures',0))+int(row.get('errors',0)) for row in suites)):
            raise ValueError('MLP teacher update/checkpoint and M1 dispatch checks must pass')
        teacher=read_shared_json(before['template_configs']['B1']['path'])['teacher_pretraining']
    a.audit.mkdir(parents=True,exist_ok=False)
    write(a.audit/'configuration.before.json',before)
    write(a.audit/'registry.before.json',registry)
    (a.audit/'settings.before.json').write_bytes(settings)
    b0=read_shared_json(before['template_configs']['M0' if group=='M1' else 'B0']['path'])
    initial=Path(q['initialization']['path'])
    if digest(initial)!=q['initialization']['sha256']:raise ValueError('MLP initialization changed')
    if group=='M1' and (b0['warm_start_sha256']!=digest(initial) or b0['teacher_pretraining'] is not None):
        raise ValueError('M1 must start from exactly the matched M0 initialization before BC')
    # Check the real checkpoint through the same loader used by future jobs.
    import torch
    torch.set_num_threads(1)
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    from fly_touhou.thrl_branching import warm_start
    overrides={**b0['model_overrides'],'temporal_core':'mlp','mlp_hidden_size':768}
    model=MOActorCritic(attention='both',action_dim=19,**overrides)
    warm=warm_start(torch,model,make_optimizer(model,PPOConfig()),initial,q['initialization']['sha256'],exact_optimizer=True)
    if group=='M1':
        qualification=a.audit/'teacher_offline_qualification.json'
        write(qualification,dict(status='passed',created_utc=utc_now(),
            tests=str(a.teacher_tests.resolve()),tests_sha256=digest(a.teacher_tests),
            tests_passed=sum(int(row.get('tests',0)) for row in suites),
            matched_initialization_sha256=digest(initial),real_checkpoint_loader_passed=True,
            synthetic_CPU_BC_and_checkpoint_roundtrip=True,critic_and_PPO_optimizer_unchanged_by_BC=True,
            actual_teacher_collection_performed=False,actual_teacher_model_fit_performed=False,
            native_processes_started=0,training_jobs_started=0))
        extra['MLP-teacher-offline-CPU']=qualification
    engine=freeze(a.runtime,a.tests,Path('./outputs/engineering/headless-20261022'),
        extra_qualification=extra,
        scientific_training_core_changed=True,
        engineering_change='Matched feedforward M0/M1 groups; M1 adds the existing teacher BC pipeline to M0 with the same pre-BC initialization. MLP BC/checkpoint and dispatch checked offline. Native execution, PPO loss, rewards and curriculum unchanged. No original-game run or training launched.')
    cfg=copy.deepcopy(b0)
    cfg.update(branch=group,display_name='整理输入＋MLP · '+('有教师' if teacher else '无教师'),model_overrides=overrides,
        teacher_pretraining=teacher,warm_start_checkpoint=str(initial),warm_start_sha256=digest(initial),
        source_runtime=engine['runtime'],source_manifest_sha256=engine['manifest_sha256'],
        profile=str(a.runtime/'profiles/th14_1.00b_original.json'),
        profile_sha256=digest(a.runtime/'profiles/th14_1.00b_original.json'),
        parameter_counts=q['parameter_counts'],new_initialization=True,old_checkpoints_used=False,
        experiment_comparison=('M1 vs M0; identical initial weights, inputs, MLP/attention, PPO, reward and curriculum; M1 alone adds teacher BC'
            if teacher else 'M0 vs B0; replace only GRU384->256 with MLP384->768->256; identical shared initial encoder/head tensors; no BC'),
        user_authorization='User requested an additional MLP experiment'+(' with teacher pretraining' if teacher else '')+'; no automatic training launch')
    for key in ('baseline_resume','teacher_pretraining_result','previous_study_output','model_config'):
        cfg.pop(key,None)
    template_path=a.audit.parent/f'branch_{group}_config.json';write(template_path,cfg)
    after=copy.deepcopy(before)
    after.update(engine=engine,updated_utc=utc_now())
    after['template_configs'][group]=dict(path=str(template_path.resolve()),sha256=digest(template_path),
        input_schema=cfg['input_schema'],label=cfg['display_name'])
    after['standby_groups']=list(dict.fromkeys(after.get('standby_groups',[])+[group]))
    after['MLP_teacher_qualification' if teacher else 'MLP_qualification']=dict(
        report=str((qualification if teacher else a.qualification).resolve()),
        report_sha256=digest(qualification if teacher else a.qualification),
        core='384->768->256',parameter_counts=q['parameter_counts'],teacher_pretraining=bool(teacher),
        original_game_validation_run=False,training_started=False)
    with controller_lock(a.center):
        if read_shared_json(a.center/'configuration.json')!=before:raise ValueError('Panel configuration changed')
        if (a.center/'settings.json').read_bytes()!=settings:raise ValueError('Preserve newly edited settings')
        if c.registry()!=registry:raise ValueError('Operator changed jobs; retry publication without replacing them')
        write(a.center/'configuration.json',after)
        promoted=ControlCenter(a.center);promoted._publish(registry)
        pointer=ROOT/'artifacts/engineering/active_thrl_study.json';active=read_shared_json(pointer)
        active['note']='新增'+group+'：整理输入/CNN/空间和对象注意力＋同规模MLP，'+('先教师示范预训练再自主PPO。M0/M1预训练前初始化完全相同。' if teacher else '无教师，无历史预热。')+'待手动启动；既有组、奖励、课程、设置和历史检查点保留。'
        write(pointer,active)
    assert promoted.registry()==registry and (a.center/'settings.json').read_bytes()==settings
    # All four original templates and all prior frozen files remain untouched.
    for key,t in before['template_configs'].items():
        assert digest(Path(t['path']))==t['sha256'] and after['template_configs'][key]==t
    for f in read_shared_json(a.runtime/'source_manifest.json')['files']:
        assert digest(a.runtime/f['path'])==f['sha256']
    write(a.audit/'configuration.after.json',after)
    write(a.audit/'report.json',dict(status='passed',created_utc=utc_now(),engine=engine,
        group=group,template=str(template_path.resolve()),parameter_counts=q['parameter_counts'],
        real_checkpoint_loader_passed=True,warm_start_audit=warm,
        settings_preserved=True,old_templates_preserved=True,registry_preserved=True,
        old_checkpoint_paths_preserved=True,training_jobs_created=0,original_game_processes_started=0,
        original_game_validation_run=False,learning_gain_claimed=False))
    print(str(a.audit/'report.json'))


if __name__=='__main__':main()
