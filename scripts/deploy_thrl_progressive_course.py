"""Promote the qualified course runtime without changing saved panel settings."""
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
    for name in ('center','course','qualification','tests','runtime','audit'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from fly_touhou.thrl_study_io import read_shared_json,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_control import ControlCenter,controller_lock,digest,GROUPS
    from fly_touhou.thrl_curriculum import SCHEMA,DEFAULTS
    from fly_touhou.thrl_course_store import CourseStore
    from prepare_thrl_control_center import freeze
    proof=read_shared_json(a.qualification)
    if proof.get('status')!='passed' or proof.get('PPO_updates')!=96:
        raise ValueError('Actual native/CUDA qualification has not completed')
    for name,sha in proof['scientific_source_hashes'].items():
        if digest(ROOT/'src/fly_touhou'/name)!=sha:
            raise ValueError('Qualified scientific source changed: '+name)
    store=CourseStore(a.course)
    if not all(store.lessons()[i]['available'] for i in range(4)):
        raise ValueError('Complete first-stage phase data required')
    a.audit.mkdir(parents=True,exist_ok=False)
    before=read_shared_json(a.center/'configuration.json')
    settings=(a.center/'settings.json').read_bytes()
    write(a.audit/'configuration.before.json',before)
    (a.audit/'settings.before.json').write_bytes(settings)
    engine=freeze(a.runtime,a.tests,Path('./outputs/engineering/headless-20261022'),
        extra_qualification={'progressive-native-CUDA':a.qualification,
            'full-stage1-semantic-phases':a.course.parent/'pool-stage1-v5/report.json',
            'semantic-loading-barrier':a.course.parent/'loading-barrier-probe-v3/report.json'},
        scientific_training_core_changed=True,
        engineering_change='Progressive/rehearsal/native-chain sampling and recurrent burn-in/refresh; architecture, THRL rewards and PPO loss/hyperparameters unchanged; exact native phase history and variable-duration loading barriers')
    template_root=a.audit.parent/'prepared-v1';template_root.mkdir(exist_ok=False)
    templates={};initials={}
    old_templates=ROOT/'artifacts/engineering/nokozero-port-20261022/prepared-four-groups-v2'
    for key in GROUPS:
        cfg=read_shared_json(old_templates/f'branch_{key}_config.json')
        for name in ('start_bank','start_bank_sha256','full_story_fraction','segment_end','baseline_resume',
                     'teacher_pretraining_result','previous_study_output'):
            cfg.pop(name,None)
        cfg.update(curriculum_protocol=SCHEMA,curriculum_manifest=str(a.course.resolve()),
            curriculum_manifest_sha256=digest(a.course),curriculum_options=copy.deepcopy(DEFAULTS),
            source_runtime=engine['runtime'],source_manifest_sha256=engine['manifest_sha256'],
            profile=str(a.runtime/'profiles/th14_1.00b_original.json'),
            profile_sha256=digest(a.runtime/'profiles/th14_1.00b_original.json'),
            original_game_RNG_controlled=True,full_story_rng_controlled=True,
            new_initialization=True,old_checkpoints_used=False,
            curriculum_initialization_frame=64,
            user_authorization='User approved progressive original-game lessons, within-lesson random starts, one shared recurrent policy with rehearsal and opening chains; four input x teacher groups preserved')
        cfg['display_name']+=' · 渐进课程'
        path=template_root/f'branch_{key}_config.json';write(path,cfg)
        templates[key]=dict(path=str(path.resolve()),sha256=digest(path),input_schema=cfg['input_schema'],label=cfg['display_name'])
        initial=Path(cfg['warm_start_checkpoint'])
        if digest(initial)!=cfg['warm_start_sha256']:raise ValueError('Protected initial weights changed')
        initials[str(initial)]=digest(initial)
    after=copy.deepcopy(before)
    after.update(engine=engine,template_configs=templates,updated_utc=utc_now(),
        curriculum_qualification=dict(protocol=SCHEMA,report=str(a.qualification.resolve()),
            report_sha256=digest(a.qualification),course=str(a.course.resolve()),
            course_sha256=digest(a.course),store=store.summary(),tests=sum(int(x.get('tests',0)) for x in ET.parse(a.tests).getroot().iter('testsuite')),
            architecture_reward_PPO_formulas_unchanged=True,old_experiments_preserved=True))
    with controller_lock(a.center):
        if read_shared_json(a.center/'configuration.json')!=before:
            raise ValueError('Panel configuration changed during preparation; do not overwrite')
        if (a.center/'settings.json').read_bytes()!=settings:
            raise ValueError('Saved settings changed during preparation; preserve operator changes')
        write(a.center/'configuration.json',after)
        pointer=ROOT/'artifacts/engineering/active_thrl_study.json'
        data=read_shared_json(pointer)
        data['note']='渐进课程：当前段随机起点并逐步扩展，达到门槛后解锁下一段；同一策略复习旧段并从开局连打。A0/A1原输入，B0/B1整理输入；1为教师BC。旧作业记录保留，新课程使用新训练按钮。课程门槛不拒绝PPO更新。'
        write(pointer,data)
    if (a.center/'settings.json').read_bytes()!=settings:raise ValueError('Saved settings were modified')
    for name,sha in initials.items():
        if digest(Path(name))!=sha:raise ValueError('Initial weights changed during deployment')
    manifest=read_shared_json(a.runtime/'source_manifest.json')
    for entry in manifest['files']:
        if digest(a.runtime/entry['path'])!=entry['sha256']:raise ValueError('Frozen source copy differs')
    write(a.audit/'configuration.after.json',after)
    write(a.audit/'report.json',dict(status='passed',created_utc=utc_now(),engine=engine,
        previous_engine=before['engine'],templates=templates,store=store.summary(),
        settings_preserved=True,settings_sha256=hashlib.sha256(settings).hexdigest(),
        protected_initials=initials,source_files_verified=len(manifest['files']),
        training_started=False,formal_learning_gain_claimed=False))
    print(str(a.audit/'report.json'))


if __name__=='__main__':main()
