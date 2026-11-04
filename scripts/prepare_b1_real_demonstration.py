"""Freeze a display-only runtime and catalogue two actually cleared B1 policies."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_demo_control import digest
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def main():
    receipts=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    inherited=read(receipts/'BC-only-full-story-development20-v2-control/plan.json')
    evidence=Path(inherited['after_output'])
    whole=read(evidence/'report.json')
    if (whole['status']!='completed' or whole['native_games']!=140
            or digest(evidence/'report.json')!=inherited['after_report_sha256']):
        raise ValueError('Require the completed unchanged fixed140 comparison')
    baseline=Path(inherited['source_runtime'])
    runtime=baseline.parent/'20261025-B1-real-neural-demonstration-v3'
    changed=['src/fly_touhou/responsive_controller.py','src/fly_touhou/thrl_demo_control.py',
        'src/fly_touhou/thrl_visible_demo.py','src/fly_touhou/thrl_demo_ui.py',
        'src/fly_touhou/hooks/th14_demo_presentation.js',
        'scripts/run_b1_real_demonstration.py','scripts/qualify_b1_visible_menu.py',
        'tests/test_thrl_visible_demo.py','tests/test_thrl_demo_ui.py']
    existing=runtime.exists()
    if existing:
        manifest=read(runtime/'source_manifest.json')
        if manifest['schema']!='B1-frozen-real-neural-demo-runtime-v1':raise ValueError('Existing runtime is unrelated')
        for item in manifest['files']:
            if digest(runtime/item['path'])!=item['sha256']:raise ValueError('Existing frozen runtime changed')
        if any(digest(ROOT/name)!=digest(runtime/name) for name in changed):
            raise ValueError('Use a new runtime version for changed demonstration sources')
    else:
        runtime.mkdir(parents=True,exist_ok=False)
        for item in read(baseline/'source_manifest.json')['files']:
            source=baseline/item['path']
            if digest(source)!=item['sha256']:raise ValueError('Qualified base changed')
            target=runtime/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for name in changed:
            target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    relative='src/fly_touhou/responsive_controller.py'
    diff=''.join(difflib.unified_diff((baseline/relative).read_text(encoding='utf-8').splitlines(True),
        (runtime/relative).read_text(encoding='utf-8').splitlines(True),fromfile='qualified-controller',tofile='visible-demo-controller'))
    (receipts/'real-demo-visible-controller-diff-v3.patch').write_text(diff,encoding='utf-8')
    files=read(runtime/'source_manifest.json')['files'] if existing else [
        dict(path=str(p.relative_to(runtime)).replace('\\','/'),sha256=digest(p),bytes=p.stat().st_size)
        for p in sorted(runtime.rglob('*')) if p.is_file()]
    if not existing:
        write(runtime/'source_manifest.json',dict(schema='B1-frozen-real-neural-demo-runtime-v1',utc=utc_now(),
            baseline=str(baseline),baseline_manifest_sha256=digest(baseline/'source_manifest.json'),
            presentation_only_changes=changed,files=files,
            reward_PPO_observation_policy_and_gameplay_hooks_unchanged=True))
    models={}
    for key,label,sub,path,sha in [
        ('ancestor','B1 旧模型 · 7/20通关 · 平均推进5.8面（默认）','full_story_initial',
            inherited['parent'],'e303dbaf7d02332153b4e258d89bf0288cf05ebda4c8fec5a122578497ab6db4'),
        ('full_teacher_PPO','B1 完整教师+PPO · 7/20通关 · 平均推进5.5面','full_story_final',
            str(Path(inherited['checkpoint']).parents[1]/'full-teacher-joint-PPO-reboot-recovery-v2/trial_000_both/final.pt'),
            '180aa51d0e06b3c7ff5a9caa8462acb1322d59fa32c7bcf4a42f87cbab32855a')]:
        result=read(evidence/sub/'report.json')
        candidate=[r for r in result['episode_results'] if r['native_seed']==52158]
        if len(candidate)!=1 or candidate[0]['outcome']!='clear' or digest(path)!=sha:
            raise ValueError('Selected neural checkpoint needs a measured matching-seed clear')
        row=candidate[0]
        models[key]=dict(label=label,checkpoint=path,checkpoint_sha256=sha,qualified_neural_policy=True,
            formal_test_episodes=result['episodes'],formal_test_clears=result['clears'],
            mean_max_stage=result['mean_max_stage'],known_clear_native_seed=52158,
            known_clear_environment_seed=row['evaluation_seed'],report=str(evidence/sub/'report.json'),
            report_sha256=digest(evidence/sub/'report.json'))
    assert len({m['known_clear_environment_seed'] for m in models.values()})==1
    catalogue=dict(schema='B1-real-neural-demo-catalogue-v1',utc=utc_now(),source_runtime=str(runtime),
        source_manifest_sha256=digest(runtime/'source_manifest.json'),models=models,default_model='ancestor',
        selection_note='两组独立测试均为7/20，尚无通关率优胜者；默认旧模型，因同批测试平均推进更远。可自行切换新模型。',
        known_native_seed=52158,known_environment_seed=models['ancestor']['known_clear_environment_seed'],
        environment_config=inherited['environment_config'],environment_config_sha256=inherited['environment_config_sha256'],
        game_exe_sha256=inherited['game_exe_sha256'],profile_sha256=inherited['profile_sha256'],
        python=str(ROOT/'.venv/Scripts/python.exe'),
        output_root='./outputs/demonstrations/B1-real-neural-v1',
        temporary_directory='./outputs/tmp',maximum_original_worlds=30)
    panel_root=ROOT/'artifacts/engineering/real-model-demonstration-v1'
    panel_root.mkdir(parents=True,exist_ok=True)
    write(panel_root/'catalogue.json',catalogue)
    write(receipts/'real-demo-preparation-v3.json',dict(runtime=str(runtime),
        manifest_sha256=catalogue['source_manifest_sha256'],models=models,native_worlds_started=0,
        optimizer_updates=0,reward_modified=False))
    print(json.dumps(dict(runtime=str(runtime),files=len(files),manifest_sha256=catalogue['source_manifest_sha256'])))


if __name__=='__main__':main()
