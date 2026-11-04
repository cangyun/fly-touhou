"""Prepare an un-published D95 demo catalogue and a new display-only runtime.

Never starts a process or modifies an existing catalogue/demo/control marker.
"""
from pathlib import Path
import argparse
import copy
import difflib
import json
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_demo_control import CATALOGUE_V1, CATALOGUE_V2, digest, validate_selection, make_launch_plan
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now

D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
D95_REPORT_SHA = 'a287e30ca2081fa93abfd76b7ff5f11422cefe6585cfce511354613aae63a56c'
S486_MANIFEST_SHA = '7678b3db16d0f54d0a1d0a7f765d8a253a73e00f7e0b5a483e1f8fe086c30a37'
OVERLAYS = ('src/fly_touhou/responsive_controller.py', 'src/fly_touhou/thrl_demo_control.py',
            'src/fly_touhou/thrl_visible_demo.py', 'src/fly_touhou/thrl_demo_ui.py',
            'src/fly_touhou/hooks/th14_demo_presentation.js', 'scripts/run_b1_real_demonstration.py')


def checked_relative(root, relative):
    path = Path(root)/relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Source file leaves its frozen root')
    return path


def freeze_display_runtime(baseline, runtime, workspace=ROOT):
    baseline, runtime, workspace = Path(baseline), Path(runtime), Path(workspace)
    if digest(baseline/'source_manifest.json') != S486_MANIFEST_SHA:
        raise ValueError('D95 needs the pinned S486 source manifest')
    manifest = read(baseline/'source_manifest.json')
    overlay_pins = {name:digest(workspace/name) for name in OVERLAYS}
    if runtime.exists():
        frozen = read(runtime/'source_manifest.json')
        if (frozen.get('schema') != 'B1-frozen-D95-real-demo-runtime-v1'
                or frozen.get('baseline_manifest_sha256') != S486_MANIFEST_SHA
                or frozen.get('display_overlay_sha256') != overlay_pins):
            raise ValueError('Use a new runtime version; existing runtime is immutable')
        for item in frozen['files']:
            if digest(checked_relative(runtime,item['path'])) != item['sha256']:
                raise ValueError('Existing frozen display runtime changed')
        return frozen
    runtime.mkdir(parents=True,exist_ok=False)
    for item in manifest['files']:
        source = checked_relative(baseline,item['path'])
        if digest(source) != item['sha256']:
            raise ValueError('Pinned S486 source file changed: '+item['path'])
        if item['path'] in OVERLAYS:
            continue
        target = checked_relative(runtime,item['path'])
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name in OVERLAYS:
        target = checked_relative(runtime,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(workspace/name,target)
        if digest(target) != overlay_pins[name]:
            raise ValueError('Display source changed during freeze')
    for item in manifest['files']:
        if item['path'] not in OVERLAYS and digest(runtime/item['path']) != item['sha256']:
            raise ValueError('A non-presentation S486 file changed')
    files = [dict(path=path.relative_to(runtime).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
             for path in sorted(runtime.rglob('*')) if path.is_file()]
    frozen = dict(schema='B1-frozen-D95-real-demo-runtime-v1',utc=utc_now(),baseline=str(baseline),
        baseline_manifest_sha256=S486_MANIFEST_SHA,display_overlay_sha256=overlay_pins,files=files,
        policy_observation_reward_PPO_unchanged_from_S486=True,
        old_demo_runtime_modified=False,original_games_started=0,optimizer_updates=0)
    write(runtime/'source_manifest.json',frozen)
    return frozen


def build_catalogue(old, development_plan, report_path, plan_path, runtime):
    if old.get('schema') != CATALOGUE_V1 or set(old['models']) != {'ancestor','full_teacher_PPO'}:
        raise ValueError('Only the two recorded legacy B1 entries may be inherited')
    if (digest(report_path) != D95_REPORT_SHA or development_plan['episodes_each'] != 20
            or development_plan['native_seed_split'] != 'development'
            or development_plan['policies']['candidate']['sha256'] != D95_SHA
            or development_plan['source_manifest_sha256'] != S486_MANIFEST_SHA):
        raise ValueError('Require the pinned complete20-game D95 development evidence')
    catalogue = copy.deepcopy(old)
    catalogue.update(schema=CATALOGUE_V2,utc=utc_now(),default_model=old['default_model'],
        selection_note='保留旧A/C及原默认项。新增D95尺寸校准候选：开发评估15/20通关，尚非正式稳定验收；可自行选择。',
        model_specific_evidence_seed_and_runtime=True,stable_completion_claimed=False,
        candidate_catalogue_only=True)
    for key, entry in catalogue['models'].items():
        entry.update(model_family='B1',evidence_scope='formal',native_seed_split='test',
            evaluation_episodes=entry['formal_test_episodes'],evaluation_clears=entry['formal_test_clears'],
            source_runtime=old['source_runtime'],source_manifest_sha256=old['source_manifest_sha256'],
            launch_catalogue_schema=CATALOGUE_V1,stable_completion_claimed=False)
    catalogue['models']['d95_geometry'] = dict(
        label='D95 尺寸校准 · 开发评估15/20通关（非正式稳定验收）',model_family='B1',
        checkpoint=development_plan['policies']['candidate']['checkpoint'],checkpoint_sha256=D95_SHA,
        qualified_neural_policy=True,evidence_scope='development',native_seed_split='development',
        evaluation_episodes=20,evaluation_clears=15,stable_completion_claimed=False,
        known_clear_native_seed=48556,known_clear_environment_seed=616040764,
        report=str(report_path),report_sha256=D95_REPORT_SHA,
        evaluation_plan=str(plan_path),evaluation_plan_sha256=digest(plan_path),evaluation_policy_key='candidate',
        policy_source_runtime=development_plan['source_runtime'],
        policy_source_manifest_sha256=S486_MANIFEST_SHA,
        source_runtime=str(runtime),source_manifest_sha256=digest(Path(runtime)/'source_manifest.json'),
        launch_catalogue_schema=CATALOGUE_V2,
        evidence_note='完整20局开发结果；已知通关开局仅用于展示，不计入新的独立测试。')
    for key in catalogue['models']:
        validate_selection(catalogue,key)
    return catalogue


def main():
    receipts = ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--old-catalogue',type=Path,default=ROOT/'artifacts/engineering/real-model-demonstration-v1/catalogue.json')
    parser.add_argument('--output',type=Path,default=receipts/'fly-port-feasibility-v1/distillation-design/d95-demo-candidate-v1')
    parser.add_argument('--runtime',type=Path,default=Path('./outputs/source_snapshots/20261026-B1-D95-real-neural-demonstration-v4'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Candidate output must be new; existing evidence is preserved')
    plan_path = receipts/'full-geometry-complete-paired-development-review-v1/sealed_inputs/plan.json'
    report_path = receipts/'full-geometry-complete-paired-development-review-v1/sealed_inputs/candidate/report.json'
    old = read(args.old_catalogue)
    old_catalogue_sha = digest(args.old_catalogue)
    legacy_runtime = Path(old['source_runtime'])
    legacy_manifest_sha = digest(legacy_runtime/'source_manifest.json')
    legacy_files = {item['path']:digest(legacy_runtime/item['path']) for item in read(legacy_runtime/'source_manifest.json')['files']}
    plan = read(plan_path)
    if digest(plan['policies']['candidate']['checkpoint']) != D95_SHA:
        raise ValueError('D95 checkpoint identity changed')
    freeze_display_runtime(plan['source_runtime'],args.runtime)
    catalogue = build_catalogue(old,plan,report_path,plan_path,args.runtime)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'catalogue.candidate.json',catalogue)
    for key in catalogue['models']:
        write(args.output/('launch-plan-'+key+'.json'),make_launch_plan(catalogue,key,args.output/('review-only-'+key)))
    relative = 'src/fly_touhou/responsive_controller.py'
    difference = ''.join(difflib.unified_diff((Path(plan['source_runtime'])/relative).read_text(encoding='utf-8').splitlines(True),
        (args.runtime/relative).read_text(encoding='utf-8').splitlines(True),fromfile='S486-controller',tofile='D95-visible-controller'))
    (args.output/'presentation-only-controller.patch').write_text(difference,encoding='utf-8')
    if (digest(args.old_catalogue) != old_catalogue_sha or digest(legacy_runtime/'source_manifest.json') != legacy_manifest_sha
            or any(digest(legacy_runtime/name) != value for name,value in legacy_files.items())):
        raise RuntimeError('A preserved catalogue/runtime changed during preparation')
    write(args.output/'preparation.json',dict(schema='B1-D95-demo-candidate-preparation-v1',utc=utc_now(),
        candidate_catalogue=str(args.output/'catalogue.candidate.json'),catalogue_sha256=digest(args.output/'catalogue.candidate.json'),
        runtime=str(args.runtime),source_manifest_sha256=digest(args.runtime/'source_manifest.json'),
        old_catalogue_unchanged=True,old_catalogue_sha256=old_catalogue_sha,
        old_runtime_unchanged=True,old_runtime_manifest_sha256=legacy_manifest_sha,
        default_model_unchanged=catalogue['default_model']==old['default_model'],
        D95_checkpoint_sha256=D95_SHA,D95_report_sha256=D95_REPORT_SHA,
        D95_native_seed=48556,D95_environment_seed=616040764,D95_evidence_scope='development',
        original_games_started=0,UI_started=False,active_control_records_modified=False,
        catalogue_published=False,optimizer_updates=0,reward_PPO_modified=False))
    print(json.dumps(dict(candidate=str(args.output/'catalogue.candidate.json'),runtime=str(args.runtime),
        published=False,original_games_started=0),ensure_ascii=False))


if __name__ == '__main__':
    main()
