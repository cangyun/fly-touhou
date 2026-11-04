"""Prepare, never dispatch, the finite frozen sampling pilot after complete40."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    prerequisite=Path('./outputs/diagnostics/B1-autonomous-dodge-20261024/full-geometry-paired-development40-v2')
    prior=read(prerequisite/'plan.json');base=Path(prior['source_runtime'])
    prepared=r/'prepared-frozen-inference-temperature-v1';receipt=read(prepared/'receipt.json')
    if receipt['total_passed']!=37 or receipt['status']!='qualified_API_only_not_deployed':
        raise ValueError('Qualified unchanged-default evaluation temperature interface required')
    runtime=base.parent/'20261025-B1-frozen-sampling-temperature-pilot-v3';runtime.mkdir(exist_ok=False)
    for row in read(base/'source_manifest.json')['files']:
        if sha(base/row['path'])!=row['sha256']:raise ValueError('Qualified native source changed')
        target=runtime/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/row['path'],target)
    changed=['src/fly_touhou/thrl_full_story_evaluation.py','scripts/evaluate_b1_sampling_temperature_pilot.py',
             'tests/test_thrl_full_story_temperature.py','tests/test_b1_sampling_temperature_pilot.py']
    for name in changed:
        target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    files=[dict(path=str(p.relative_to(runtime)).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(runtime.rglob('*')) if p.is_file()]
    write(runtime/'source_manifest.json',dict(schema='B1-frozen-sampling-temperature-pilot-runtime-v1',utc=utc_now(),
        baseline=str(base),baseline_sha256=sha(base/'source_manifest.json'),changed_files=changed,files=files,
        original_rules_observations_reward_and_PPO_unchanged=True))
    control=r/'frozen-sampling-temperature-pilot-v3-control';control.mkdir(exist_ok=False);plan_path=control/'plan.json'
    plan=dict(schema='B1-frozen-sampling-temperature-pilot-plan-v1',source_runtime=str(runtime),
        source_manifest_sha256=sha(runtime/'source_manifest.json'),plan_path=str(plan_path.resolve()),
        checkpoint=prior['policies']['candidate']['checkpoint'],checkpoint_sha256=prior['policies']['candidate']['sha256'],
        prior_report=str(prerequisite/'report.json'),prior_plan=str(prerequisite/'plan.json'),prior_plan_sha256=sha(prerequisite/'plan.json'),
        expected_prior_native_seeds=prior['expected_native_seeds'],expected_native_seeds=prior['expected_native_seeds'][:14],
        seed=prior['seed'],environment_config=prior['environment_config'],environment_config_sha256=prior['environment_config_sha256'],
        game_exe_sha256=prior['game_exe_sha256'],profile_sha256=prior['profile_sha256'],workers_each=14,episodes_each=14,
        arms={'T1':1.,'T05':.5},maximum_original_worlds=30,policy_actions=19,action_selection='sampled',
        native_seed_split='development',native_frame_limit=None,continues=0,
        output='./outputs/diagnostics/B1-autonomous-dodge-20261024/frozen-sampling-temperature-development28-v3',
        control_center_root=prior['control_center_root'],panel_job_id='20261025-B1-frozen-sampling-temperature-pilot-v3',python=prior['python'],
        managed_demo_control_root=str(ROOT/'artifacts/engineering/real-model-demonstration-v1'),
        no_PPO_sampler_or_reward_change=True,dispatch_requires_preferable_complete_calibrated_result_and_prior_pool_exit=True)
    write(plan_path,plan)
    temporary='./outputs/tmp'
    write(control/'background_spec.json',dict(schema='fly-touhou-background-job-v1',python=plan['python'],
        script=str(runtime/'scripts/evaluate_b1_sampling_temperature_pilot.py'),arguments=['--plan',str(plan_path.resolve())],
        cwd=str(runtime),control_directory=str(control.resolve()),environment=dict(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        TEMP=temporary,TMP=temporary,TMPDIR=temporary)))
    write(r/'frozen-sampling-temperature-pilot-preparation-v3.json',dict(utc=utc_now(),status='prepared_not_dispatched',
        source_runtime=str(runtime),source_manifest_sha256=plan['source_manifest_sha256'],files=len(files),plan=str(plan_path),
        prospective_original_worlds=28,actual_native_worlds_started=0,new_optimizer_updates=0,new_monitors=0,
        managed_demo_reserved_without_modification=True,dispatch_requires_prior_completion_and_exit=True))
    print(json.dumps(dict(runtime=str(runtime),files=len(files),plan=str(plan_path)),ensure_ascii=False))


if __name__=='__main__':main()
