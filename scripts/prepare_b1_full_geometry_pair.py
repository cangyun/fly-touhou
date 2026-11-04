"""Freeze a finite30-world paired development comparison after calibration."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_full_geometry_data import digest
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_control import alive


def main():
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    prior=read(r/'BC-only-full-story-development20-v2-control/plan.json')
    fitplan=read(r/'full-geometry-risk-calibration-v1-control/plan.json')
    fit=Path(fitplan['output']);result=read(fit/'result.json')
    if (read(r/'full-geometry-risk-calibration-v1-control/status.json').get('exit_code')!=0
            or alive(read(fit/'launch.json')) or result['status']!='completed'
            or not result['selected_proxy_improved']):
        raise RuntimeError('Require a completed, exited, improved risk calibration before native comparison')
    base=Path(fitplan['source_runtime']);runtime=base.parent/'20261025-B1-full-geometry-paired-development-v2'
    runtime.mkdir(parents=True,exist_ok=False)
    for row in read(base/'source_manifest.json')['files']:
        if digest(base/row['path'])!=row['sha256']:raise ValueError('Qualified calibration source changed')
        target=runtime/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/row['path'],target)
    changed=['scripts/evaluate_b1_full_geometry_pair.py','tests/test_b1_full_geometry_pair.py']
    for name in changed:
        target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    files=[dict(path=str(p.relative_to(runtime)).replace('\\','/'),sha256=digest(p),bytes=p.stat().st_size)
        for p in sorted(runtime.rglob('*')) if p.is_file()]
    write(runtime/'source_manifest.json',dict(schema='B1-frozen-full-geometry-paired-development-v1',utc=utc_now(),
        baseline=str(base),baseline_sha256=digest(base/'source_manifest.json'),changed_files=changed,files=files,
        original_native_execution_input_reward_and_PPO_unchanged=True))
    control=r/'full-geometry-paired-development-v2-control';control.mkdir(exist_ok=False)
    plan_path=control/'plan.json'
    plan=dict(schema='B1-full-geometry-paired-development-plan-v1',source_runtime=str(runtime),
        source_manifest_sha256=digest(runtime/'source_manifest.json'),plan_path=str(plan_path.resolve()),
        policies=dict(baseline=dict(checkpoint=fitplan['parent'],sha256=fitplan['parent_sha256']),
            candidate=dict(checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'])),
        calibration_result=str(fit/'result.json'),calibration_result_sha256=digest(fit/'result.json'),
        environment_config=prior['environment_config'],environment_config_sha256=prior['environment_config_sha256'],
        game_exe_sha256=prior['game_exe_sha256'],profile_sha256=prior['profile_sha256'],
        output='./outputs/diagnostics/B1-autonomous-dodge-20261024/full-geometry-paired-development40-v2',
        seed=prior['seed'],expected_native_seeds=prior['expected_native_seeds'],workers_each=14,episodes_each=20,
        maximum_original_worlds=30,action_selection='greedy',policy_actions=19,native_frame_limit=None,
        continues=0,native_seed_split='development',teacher_actions=0,optimizer_updates=0,
        reward_modified=False,base_actor_GRU_critic_laser_frozen=True,
        managed_demo_control_root=str(ROOT/'artifacts/engineering/real-model-demonstration-v1'),
        control_center_root=prior['control_center_root'],panel_job_id='20261025-B1-full-geometry-paired-development40-v2',
        python=str(ROOT/'.venv/Scripts/python.exe'))
    write(plan_path,plan)
    temporary='./outputs/tmp'
    write(control/'background_spec.json',dict(schema='fly-touhou-background-job-v1',python=plan['python'],
        script=str(runtime/'scripts/evaluate_b1_full_geometry_pair.py'),arguments=['--plan',str(plan_path.resolve())],
        cwd=str(runtime),control_directory=str(control.resolve()),environment=dict(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TEMP=temporary,TMP=temporary,TMPDIR=temporary)))
    write(r/'full-geometry-paired-development-preparation-v2.json',dict(utc=utc_now(),source_runtime=str(runtime),
        source_manifest_sha256=plan['source_manifest_sha256'],files=len(files),plan=str(plan_path.resolve()),
        prospective_owned_original_worlds=28,maximum_original_worlds=30,managed_demo_reserved=True,
        native_worlds_started=0,new_optimizer_updates=0,process_launched=False))
    print(json.dumps(dict(runtime=str(runtime),manifest_sha256=plan['source_manifest_sha256'],files=len(files),
        spec=str(control/'background_spec.json')),ensure_ascii=False))


if __name__=='__main__':main()
