"""Prepare an isolated, finite point-risk calibration using training data only."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_full_geometry_data import digest,partition_seeds
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write


def main():
    import numpy as np
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    prior=read(r/'BC-only-full-story-development20-v2-control/plan.json')
    base=Path(prior['source_runtime']);runtime=base.parent/'20261025-B1-full-geometry-risk-calibration-v1'
    b=Path('./outputs/engineering/B1-autonomous-dodge-20261024')
    parent=b/'full-teacher-joint-PPO-reboot-recovery-v2/trial_000_both/final.pt'
    parent_sha='180aa51d0e06b3c7ff5a9caa8462acb1322d59fa32c7bcf4a42f87cbab32855a'
    if digest(parent)!=parent_sha:raise ValueError('The exact completed neural parent changed')
    archive=b/'teacher-full-natural30-v1'
    strata=read(Path('./outputs/diagnostics/B1-autonomous-dodge-20261024/BC-heldout-stratification-v1/report.json'))
    if digest(archive/'demonstrations.json')!=strata['teacher_manifest_sha256']:
        raise ValueError('The complete original training archive changed')
    native=[]
    for p in (archive/'trajectories').glob('world*/index.json'):
        index=read(p)
        if index['status']!='completed' or index['natural_episode_complete'] is not True:raise ValueError('Incomplete native source')
        native.append(index['metadata']['native_seed'])
    old_seen=set()
    for name in ['teacher30-v1','student-aggregation-v1']:
        old_seen.update(np.load(b/name/'demonstrations/native_seeds.npy',mmap_mode='r').reshape(-1).tolist())
    preferred=strata['original_heldout_native_seeds']
    held=[seed for seed in preferred if seed not in old_seen]
    for seed in sorted(set(native)-old_seen-set(held)):
        if len(held)==6:break
        held.append(seed)
    partition_seeds(native,held)
    runtime.mkdir(parents=True,exist_ok=False)
    for row in read(base/'source_manifest.json')['files']:
        if digest(base/row['path'])!=row['sha256']:raise ValueError('Qualified base source changed')
        target=runtime/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/row['path'],target)
    changed=['src/fly_touhou/thrl_full_geometry_data.py','scripts/train_b1_full_geometry_risk.py',
        'scripts/register_b1_goal_phase.py','tests/test_thrl_full_geometry_data.py']
    for name in changed:
        target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    files=[dict(path=str(p.relative_to(runtime)).replace('\\','/'),sha256=digest(p),bytes=p.stat().st_size)
        for p in sorted(runtime.rglob('*')) if p.is_file()]
    write(runtime/'source_manifest.json',dict(schema='B1-frozen-full-geometry-calibration-v1',utc=utc_now(),
        baseline=str(base),baseline_sha256=digest(base/'source_manifest.json'),changed_files=changed,
        native_PPO_reward_input_and_policy_architecture_unchanged=True,files=files))
    control=r/'full-geometry-risk-calibration-v1-control';control.mkdir(exist_ok=False)
    plan=dict(schema='B1-full-geometry-point-risk-calibration-plan-v1',source_runtime=str(runtime),
        source_manifest_sha256=digest(runtime/'source_manifest.json'),parent=str(parent),parent_sha256=parent_sha,
        archive=str(archive),archive_manifest_sha256=digest(archive/'demonstrations.json'),
        held_native_seeds=held,old_point_source_overlap_with_held=sorted(old_seen&set(held)),
        environment_config=prior['environment_config'],output=str(b/'full-geometry-point-risk-calibration-v1'),
        seed=2026102543,epochs_max=40,steps_per_epoch=200,max_risk_fit_Adam=8000,
        synthetic_player_circle_envelope=[2,24],synthetic_player_half_extents_envelope=[1,3],
        synthetic_geometries_not_claimed_as_native_trajectories=True,
        independent_test_failure_states_used=False,PPO_optimizer_updates=0,original_worlds=0,
        original_worlds_next_paired_development=[15,15],maximum_original_worlds=30,
        reward_modified=False,base_actor_GRU_critic_laser_frozen=True,
        control_center_root=str(ROOT/'artifacts/engineering/training-control-center-v1'),
        panel_job_id='20261025-B1-full-geometry-risk-calibration-v1')
    write(control/'plan.json',plan)
    temporary='./outputs/tmp'
    spec=dict(schema='fly-touhou-background-job-v1',python=str(ROOT/'.venv/Scripts/python.exe'),
        script=str(runtime/'scripts/train_b1_full_geometry_risk.py'),arguments=['--plan',str(control/'plan.json')],
        cwd=str(runtime),control_directory=str(control),environment=dict(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TEMP=temporary,TMP=temporary,TMPDIR=temporary))
    write(control/'background_spec.json',spec)
    write(r/'full-geometry-calibration-preparation-v1.json',dict(utc=utc_now(),source_runtime=str(runtime),
        source_manifest_sha256=plan['source_manifest_sha256'],plan=str(control/'plan.json'),
        plan_sha256=digest(control/'plan.json'),files=len(files),held_native_seeds=held,
        native_worlds_started=0,optimizer_updates=0,process_launched=False))
    print(json.dumps(dict(runtime=str(runtime),manifest_sha256=plan['source_manifest_sha256'],
        files=len(files),held_native_seeds=held,spec=str(control/'background_spec.json')),ensure_ascii=False))


if __name__=='__main__':main()
