"""Prepare one immutable CUDA warmup launch;never launch or alter collection."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    records=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    preparation=read(records/'full-teacher-actor-fitter-preparation-v1.json')
    runtime=Path(preparation['runtime']);manifest=runtime/'source_manifest.json'
    if sha(manifest)!=preparation['manifest_sha256']:raise ValueError('Qualified actor fitter changed')
    for row in read(manifest)['files']:
        if sha(runtime/row['path'])!=row['sha256']:raise ValueError('Frozen actor source differs: '+row['path'])
    suites=list(ET.parse(preparation['frozen_tests']['path']).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=30 or any(
            int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):
        raise ValueError('30 frozen fitter checks required')
    state=read(records/'goal_state.json');parent=state['next_policy_initializer']
    qualification=Path(preparation['actual_CUDA_equivalence']);q=read(qualification)
    if (q['status']!='passed' or q['device']!='cuda' or q['checkpoint_sha256']!=parent['sha256']
            or sha(parent['path'])!=parent['sha256']
            or q['source_hashes']['thrl_cached_actor_fit.py']!=sha(runtime/'src/fly_touhou/thrl_cached_actor_fit.py')):
        raise ValueError('Actual CUDA equivalence or chosen parent identity differs')
    base=Path('./outputs/engineering/B1-autonomous-dodge-20261024')
    data=base/'teacher-full-natural30-v1';output=base/'teacher-full-actor-warmup-v1'
    control=records/'teacher-full-actor-warmup-v1-control'
    if control.exists() or output.exists():raise FileExistsError('Prepared or launched actor job already exists')
    args=['--parent-checkpoint',parent['path'],'--parent-sha256',parent['sha256'],
        '--cuda-qualification',str(qualification),'--demonstrations',str(data),'--output',str(output),
        '--epochs','12','--minimum-epochs','4','--patience','4','--seq-len','16',
        '--batch-sequences','64','--learning-rate','0.0001','--seed','31']
    spec=dict(schema='fly-touhou-background-job-v1',python=str(ROOT/'.venv/Scripts/python.exe'),
        script=str(runtime/'scripts/train_b1_full_teacher_actor.py'),arguments=args,cwd=str(runtime),
        control_directory=str(control),environment=dict(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONPATH=str(runtime/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TEMP='./outputs/tmp',TMP='./outputs/tmp',TMPDIR='./outputs/tmp'))
    control.mkdir(parents=True);write(control/'background_spec.json',spec)
    readiness=dict(schema='B1-full-natural-teacher-CUDA-actor-warmup-launch-prepared-v1',utc=utc_now(),
        spec=str(control/'background_spec.json'),spec_sha256=sha(control/'background_spec.json'),
        source_runtime=str(runtime),source_manifest_sha256=preparation['manifest_sha256'],
        frozen_tests_passed=30,actual_CUDA_qualification=str(qualification),native_worlds_started=0,
        process_launched=False,neural_optimizer_steps=0,requires_complete_real_teacher_cohort=True,
        requires_collector_actual_process_exited=True,requires_global_original_world_count_zero=True,
        no_automatic_restart=True,actor_output=str(output),expected_TensorBoard_run=output.name)
    write(control/'preparation.json',readiness)
    state['prepared_full_teacher_actor_job']=readiness;state['utc']=state['updated_utc']=utc_now()
    state['latest_goal_turn_classification']=dict(classification='progress',utc=utc_now(),
        evidence='Pinned CUDA full-trajectory actor warmup launch prepared;no overlap with live teacher collection',
        main_neural_optimizer_updates=0,goal_achieved=False,blocking_condition=None)
    write(records/'goal_state.json',state)
    print(json.dumps(readiness))


if __name__=='__main__':main()
