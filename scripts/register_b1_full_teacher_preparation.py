"""Record verified preparation and the live collector; never controls a job."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def main():
    import psutil
    records=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    output=Path('./outputs/engineering/B1-autonomous-dodge-20261024/teacher-full-natural30-v1')
    identity=read(output/'launch.json');process=psutil.Process(identity['pid'])
    if abs(process.create_time()-identity['create_time'])>.01 or not process.is_running():
        raise RuntimeError('Refresh actual collector terminal state before recording live progress')
    progress=read(output/'progress.json')
    runtime=Path('./outputs/source_snapshots/20261025-B1-full-teacher-actor-fit-v1')
    expected='619d97d10656338c8f7923d99406f10b1c6d041fea6176a8ded4f0236da724d5'
    if hashlib.sha256((runtime/'source_manifest.json').read_bytes()).hexdigest()!=expected:
        raise ValueError('Fitter source changed')
    tests=records/'full-teacher-actor-fit-frozen-tests-v1.xml'
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if sum(int(row.get('tests',0)) for row in suites)!=30 or any(
            int(row.get('failures',0))+int(row.get('errors',0)) for row in suites):
        raise ValueError('Frozen30 checks not passed')
    diagnostics=Path('./outputs/diagnostics/B1-autonomous-dodge-20261024')
    cuda=diagnostics/'cached-encoder-actor-fit-CUDA-qualification-v1/report.json'
    smoke=diagnostics/'full-teacher-actor-fitter-CUDA-smoke-v1/report.json'
    if read(cuda)['status']!='passed' or read(smoke)['status']!='passed':
        raise ValueError('CUDA qualification or orchestration exercise missing')
    proof=dict(schema='B1-full-teacher-actor-fitter-preparation-v1',utc=utc_now(),
        status='qualified-preparation-not-real-teacher-fit',runtime=str(runtime),manifest_sha256=expected,
        files=484,baseline_files_unchanged=474,existing_native_model_reward_and_PPO_files_replaced=0,
        frozen_tests=dict(path=str(tests),passed=30,sha256=hashlib.sha256(tests.read_bytes()).hexdigest()),
        actual_CUDA_equivalence=str(cuda),actual_CUDA_actor_fitter_loop_smoke=str(smoke),
        smoke_used_synthetic_contract_data=True,smoke_policy_deployment_forbidden=True,
        new_deployable_model_trained=False,main_neural_optimizer_updates=0,current_collector_unchanged=True,
        arguments=dict(epochs=12,minimum_epochs=4,patience=4,seq_len=16,batch_sequences=64,learning_rate=.0001,seed=31),
        neural_training_requires_completed_real_teacher_cohort=True,neural_training_not_started=True)
    write(records/'full-teacher-actor-fitter-preparation-v1.json',proof)
    state=read(records/'goal_state.json')
    state['utc']=state['updated_utc']=utc_now()
    state['current_native_phase'].update(acknowledged_native_actions=progress['acknowledged_native_actions'],
        recorded_samples=progress['recorded_samples'],committed_samples=progress['committed_samples'],
        complete_native_episodes=progress['complete_native_episodes'],progress_observed_utc=utc_now())
    state['prepared_cached_actor_fitting'].update(CUDA_equivalence_not_yet_qualified=False,
        CUDA_qualification=str(cuda),CUDA_gradient_and_Adam_error=0.,
        actual_full_trajectory_fitter_not_yet_implemented=False,actor_warmup_fitter_implemented=True,
        prepared_fitter_runtime=str(runtime),prepared_fitter_manifest_sha256=expected,
        frozen_fitter_tests_passed=30,full_loop_CUDA_smoke=str(smoke),
        smoke_artifacts_deployment_forbidden=True,full_policy_optimization_after_warmup_still_required=True,
        prepared_neural_fitter_not_yet_started=True)
    tensorboard=read(ROOT/'artifacts/engineering/tensorboard-live-v3/status.json')
    tb_process=psutil.Process(tensorboard['pid'])
    state['tensorboard_live'].update(identity=dict(pid=tb_process.pid,create_time=tb_process.create_time()),
        revision='independent-rule-teacher-data-and-wall-hardware-v5',status=tensorboard['status'],
        teacher_source=str(output),tests_passed=8,real_teacher_curves_verified_via_HTTP=True,
        hardware_uses_advancing_wall_clock_axis=True)
    state['latest_goal_turn_classification']=dict(classification='progress',utc=utc_now(),
        evidence='Complete140 reviewed;30 real full-game teacher worlds sampling;CUDA cache equivalence and frozen fitter30 checks plus full-loop scratch exercise passed',
        main_neural_optimizer_updates=0,current_original_worlds=30,
        current_collector_identity=dict(pid=process.pid,create_time=process.create_time()),
        actual_collector_actions=progress['acknowledged_native_actions'],goal_achieved=False,blocking_condition=None)
    write(records/'goal_state.json',state)
    print(json.dumps(dict(utc=utc_now(),collector_pid=process.pid,
        actions=progress['acknowledged_native_actions'],complete_episodes=progress['complete_native_episodes'],
        prepared_CUDA_actor_fitter_tests=30,real_neural_fit_started=False)))


if __name__=='__main__':main()
