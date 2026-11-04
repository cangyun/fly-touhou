"""Record a qualified followup and actual display curves;start no workloads."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def main():
    import psutil
    records=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    control=ROOT/'artifacts/engineering/tensorboard-live-v3'
    restart=read(control/'actor-display-restart-20261025-v1.json')
    status=read(control/'status.json');identity=restart['new_identity']
    process=psutil.Process(identity['pid'])
    if abs(process.create_time()-identity['create_time'])>.01 or status['pid']!=process.pid or status['last_read_error']:
        raise ValueError('Read-only display is not healthy')
    curves={}
    for tag in ('teacher_data/recorded_samples','teacher_data/committed_samples','teacher_data/free_storage_GiB',
                'hardware_live/system_gpu_utilization_percent'):
        url='http://127.0.0.1:6006/data/plugin/scalars/scalars?'+urllib.parse.urlencode(
            dict(run='teacher-full-natural30-v1',tag=tag))
        with urllib.request.urlopen(url,timeout=5) as response:rows=json.load(response)
        if not rows:raise ValueError('Actual teacher curve missing: '+tag)
        curves[tag]=rows[-1]
    runtime=Path('./outputs/source_snapshots/20261025-B1-full-teacher-to-full-PPO-v1')
    manifest=runtime/'source_manifest.json';expected='3b8c9d6a506e9cc2cae282a095f627882c9beedbb0b0f379e22bdf5d9ce887c0'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:raise ValueError('Followup runtime differs')
    tests=records/'full-teacher-to-PPO-frozen-tests-v1.xml'
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=37 or any(
            int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):
        raise ValueError('37 frozen checks required')
    proof=dict(schema='B1-full-teacher-to-PPO-preparation-v1',utc=utc_now(),runtime=str(runtime),manifest_sha256=expected,
        files=489,unchanged_baseline_files=484,existing_native_PPO_model_reward_files_replaced=0,
        frozen_tests=dict(passed=37,path=str(tests),sha256=hashlib.sha256(tests.read_bytes()).hexdigest()),
        full_shared_base_trainable_after_BC=True,fitted_point_and_laser_risk_still_frozen=True,
        PPO_Adam_explicitly_fresh=True,restricted_BC_optimizer_not_reinterpreted=True,
        BC_cost_and_selected_steps_kept_distinct=True,ancestral_PPO_cost_preserved=True,
        real_teacher_fit_not_yet_started=True,following_PPO_not_yet_started=True,
        native_worlds_started=0,main_neural_optimizer_updates=0,
        numerical_smoke_and_synthetic_fixtures_forbidden_in_transfer=True,
        training_budget_not_yet_declared=True,original_goal_achieved=False)
    write(records/'full-teacher-to-PPO-preparation-v1.json',proof)
    restart.update(actual_HTTP_curves=curves,HTTP_verified_utc=utc_now(),BC_curves_prepared_but_no_real_fit_yet=True)
    write(control/'actor-display-restart-20261025-v1.json',restart)
    state=read(records/'goal_state.json');state['utc']=state['updated_utc']=utc_now()
    state['prepared_full_teacher_to_PPO']=proof
    state['tensorboard_live'].update(identity=identity,revision='independent-native-teacher-and-CUDA-BC-v6',
        tests_passed=12,status=status['status'],real_teacher_curves_verified_via_HTTP=True,
        BC_epoch_live_selection_curves_prepared=True,BC_results_not_native_efficacy=True)
    native=state['current_native_phase'];folder=Path(native['output']);live=read(folder/'progress.json')
    collector=read(folder/'launch.json');p=psutil.Process(collector['pid'])
    if abs(p.create_time()-collector['create_time'])>.01:raise ValueError('Collector identity drifted')
    for key in ('acknowledged_native_actions','recorded_samples','committed_samples','complete_native_episodes'):
        native[key]=live[key]
    native.update(current_stage_counts=dict(Counter(r['stage'] for r in live['current_environments'])),
                  progress_observed_utc=utc_now())
    state['latest_goal_turn_classification']=dict(classification='progress',utc=utc_now(),
        evidence='30 natural teacher worlds remain healthy;independent BC TensorBoard prepared;37 frozen full-policy transfer checks passed',
        main_neural_optimizer_updates=0,actual_collector_actions=live['acknowledged_native_actions'],
        goal_achieved=False,blocking_condition=None)
    write(records/'goal_state.json',state)
    print(json.dumps(dict(utc=utc_now(),TensorBoard_PID=process.pid,actual_HTTP_curves=curves,
        native_collector_PID=p.pid,teacher_actions=live['acknowledged_native_actions'],
        stage_counts=native['current_stage_counts'],full_policy_frozen_tests=37,neural_fit_not_yet_started=True)))


if __name__=='__main__':main()
