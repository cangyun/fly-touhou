"""Queue the already-prepared sensor-patched continuation after current evaluation.

This writes workspace control files only; starting the controller is a separate
step. It leaves the active native evaluation and its stop watcher untouched.
"""
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def main():
    control=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/compact-full-story-20261030'
    old=control/'v10-native-boss-interface-repair-20261031'
    next_control=control/'v11-sensor-patched-continuation-20261031'
    next_control.mkdir(parents=True,exist_ok=True)
    current_path=control/'current-job.json';job=read_shared_json(current_path)
    previous=Path(job['active_output']);progress=read_shared_json(previous/'progress.json')
    cycle=int(progress['completed_cycle'])
    if cycle!=29:raise ValueError('The prepared continuation expects completed cycle29')
    prepared=read_shared_json(old/'resume-spiking-sensor-patched.json')
    # Keep the same original-game worlds alive long enough to encounter later
    # stages. Evaluate every seven cycles, and only extend beyond the first
    # fourteen when the frozen report improves on the actual prior incumbent.
    rows=8*1024
    initial_cycles,max_cycles=14,21
    argv=list(prepared['argv'])
    for name,value in (('--cycles',max_cycles),('--initial-decision-budget',initial_cycles*rows)):
        argv[argv.index(name)+1]=str(value)
    # Cycle 29 measured KL .228 with scale .05. This is the existing adaptive
    # rule's next scale with the newly lowered floor: .05*sqrt(.03/.228).
    if '--policy-lr-scale' in argv:
        argv[argv.index('--policy-lr-scale')+1]='0.01814'
    else:
        argv.extend(['--policy-lr-scale','0.01814'])
    prepared.update(argv=argv,initial_new_decisions=initial_cycles*rows,
        maximum_new_decisions=max_cycles*rows,evaluation_every_cycles=7,
        coverage='Same persistent student-controlled full-story worlds across evaluations',
        initial_policy_lr_scale=.01814)
    write_json(old/'resume-spiking-sensor-patched.json',prepared)
    old_status=read_shared_json(old/'status.json')
    source_costs=dict(job['source_costs'])
    source_costs.update(PPO_updates=int(progress['PPO_updates']),new_student_actions=int(progress['new_student_actions']))
    plan=dict(current_job=str(current_path),previous_output=str(previous),
        previous_executor=job['actual_executor'],previous_stop_cycle=cycle,
        previous_controller=dict(PID=old_status['PID'],create_time=old_status['create_time']),
        previous_evaluation_follows_cycle=True,use_operator_stop_checkpoint=True,
        stop_for_qualifying_frozen_clears=3,incumbent_from_current=True,
        previous_history_status='stopped_after_cycle29_frozen_development_for_sensor_repairs',
        previous_completed_frozen_report=str(previous/'development-cycle029/report.json'),
        output=prepared['output'],argv=argv,workers=8,steps=1024,updates_per_cycle=32,
        initial_new_decisions=initial_cycles*rows,maximum_new_decisions=max_cycles*rows,evaluation_every_cycles=7,
        source_costs=source_costs,paid_total_before_previous=job['paid_total_costs_before_job'],
        incumbent_checkpoint=job['best_verified_parameters'],incumbent_report=job['best_verified_native_report'],
        source_binding_proof=str(old/'resume-spiking-sensor-patched.json'),
        tensorboard_configuration=str(ROOT/'artifacts/engineering/tensorboard-live-v3/configuration.json'),
        tensorboard_series='compact_SNN_sensor_patched_from_cycle29',native_motion_forecast=True,
        active_stage='sensor_patched_from_cycle29_full_story_PPO',
        conditional_wall_guidance_coefficient=.01)
    plan_path=next_control/'plan.json';write_json(plan_path,plan)
    job.update(scheduling_suspended=False,pause_after_current_test=False,
        goal_resumed_utc=datetime.now(timezone.utc).isoformat(),
        next_training_preparation=str(plan_path),
        next_goal_action='Finish cycle29 frozen development, then continue the patched model from its saved optimizer')
    write_json(current_path,job)
    print(f'Queued patched continuation after the current evaluation: {plan_path}')


if __name__=='__main__':main()
