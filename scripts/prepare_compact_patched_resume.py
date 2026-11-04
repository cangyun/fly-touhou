"""Prepare, but never launch, the next compact-SNN continuation.

The source is the latest completed whole-cycle model and Adam/RNG checkpoint.
The existing frozen evaluation and the user's paused scheduling stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def replace_arg(argv,name,value):
    if name in argv:
        index=argv.index(name)
        argv[index+1]=str(value)
    else:
        argv.extend([name,str(value)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,default=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/compact-full-story-20261030')
    parser.add_argument('--cycles',type=int,default=7)
    args=parser.parse_args()
    control=args.control.resolve();work=control/'v10-native-boss-interface-repair-20261031'
    current_path=control/'current-job.json'
    job=read_shared_json(current_path);source=Path(job['active_output'])
    progress=read_shared_json(source/'progress.json')
    cycle=int(progress['completed_cycle'])
    if cycle<29:raise ValueError('The resumed source must include the completed cycle 29')
    parameters=source/f'cycle{cycle:03}-parameters.pt'
    receipt=source/f'cycle{cycle:03}-training-path.json'
    if not parameters.exists() or not receipt.exists():
        raise FileNotFoundError('The latest completed cycle has no complete model and Adam checkpoint')
    training=Path(read_shared_json(receipt)['path'])
    if not training.exists():raise FileNotFoundError('The cycle training checkpoint is missing')
    if args.cycles<1:raise ValueError('A positive cycle count is required')

    candidate=read_shared_json(work/'environment-config-curve-candidate.json')
    native_profile=read_shared_json(candidate['profile'])
    profile_path=work/'th14_1.00b_spiking_sensor_patched.json'
    write_json(profile_path,native_profile)
    candidate.update(profile=str(profile_path),compact_native_focus_gate=True,
        compact_native_curve_laser_geometry=True,compact_unknown_laser_player_motion=True,
        compact_native_motion_forecast=True,compact_conditional_wall_guidance_coefficient=.01)
    config_path=work/'environment-config-spiking-sensor-patched.json'
    write_json(config_path,candidate)

    dispatch=read_shared_json(work/'training-dispatch.json')
    argv=list(dispatch['argv'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=source.parent/f'compact-SNN-sensor-patched-from{cycle:03}-{stamp}'
    updates={
        '--checkpoint':parameters,'--resume-training':training,'--environment-config':config_path,
        '--output':destination,'--cycle-offset':cycle,'--cycles':args.cycles,
        '--initial-decision-budget':args.cycles*int(argv[argv.index('--workers')+1])*int(argv[argv.index('--steps')+1]),
        '--minimum-policy-lr-scale':.005,'--boss-attack-guidance-coefficient':.02,
    }
    for name,value in updates.items():replace_arg(argv,name,value)
    for switch in ('--native-focus-gate','--native-curve-laser-geometry'):
        if switch not in argv:argv.append(switch)
    plan_path=work/'resume-spiking-sensor-patched.json'
    plan=dict(status='prepared_not_started',source_completed_cycle=cycle,
        source_parameters=str(parameters),source_training=str(training),environment_config=str(config_path),
        argv=argv,output=str(destination),native_focus_gate=True,native_curve_laser_geometry=True,
        unknown_laser_player_motion=True,brain_topology_changed=False,
        minimum_policy_lr_scale=.005,boss_attack_guidance_coefficient=.02,
        reason_for_lr='PPO mean KL0.228 at cycle29 against target0.03; inherited scale stuck at0.05',
        reason_for_attack_guidance='Existing teacher is substantially more likely to close on Boss in measured safe windows',
        action='Only prepare the next continuation; no learner or original game is started')
    write_json(plan_path,plan)
    job.update(next_training_source_parameters=str(parameters),next_training_source_training=str(training),
        next_training_source_completed_cycle=cycle,next_training_source_is_provisional=False,
        next_environment_repair=str(config_path),next_training_preparation=str(plan_path),
        sensor_patches_prepared=True)
    write_json(current_path,job)
    print(f'Prepared cycle {cycle} continuation: {plan_path}; no training was started.')


if __name__=='__main__':main()
