"""Prepare a future PPO continuation with late-stage rule-teacher replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def set_value(argv,name,*values):
    if name in argv:
        position=argv.index(name)
        if len(values)!=1:raise ValueError('Only single-value options can be replaced')
        argv[position+1]=str(values[0])
    else:
        argv.extend([name,*map(str,values)])


def remove_single(argv,name):
    if name in argv:
        position=argv.index(name)
        del argv[position:position+2]


def remove_repeated(argv,name,values=1):
    while name in argv:
        position=argv.index(name)
        del argv[position:position+1+values]


def replace_multi(argv,name,values):
    """Replace a variable-length argument list without disturbing later options."""
    if name in argv:
        start=argv.index(name);end=start+1
        while end<len(argv) and not argv[end].startswith('--'):end+=1
        del argv[start:end]
    argv.extend([name,*map(str,values)])


def teacher_sampling(teacher):
    stages={int(row['stage']) for row in teacher['windows'] if row['split']=='train'}
    if stages=={3,4,5,6}:return {3:.35,4:.35,5:.15,6:.15}
    if stages=={2,3,4,5,6}:return {2:.10,3:.30,4:.30,5:.15,6:.15}
    if stages=={1,2,3,4,5,6}:return {1:.05,2:.10,3:.30,4:.30,5:.125,6:.125}
    if not ({3,4}<=stages<=set(range(1,7))):
        raise ValueError('The stage3/4 repair needs examples of both stages3 and4')
    # A real stage3/4 bank must not be rejected just because no teacher game
    # has reached stage5/6 yet. Keep the same relative priorities on the
    # stages actually collected; the student still plays full native games.
    base={1:.05,2:.10,3:.30,4:.30,5:.125,6:.125}
    total=sum(base[s] for s in stages)
    return {s:base[s]/total for s in sorted(stages)}


def saved_context_multiplier(checkpoint):
    groups=[group for group in checkpoint['optimizer']['param_groups'] if group.get('name')=='new_context_ports']
    if len(groups)!=1:raise ValueError('Expected one saved context-port optimizer group')
    return float(groups[0].get('policy_group_multiplier',1.))


def add_student_replay(argv,bank,coefficient):
    """Rehearse student-visited late stages as supervision, never PPO."""
    if not 0<coefficient<=1:
        raise ValueError('Student late-stage replay coefficient must be in (0,1]')
    payload=torch.load(bank,map_location='cpu',weights_only=False)
    if payload.get('schema')!='compact-student-late-stage-supervision-bank-v1':
        raise ValueError('Expected a student-controlled late-stage replay bank')
    counts={stage:len(payload['windows'].get(stage,[])) for stage in (3,4)}
    if not sum(counts.values()):raise ValueError('Student late-stage bank has no windows yet')
    token=str(bank)
    if not any(argv[i]==token and i>0 and argv[i-1]=='--late-stage-replay-bank'
               for i in range(len(argv))):
        argv.extend(['--late-stage-replay-bank',token])
    set_value(argv,'--late-stage-replay-coefficient',coefficient)
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','checkpoint','parameters','teacher-manifest','output-run','plan'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--environment-config',type=Path,
        help='Optional evaluated sensor profile;otherwise keep the source job configuration')
    p.add_argument('--budget-benefit-reference',type=Path,
        help='Optional completed frozen comparison;otherwise preserve the source reference')
    p.add_argument('--context-policy-lr-multiplier',type=float,
        help='Optional context-port LR override;otherwise preserve the saved optimizer value')
    p.add_argument('--policy-group-lr-multipliers',type=json.loads,
        help='Optional named policy-group overrides;otherwise inherit the actual saved optimizer rates')
    p.add_argument('--native-rule-teacher-windows',type=int,
        help='Pack1..7 teacher windows per PPO update;otherwise inherit the saved setting or use1')
    p.add_argument('--minimum-policy-lr-scale',type=float,default=.00005,
        help='Allow KL feedback below the old saturated0.005 floor;the current saved scale is preserved')
    p.add_argument('--student-late-bank',type=Path,
        help='Optional student-controlled stage3/4+ supervision bank with matching sensor semantics')
    p.add_argument('--student-late-coefficient',type=float,default=.05)
    p.add_argument('--allow-owned-game-controller',nargs=2,action='append',default=[],
        metavar=('PID','CREATED'))
    p.add_argument('--cycles',type=int,default=128,
        help='Hard upper bound, not the default length of a short reset-and-resume job')
    p.add_argument('--games-per-world',type=int,default=3,
        help='Complete natural full games in the same persistent world pool;no stage-start curriculum')
    p.add_argument('--wall-hours',type=float,default=72.,
        help='Explicit finite wall budget for the persistent game round;no implicit8h early truncation')
    p.add_argument('--evaluation-every',type=int,
        help='Run a frozen evaluation at this cycle interval while retaining the student game states')
    p.add_argument('--workers',type=int,help='Override persistent native world count')
    p.add_argument('--steps',type=int,help='Average decisions/world per segment in ready-driven mode')
    p.add_argument('--ready-driven',action='store_true')
    p.add_argument('--ready-per-world-quota',action='store_true')
    p.add_argument('--ppo-sample-decisions',type=int)
    p.add_argument('--keep-rollout-tapes',type=int)
    p.add_argument('--ppo-stage-weights',type=float,nargs=6)
    p.add_argument('--student-decision-budget',type=int)
    p.add_argument('--max-adam-updates',type=int)
    p.add_argument('--evaluation-every-decisions',type=int)
    p.add_argument('--first-evaluation-decisions',type=int)
    p.add_argument('--parameter-checkpoint-every-decisions',type=int)
    a=p.parse_args()
    source=read_shared_json(a.source_run/'launch.json')
    # A live collection control supersedes the original launch quota. In
    # particular,do not silently reinstate the withdrawn8-update cap.
    collection_path=a.source_run/'collection-control.json'
    if a.max_adam_updates is None and collection_path.exists():
        a.max_adam_updates=read_shared_json(collection_path).get('max_updates')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cycle=int(checkpoint['completed_cycle'])
    costs=checkpoint['model_payload']['costs']
    if not a.parameters.is_file() or a.cycles<1:raise ValueError('A whole-cycle model and positive cycle count are required')
    evaluation_every=min(16,a.cycles) if a.evaluation_every is None else a.evaluation_every
    if not 1<=evaluation_every<=a.cycles:raise ValueError('Evaluation interval must be between1 and the cycle count')
    teacher=read_shared_json(a.teacher_manifest)
    sampling=teacher_sampling(teacher)
    context_multiplier=(saved_context_multiplier(checkpoint) if a.context_policy_lr_multiplier is None
                        else a.context_policy_lr_multiplier)
    teacher_windows=(checkpoint.get('native_rule_teacher_windows',1) if a.native_rule_teacher_windows is None
                     else a.native_rule_teacher_windows)
    if not 1<=teacher_windows<=7:raise ValueError('One to seven native teacher windows required')
    if a.policy_group_lr_multipliers is not None:
        if not isinstance(a.policy_group_lr_multipliers,dict):raise ValueError('Named policy-group multiplier mapping required')
        if 'new_context_ports' in a.policy_group_lr_multipliers:
            if (a.context_policy_lr_multiplier is not None
                    and a.context_policy_lr_multiplier!=a.policy_group_lr_multipliers['new_context_ports']):
                raise ValueError('Conflicting context-port multiplier settings')
            context_multiplier=a.policy_group_lr_multipliers['new_context_ports']
    argv=list(source['argv'])
    # Runtime exploration changes are saved with the optimizer checkpoint.
    # Do not reintroduce an obsolete temperature from the original command.
    for name in ('movement_sampling_temperature','bomb_sampling_temperature'):
        if name in checkpoint:set_value(argv,'--'+name.replace('_','-'),checkpoint[name])
    # Previous jobs' late-stage labels used the old native geometry. The new
    # job rebuilds student replay as it plays, or uses the explicit new bank.
    remove_repeated(argv,'--late-stage-replay-source')
    remove_repeated(argv,'--late-stage-replay-bank')
    remove_repeated(argv,'--allow-owned-game-controller',2)
    workers=int(argv[argv.index('--workers')+1]) if '--workers' in argv else 16
    steps=int(argv[argv.index('--steps')+1]) if '--steps' in argv else 512
    source_segment_decisions=workers*steps
    if a.workers is not None:workers=a.workers
    if a.steps is not None:steps=a.steps
    if workers<1 or steps<16 or steps%16 or (a.ready_driven and workers*steps%256):
        raise ValueError('Positive worlds and complete16-step/256-row collection blocks required')
    if a.ready_driven and '--ready-driven' not in argv:argv.append('--ready-driven')
    if a.ready_per_world_quota and '--ready-per-world-quota' not in argv:argv.append('--ready-per-world-quota')
    if a.ppo_stage_weights is not None:replace_multi(argv,'--ppo-stage-weights',a.ppo_stage_weights)
    if a.ready_driven and '--lr-growth-reference-decisions' not in argv:
        set_value(argv,'--lr-growth-reference-decisions',source_segment_decisions)
    for key,value in {
        '--checkpoint':a.parameters,'--resume-training':a.checkpoint,
        '--output':a.output_run,'--cycle-offset':cycle,'--cycles':a.cycles,
        '--evaluation-every':evaluation_every,
        '--workers':workers,'--steps':steps,
        '--games-per-world':a.games_per_world,
        '--wall-hours':a.wall_hours,
        '--initial-decision-budget':a.student_decision_budget or a.cycles*workers*steps,
        '--native-rule-teacher-manifest':a.teacher_manifest,
        '--native-rule-teacher-coefficient':.125,'--native-rule-teacher-every':1,
        '--native-rule-teacher-windows':teacher_windows,
        '--minimum-policy-lr-scale':a.minimum_policy_lr_scale,
        '--context-policy-lr-multiplier':context_multiplier,
    }.items():set_value(argv,key,value)
    for name in ('student_decision_budget','max_adam_updates','evaluation_every_decisions',
                 'first_evaluation_decisions','parameter_checkpoint_every_decisions','ppo_sample_decisions','keep_rollout_tapes'):
        value=getattr(a,name)
        if value is not None:set_value(argv,'--'+name.replace('_','-'),value)
    if a.budget_benefit_reference is not None:
        set_value(argv,'--budget-benefit-reference',a.budget_benefit_reference)
    elif '--budget-benefit-reference' not in argv:
        reference=a.source_run/'external-budget-reference.json'
        if not reference.exists():raise ValueError('A completed frozen benefit reference is required')
        set_value(argv,'--budget-benefit-reference',read_shared_json(reference)['path'])
    # The saved effective environment includes live sensor repairs that the
    # original command line may predate. Preserve those repairs on resume.
    effective_environment=a.source_run/'environment-config.json'
    if a.environment_config is not None:set_value(argv,'--environment-config',a.environment_config)
    elif effective_environment.is_file():set_value(argv,'--environment-config',effective_environment)
    remove_single(argv,'--policy-lr-scale') # retain the saved adaptive scale and Adam moments
    # Runtime optimizer controls supersede historical CLI flags in the source
    # launch. A resumed run must not silently re-enable discarded KL control.
    effective_launch=read_shared_json(a.source_run/'launch.json')
    for name in ('adaptive_policy_lr','kl_early_stop'):
        yes='--'+name.replace('_','-');no='--no-'+name.replace('_','-')
        argv[:]=[item for item in argv if item not in (yes,no)]
        argv.append(yes if effective_launch.get(name,False) else no)
    if not effective_launch.get('adaptive_policy_lr',False) and not effective_launch.get('kl_early_stop',False):
        remove_single(argv,'--target-kl')
    remove_single(argv,'--neural-risk-scale') # inherit the saved model's current readout scale
    remove_single(argv,'--policy-group-lr-multipliers') # source CLI may predate later runtime changes
    if a.policy_group_lr_multipliers is not None:
        set_value(argv,'--policy-group-lr-multipliers',json.dumps(a.policy_group_lr_multipliers))
    replace_multi(argv,'--native-rule-teacher-stage-weights',sampling.values())
    student_windows=(add_student_replay(argv,a.student_late_bank,a.student_late_coefficient)
                     if a.student_late_bank is not None else None)
    for pid,created in a.allow_owned_game_controller:
        argv.extend(['--allow-owned-game-controller',pid,created])
    result=dict(status='prepared_not_started',source_cycle=cycle,source_costs=costs,
        source_training=str(a.checkpoint),source_parameters=str(a.parameters),
        argv=[sys.executable,'-B',*argv],
        policy_scale='inherited from saved optimizer',context_policy_lr_multiplier=context_multiplier,
        policy_group_lr_overrides=a.policy_group_lr_multipliers,
        minimum_policy_lr_scale=a.minimum_policy_lr_scale,
        policy_optimizer_precision='compact-policy-master-fp64-v1',
        teacher_sampling=sampling,teacher_coefficient=.125,teacher_every_PPO_updates=1,
        teacher_windows_per_PPO_update=teacher_windows,
        teacher_schedule='Weighted deficit quotas across packed windows;existing mean supervised loss at each PPO update',
        target_games_per_world=a.games_per_world,maximum_cycles=a.cycles,
        native_games_restored_from_checkpoint=False,
        minimum_late_stage_samples_not_guaranteed=True,
        evaluation_every_cycles=evaluation_every,
        collection=dict(workers=workers,segment_decisions=workers*steps,
            fixed_student_decision_budget=a.student_decision_budget,
            max_adam_updates=a.max_adam_updates,
            evaluation_every_decisions=a.evaluation_every_decisions,
            first_evaluation_decisions=a.first_evaluation_decisions,
            mode=('ready-per-world-long-batch-v1' if '--ready-per-world-quota' in argv else 'ready-global-blocks-v1') if '--ready-driven' in argv else 'synchronous-vector',
            PPO_sample_decisions=a.ppo_sample_decisions,
            PPO_stage_weights=a.ppo_stage_weights,
            native_games_persist_between_updates=True),
        student_late_bank=str(a.student_late_bank) if a.student_late_bank else None,
        student_late_windows=student_windows,student_late_windows_are_preparation_snapshot=True,
        student_late_coefficient=a.student_late_coefficient if a.student_late_bank else 0.,
        allowed_owned_game_controllers=a.allow_owned_game_controller,
        teacher_games_are_PPO_samples=False,brain_topology_changed=False,
        note='Provisional completed-checkpoint continuation. Refresh from the newest whole-cycle checkpoint before use; use a student bank with matching sensor semantics; no running game is stopped.')
    a.plan.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.plan,result)
    print(json.dumps(dict(plan=str(a.plan),source_cycle=cycle,source_costs=costs,teacher_PPO_samples=0),ensure_ascii=False))


if __name__=='__main__':main()
