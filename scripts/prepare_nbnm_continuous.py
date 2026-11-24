"""Prepare a shared-policy forward-chain trial; never launch training."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def prepare(base, *, checkpoint, optimizer, rehearsal, story_bank, output, run_output, tensorboard):
    spec=copy.deepcopy(base)
    hard=spec['tasks']['s4_hard'];retain=spec['tasks']['s3_hard']
    bridge=dict(mode='chain',stage=3,attack='BossCard1',end_stage=3,end_attack='BossCard2')
    spec.update(initial_checkpoint=str(checkpoint),warmstart_optimizer_checkpoint=str(optimizer),
        workers=16,segment_decisions=8192,decision_budget=32768,
        tasks=dict(s4_hard=hard,s3_hard=retain,bridge=bridge),
        world_groups=['s4_hard']*8+['s3_hard']*4+['bridge']*4,
        group_weights=dict(s4_hard=.5,s3_hard=.25,bridge=.25),evaluation_workers=4,
        evaluation_tasks=['s4_hard','s3_hard','bridge'],
        task_entries=dict(bridge=dict(entry_mode='story_prefix',start_bank=str(story_bank))),
        output=str(run_output),tensorboard_directory=str(tensorboard),
        rehearsal=dict(manifest=str(rehearsal),coefficient=.03,every_updates=8),
        evaluate_at_end=True,cuda_graph_inference=True,status_port=0,
        experiment_notes=dict(single_shared_policy=True,brain_topology_changed=False,
            teacher='NBNM no-Bomb movement guidance, all18 candidates; native4+20 and4+8+12 paths',
            curriculum='Forward continuous boundaries; no expert switching or reverse resets',
            preparation_only=True))
    spec['nbnm'].update(task=hard,teacher_mode='nbnm-route-v1',rule_guidance_coefficient=.1,
        task_context='critic-only-v1',history_cuda_graph=True,
        objective=dict(phase='completion',speed_weight=.01,time_scale_frames=3600))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from fly_touhou.nbnm_run import validate_spec
    validation=validate_spec(spec)
    for name,value in [('route.json',spec),('validation.json',validation)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    # These are manual next-phase configurations, not automatic resets or
    # permission to train. They keep the SAME network architecture/checkpoint.
    next_stages=[dict(name='stage3-boss-tail',task=dict(mode='chain',stage=3,attack='BossCard1',end_stage=3),
                     group_weights=dict(s4_hard=.25,s3_hard=.25,bridge=.5)),
                 dict(name='stages3-through4',task=dict(mode='chain',stage=3,attack='BossCard1',end_stage=4),
                     group_weights=dict(s4_hard=.25,s3_hard=.25,bridge=.5)),
                 dict(name='opening-through-stage4',task=dict(mode='chain',stage=1,end_stage=4),
                     group_weights=dict(s4_hard=.25,s3_hard=.125,bridge=.625)),
                 dict(name='opening-through-clear',task=dict(mode='full_story'),
                     group_weights=dict(s4_hard=.125,s3_hard=.125,bridge=.75))]
    (output/'forward-course.json').write_text(json.dumps(dict(stages=next_stages,
        final='full_story from untouched opening, no memory/position/resource resets',
        local_success_is_full_clear=False,automatic_launch=False,
        promotion='Compare task-specific development success, chain outcomes and wall time; do not average easy cards over regressions'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('base-config','checkpoint','optimizer','rehearsal','story-bank','output','run-output','tensorboard'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    prepare(json.loads(a.base_config.read_text(encoding='utf-8-sig')),checkpoint=a.checkpoint,optimizer=a.optimizer,
        rehearsal=a.rehearsal,story_bank=a.story_bank,output=a.output,run_output=a.run_output,tensorboard=a.tensorboard)
    print(a.output/'route.json')


if __name__=='__main__':main()
