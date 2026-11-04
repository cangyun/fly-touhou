"""Record actual native prefix commands for later frozen stage-six branches.

Recording only: nothing is replayed, reset, advanced, or used to choose actions.
"""
from pathlib import Path
from .thrl_durable_records import write_json


def start(env):
    if not getattr(env,'stage6_prefix_capture',None):return
    Path(env.stage6_prefix_capture).parent.mkdir(parents=True,exist_ok=True)
    from .native_replay import semantic_state
    env._frozen_prefix=dict(start=semantic_state(env.snapshot),raw_input_segments=[],
        transitions=[],ticks=0,native_seed=env.native_seed,status='recording')
    env.controller.begin_input_tape()


def before_step(env):
    record=getattr(env,'_frozen_prefix',None)
    if not record or record['status']!='recording':return None
    s=env.snapshot
    return dict(segment=len(env.controller._input_tape),ticks=record['ticks'],
        stage=s.stage,stage_frame=s.stage_frame,frame=s.frame,
        nonplay_ticks=env._nonplay_ticks,rng=dict(s.extra['native_rng_state']),
        resources=dict(lives=s.lives,bombs=s.bombs,power=s.power,score=s.score))


def after_step(env,before,info,done):
    if getattr(env,'frozen_stage6_metrics',None):
        from .compact_frozen_stage6_metrics import observe
        observe(env,info,done)
    if before is None:return
    record=env._frozen_prefix
    commands=env.controller._input_tape[before['segment']:]
    record['ticks']+=sum(t for _,t in commands)
    s=env.snapshot
    bridge_ticks=env._nonplay_ticks-before.get('nonplay_ticks',env._nonplay_ticks)
    if s.stage!=before['stage'] or bridge_ticks:
        from .native_replay import semantic_state
        record['transitions'].append(dict(before=before,after=dict(stage=s.stage,
            stage_frame=s.stage_frame,frame=s.frame,ticks=record['ticks']),
            initial_tape=commands[:-bridge_ticks] if bridge_ticks else commands,
            loading_bridge_ticks=bridge_ticks,state=semantic_state(s)))
    if s.stage==5 and not record.get('stage5_capture'):
        from .native_replay import semantic_state
        target=Path(env.stage6_prefix_capture).parent.parent/'stage5-prefix'/Path(env.stage6_prefix_capture).name
        target.parent.mkdir(parents=True,exist_ok=True)
        write_json(target,dict(record,schema='frozen-SNN-actual-prefix-commands-v1',status='stage5_entry',
            raw_input_segments=list(env.controller._input_tape),landing=semantic_state(s),
            native_events={k:s.extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')},
            teacher_actions=0,optimizer_updates=0,prefix_was_played_live_by_frozen_network=True))
        record['stage5_capture']=str(target)
    if s.stage!=6 and not done:return
    from .native_replay import semantic_state
    record.update(schema='frozen-SNN-actual-prefix-commands-v1',
        raw_input_segments=env.controller.end_input_tape(),landing=semantic_state(s),
        native_events={k:s.extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')},
        status='stage6_entry' if s.stage==6 else 'terminated_before_stage6',
        teacher_actions=0,optimizer_updates=0,prefix_was_played_live_by_frozen_network=True)
    write_json(Path(env.stage6_prefix_capture),record)
    info['stage6_prefix_capture']=env.stage6_prefix_capture
