"""Original-game course capture and exact independent replay verification."""
from pathlib import Path
import time
import uuid

from .atomic_json_io import atomic_write_json
from .native import sha256_file
from .native_replay import state_fingerprint,semantic_state,VISUAL_ALIGNMENT_SCHEMA,replay_entry
from .thrl_course_store import TRACE_SCHEMA,PhaseTracker,trajectory_entry,validate_trace
from .thrl_curriculum import seed_split


def capture_trajectory(executable,profile,output,seed,*,max_stage=1,max_decisions=10000,
                       headless=True,check_stop=lambda:None,on_progress=lambda row:None,
                       controller_observer=None):
    from .thrl_native_env import THRLNativeEnv
    from .thrl_teacher import make_teacher,native36_to_source19
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=THRLNativeEnv(executable,profile,private_appdata=output/'capture/userdata',
        headless=headless,binary_reader=True,native_seed=seed,policy_action_count=19)
    env.course_loading_bridge_frames=4
    if controller_observer is not None:
        factory=env._create_controller
        env._create_controller=lambda:controller_observer(factory())
    tracker=PhaseTracker();teacher=make_teacher();started=time.monotonic();records=[]
    trace=dict(schema=TRACE_SCHEMA,id=f'seed{seed}-{uuid.uuid4().hex[:12]}',native_seed=seed,
               split=seed_split(seed),game_exe_sha256=sha256_file(executable),base_stage_frame=64,
               verified_independent_resets=0,resources_injected=False,native_deaths=True,
               source='ResourceTeacherV6 preparation; no PPO, no resource injection',records=records,
               native_six_stage_clear=False)
    try:
        env.reset();snapshot=env.snapshot
        while snapshot.stage==1 and snapshot.stage_frame<64:
            check_stop();snapshot=env.controller.advance(1,1)
        if snapshot.stage!=1 or snapshot.stage_frame!=64:
            raise ValueError('Course capture missed the canonical frame64 base')
        env.snapshot=snapshot;env._encode(snapshot);env.reward_model.reset(snapshot)
        trace['base_state_sha256']=state_fingerprint(snapshot)
        trace['base_visual_rng_alignment']=dict(schema=VISUAL_ALIGNMENT_SCHEMA,reference_state=semantic_state(snapshot))
        env.controller.begin_input_tape()
        for i in range(max_decisions+1):
            check_stop();snapshot=env.snapshot
            phase=tracker.observe(snapshot)
            events=snapshot.extra['thrl_events'];roles=events['ecl_roles']
            row=dict(ticks=sum(t for _,t in env.controller._input_tape),
                state_sha256=state_fingerprint(snapshot),frame=snapshot.frame,stage=snapshot.stage,
                stage_frame=snapshot.stage_frame,phase=phase,lives=snapshot.lives,bombs=snapshot.bombs,
                phase_state=tracker.state_dict(),
                power=snapshot.power,playable=bool(snapshot.player.state==1 and not snapshot.dialogue),
                attacks=[r['ecl_sub_name'] for r in roles.get('midbosses',[])+roles.get('main_bosses',[])],
                reward_tracking=dict(schema='thrl-prefix-reward-state-v1',cfg_bombs=env.reward_model.cfg_bombs,
                    rem_bombs_internal=env.reward_model.previous.rem_bombs_internal,
                    **{k:events[k] for k in ('misses','bombs_used','point_items')}))
            # Stage-frame0 is a loading seam, not a stable replay landing.
            # Exclude it entirely, including from recurrent warm-up history.
            if snapshot.stage_frame>=4:records.append(row)
            if i%64==0:on_progress(dict(stage=snapshot.stage,stage_frame=snapshot.stage_frame,
                phase=phase,decisions=i,pid=env.pid,wall_seconds=time.monotonic()-started))
            if (snapshot.stage>max_stage and snapshot.stage_frame>=4) or i==max_decisions:break
            action=int(native36_to_source19(teacher.probabilities(snapshot)).argmax())
            _,_,terminated,truncated,info=env.step(action)
            if terminated or truncated:
                trace['capture_end']=info['episode_summary']
                trace['native_six_stage_clear']=info.get('lunatic_six_stage_no_continue',False)
                break
        trace['raw_input_segments']=[list(row) for row in env.controller.end_input_tape()]
        # The final terminal action cannot be replayed past a dead input hook;
        # retain it in capture_end, and use the last playable checkpoint tape.
        from .thrl_course_store import slice_tape
        trace['raw_input_segments']=slice_tape(trace['raw_input_segments'],0,records[-1]['ticks'])
        atomic_write_json(output/'captured.json',trace)
    finally:
        env.close()
    return trace


def verify_trajectory(trace,executable,profile,output,*,check_stop=lambda:None,on_progress=lambda row:None,
                      controller_observer=None,refresh_phases=False):
    from .thrl_native_events import HiddenTHRLDialogueGeometryController
    if refresh_phases:
        from copy import deepcopy
        trace=deepcopy(trace)
    output=Path(output);receipts=[]
    entry=trajectory_entry(trace,len(trace['records'])-1)
    for repeat in range(2):
        check_stop()
        controller=HiddenTHRLDialogueGeometryController(executable,profile,
            private_appdata=output/f'verify{repeat}/userdata',headless=True,binary_reader=True,
            native_seed=trace['native_seed'],timeout=25,worker_name=f'course-verify-{trace["native_seed"]}-{repeat}')
        if controller_observer is not None:controller=controller_observer(controller)
        started=time.monotonic();checked=0;tracker=PhaseTracker()
        def observed(snapshot,row):
            nonlocal checked
            phase=tracker.observe(snapshot);phase_state=tracker.state_dict()
            if refresh_phases and repeat==0:
                row.update(phase=phase,phase_state=phase_state)
            elif phase!=row['phase'] or (row.get('phase_state') is not None and row['phase_state']!=phase_state):
                raise ValueError('Native course phase history did not reproduce')
            checked+=1
            if checked%128==0:
                check_stop();on_progress(dict(repeat=repeat,verified_checkpoints=checked,pid=controller.pid))
        try:
            controller.start();controller.ensure_run(3)
            replay_entry(controller,entry,checkpoints=trace['records'],on_checkpoint=observed)
            receipts.append(dict(repeat=repeat,pid=controller.pid,checkpoints=checked,
                                 wall_seconds=time.monotonic()-started,landing_state_sha256=state_fingerprint(controller.read())))
        finally:controller.close()
    trace['verified_independent_resets']=len(receipts);trace['verification_receipts']=receipts
    validate_trace(trace)
    atomic_write_json(output/'verified.json',trace)
    return trace
